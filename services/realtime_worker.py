from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, execute, fetch_all
from packages.common.runtime import get_mode, set_collector_ok, get_drift_status, get_drift_metrics
from packages.common.risk import log_risk_event as log_risk_event_base
from packages.common.portfolio import get_portfolio_metrics
from packages.common.symbol_map import to_internal
from packages.common.time import ms_to_dt
from packages.common.trade_group import build_trade_group_id
from packages.common.bus import RedisBus
from services.collector.buffer import IngestBuffer
from services.collector.market_data import (
    fetch_klines,
    fetch_premium_index,
    fetch_open_interest,
    fetch_long_short_ratio,
    fetch_top_long_short_ratio,
    fetch_taker_buy_sell_ratio,
)
from services.features.compute import compute_features_for_symbol
from services.inference.predictor import Predictor
from services.policy.decide import PolicyConfig, decide
from services.registry.ensure_default import ensure_default_model
from services.risk.guard import RiskConfig, check_risk
from services.simulator.fill import FillModel, simulate_trade_path
from services.simulator.trade_writer import TradeGroupWriter
from services.userstream.stream_manager import UserStreamManager
from services.monitoring.drift import compute_missing_rate


def _load_candles(symbol: str, limit: int = 500) -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT symbol, ts, open, high, low, close, volume
        FROM candles_1m
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    return df.sort_values("ts")


def _active_universe(defaults: List[str]) -> List[str]:
    rows = fetch_all("SELECT symbol FROM instruments WHERE status='active' ORDER BY symbol")
    if not rows:
        return defaults
    return [r[0] for r in rows]


def _load_premium(symbol: str, limit: int = 500) -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT symbol, ts, mark_price, index_price, last_price, last_funding_rate, next_funding_time
        FROM premium_index
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(
        rows,
        columns=["symbol", "ts", "mark_price", "index_price", "last_price", "last_funding_rate", "next_funding_time"],
    )
    return df.sort_values("ts")


def _load_open_interest(symbol: str, limit: int = 500) -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT symbol, ts, open_interest, open_interest_value
        FROM open_interest
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open_interest", "open_interest_value"])
    return df.sort_values("ts")


def _load_long_short_ratio(symbol: str, limit: int = 500) -> pd.DataFrame:
    rows = fetch_all(
        """
        SELECT symbol, ts, long_short_ratio, long_account, short_account,
               top_long_short_ratio, top_long_account, top_short_account,
               taker_buy_sell_ratio, taker_buy_vol, taker_sell_vol
        FROM long_short_ratio
        WHERE symbol = %s
        ORDER BY ts DESC
        LIMIT %s
        """,
        (symbol, limit),
    )
    df = pd.DataFrame(
        rows,
        columns=[
            "symbol", "ts", "long_short_ratio", "long_account", "short_account",
            "top_long_short_ratio", "top_long_account", "top_short_account",
            "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol",
        ],
    )
    return df.sort_values("ts")


REASON_MESSAGES = {
    "DATA_STALE": "Latest candle lag exceeded threshold",
    "USERSTREAM_DOWN": "Userstream disconnected",
    "ORDER_FAILURE_SPIKE": "Order failure rate spike",
    "DRIFT_BLOCK": "Drift metrics exceeded blocking threshold",
    "DRIFT_ALERT": "Drift metrics exceeded alert threshold",
    "MISSING_BLOCK": "Missing rate blocked trading",
    "MISSING_DATA": "Missing data alert",
    "MAX_EXPOSURE": "Total exposure above allowed limit",
    "MAX_DIRECTIONAL_EXPOSURE": "Directional exposure above allowed limit",
    "MAX_POSITIONS": "Open positions limit reached",
    "MARGIN_LIMIT": "Used margin ratio exceeded",
    "DAILY_STOP": "Daily loss limit reached",
}
REASON_SEVERITY = {k: 3 for k in REASON_MESSAGES}


def _log_risk_event(event_type: str, message: str, symbol: str | None = None, severity: int = 2, details: Dict | None = None) -> None:
    log_risk_event_base(event_type, message, symbol or "", severity, details)


def _safe_json(payload: Any) -> str:
    def _default(obj: Any) -> Any:
        try:
            return float(obj)
        except Exception:  # noqa: BLE001
            return str(obj)

    return json.dumps(payload, default=_default)


def _coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return value


def main() -> None:
    settings = get_settings()
    bus = RedisBus()
    buffer = IngestBuffer()
    predictor = Predictor()
    policy_cfg = PolicyConfig(settings.ev_min, settings.q05_min, settings.mae_max, settings.max_positions)
    risk_cfg = RiskConfig(
        settings.max_used_margin_pct,
        settings.daily_loss_limit_pct,
        settings.max_positions,
        settings.max_total_notional_pct,
        settings.max_directional_notional_pct,
        settings.drift_missing_alert_rate,
        settings.drift_missing_block_rate,
    )
    fill = FillModel(settings.taker_fee_rate, settings.slippage_k)
    userstream = UserStreamManager()
    userstream.start()

    model_id = ensure_default_model()
    last_ts: Dict[str, datetime] = {}

    while True:
        try:
            premium_snapshot = fetch_premium_index()
            premium_map = {to_internal(item["symbol"], "rest"): item for item in premium_snapshot}
            universe = _active_universe(settings.universe_list())
            for symbol in universe:
                klines = fetch_klines(symbol, "1m", limit=2)
                if not klines:
                    continue
                kline = klines[0]
                close_time = ms_to_dt(int(kline[6]))
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                if last_ts.get(symbol) == close_time:
                    continue
                last_ts[symbol] = close_time
                buffer.add_candle(
                    (
                        symbol,
                        close_time,
                        float(kline[1]),
                        float(kline[2]),
                        float(kline[3]),
                        float(kline[4]),
                        float(kline[5]),
                    )
                )
                premium = premium_map.get(symbol)
                if premium:
                    buffer.add_premium(
                        (
                            symbol,
                            close_time,
                            float(premium.get("markPrice", premium.get("mark_price", 0.0))),
                            float(premium.get("indexPrice", premium.get("index_price", 0.0))),
                            float(premium.get("lastPrice", premium.get("last_price", 0.0))),
                            float(premium.get("lastFundingRate", premium.get("last_funding_rate", 0.0))),
                            ms_to_dt(int(premium.get("nextFundingTime", int(time.time() * 1000)))),
                        )
                    )

                # Open Interest 수집
                try:
                    oi_data = fetch_open_interest(symbol)
                    if oi_data:
                        oi_value = float(oi_data.get("openInterest", 0.0))
                        mark_price = float(premium.get("markPrice", premium.get("mark_price", 1.0))) if premium else 1.0
                        buffer.add_open_interest(
                            (
                                symbol,
                                close_time,
                                oi_value,
                                oi_value * mark_price,  # OI 가치 (USD)
                            )
                        )
                except Exception:
                    pass  # OI 수집 실패는 무시

                # Long/Short 비율 수집
                try:
                    ls_data = fetch_long_short_ratio(symbol, period="5m", limit=1)
                    top_ls_data = fetch_top_long_short_ratio(symbol, period="5m", limit=1)
                    taker_data = fetch_taker_buy_sell_ratio(symbol, period="5m", limit=1)

                    ls = ls_data[0] if ls_data else {}
                    top_ls = top_ls_data[0] if top_ls_data else {}
                    taker = taker_data[0] if taker_data else {}

                    buffer.add_long_short_ratio(
                        (
                            symbol,
                            close_time,
                            float(ls.get("longShortRatio", 1.0)),
                            float(ls.get("longAccount", 0.5)),
                            float(ls.get("shortAccount", 0.5)),
                            float(top_ls.get("longShortRatio", 1.0)),
                            float(top_ls.get("longAccount", 0.5)),
                            float(top_ls.get("shortAccount", 0.5)),
                            float(taker.get("buySellRatio", 1.0)),
                            float(taker.get("buyVol", 0.0)),
                            float(taker.get("sellVol", 0.0)),
                        )
                    )
                except Exception:
                    pass  # L/S 수집 실패는 무시

            if buffer.should_flush():
                buffer.flush()
            set_collector_ok(True)

            for symbol in universe:
                candles = _load_candles(symbol)
                if candles.empty:
                    continue
                premium = _load_premium(symbol)
                btc_candles = _load_candles("BTCUSDT") if symbol != "BTCUSDT" else candles
                open_interest_df = _load_open_interest(symbol)
                long_short_ratio_df = _load_long_short_ratio(symbol)
                feats = compute_features_for_symbol(
                    symbol, candles, premium, btc_candles,
                    open_interest=open_interest_df,
                    long_short_ratio=long_short_ratio_df,
                )
                if feats.empty:
                    continue
                latest = feats.iloc[-1]
                atr_value = float(latest["atr"]) if pd.notna(latest["atr"]) else None
                funding_z = float(latest["funding_z"]) if pd.notna(latest["funding_z"]) else None
                btc_regime = int(latest["btc_regime"]) if pd.notna(latest["btc_regime"]) else None
                features_clean = {k: _coerce_value(v) for k, v in latest["features"].items()}
                bulk_upsert(
                    "features_1m",
                    ["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"],
                    [
                        (
                            latest["symbol"],
                            latest["ts"],
                            int(latest["schema_version"]),
                            _safe_json(features_clean),
                            atr_value,
                            funding_z,
                            btc_regime,
                        )
                    ],
                    conflict_cols=["symbol", "ts"],
                    update_cols=["schema_version", "features", "atr", "funding_z", "btc_regime"],
                )

                preds = predictor.predict(latest["features"])
                last_close = float(candles.iloc[-1]["close"]) if not candles.empty else 0.0
                atr = float(latest["atr"]) if pd.notna(latest["atr"]) else 0.0
                sl_price = last_close - atr * 1.0
                tp_price = last_close + atr * 1.5
                state = {"equity": 10000.0, "sl_price": sl_price, "tp_price": tp_price}
                decision = decide(symbol, preds, state, policy_cfg)
                data_stale = False
                if isinstance(latest["ts"], pd.Timestamp):
                    lag_sec = (datetime.now(timezone.utc) - latest["ts"]).total_seconds()
                    data_stale = lag_sec > settings.data_stale_sec
                missing = compute_missing_rate(pd.DataFrame([latest["features"]]))
                drift_status = get_drift_status()
                drift_metrics = get_drift_metrics()
                missing_rate = drift_metrics.get("missing_rate", missing)
                metrics = get_portfolio_metrics()
                portfolio_state = {
                    "positions": {},
                    "daily_pnl": metrics["daily_pnl"],
                    "total_notional": metrics["total_notional"],
                    "directional_notional": {"long": metrics["long_notional"], "short": metrics["short_notional"]},
                    "open_positions": metrics["open_positions"],
                    "daily_loss": metrics["daily_loss"],
                    "userstream_ok": True,
                    "data_stale": data_stale,
                    "order_failure_spike": False,
                    "drift_status": drift_status,
                    "drift_metrics": drift_metrics,
                    "missing_rate": missing_rate,
                    "drift_latency_ms": drift_metrics.get("latency_ms", 0.0),
                }
                account_state = {"equity": 10000.0, "used_margin": 0.0}
                blocks = check_risk(decision, portfolio_state, account_state, risk_cfg)
                decision["block_reasons"] += blocks
                for reason in blocks:
                    msg = REASON_MESSAGES.get(reason, "Risk block triggered")
                    severity = REASON_SEVERITY.get(reason, 3)
                    _log_risk_event(reason, msg, symbol=symbol, severity=severity)

                trade_group_id = build_trade_group_id(symbol, latest["ts"], decision["decision"])
                signal_row = {
                    "symbol": symbol,
                    "ts": latest["ts"],
                    "model_id": model_id,
                    "ev_long": decision["ev_long"],
                    "ev_short": decision["ev_short"],
                    "er_long": preds.get("er_long"),
                    "er_short": preds.get("er_short"),
                    "q05_long": preds.get("q05_long"),
                    "q05_short": preds.get("q05_short"),
                    "e_mae_long": preds.get("e_mae_long"),
                    "e_mae_short": preds.get("e_mae_short"),
                    "e_hold_long_min": int(preds.get("e_hold_long", 0)),
                    "e_hold_short_min": int(preds.get("e_hold_short", 0)),
                    "decision": decision["decision"],
                    "size_notional": decision.get("size_notional"),
                    "leverage": decision.get("leverage"),
                    "sl_price": decision.get("sl_price"),
                    "tp_price": decision.get("tp_price"),
                    "block_reason_codes": json.dumps(decision.get("block_reasons", [])),
                    "explain": json.dumps({}),
                    "trade_group_id": trade_group_id,
                }

                orders_data: List[Dict[str, Any]] = []
                fills_data: List[Dict[str, Any]] = []
                position_events: List[Dict[str, Any]] = []

                mode = get_mode()
                if mode == "shadow" and decision["decision"] in ("LONG", "SHORT") and not decision["block_reasons"]:
                    sim = simulate_trade_path(
                        candles=candles,
                        premium=premium,
                        entry_ts=str(latest["ts"]),
                        side=1 if decision["decision"] == "LONG" else -1,
                        notional=decision.get("size_notional", 0.0) or 0.0,
                        sl_price=sl_price,
                        tp_price=tp_price,
                        fill=fill,
                    )
                    entry_price = sim.get("entry_price") or 1.0
                    entry_qty = (decision.get("size_notional", 0.0) or 0.0) / entry_price
                    entry_order_id = int(time.time() * 1000)
                    exit_order_id = entry_order_id + 1
                    side_str = "BUY" if decision["decision"] == "LONG" else "SELL"
                    opposite_side = "SELL" if side_str == "BUY" else "BUY"
                    now_ts = datetime.now(timezone.utc)

                    entry_order = {
                        "order_id": entry_order_id,
                        "client_order_id": f"shadow-{signal_row['trade_group_id']}-entry",
                        "symbol": symbol,
                        "side": side_str,
                        "type": "MARKET",
                        "status": "FILLED",
                        "reduce_only": False,
                        "qty": entry_qty,
                        "price": sim.get("entry_price"),
                        "stop_price": None,
                        "created_at": now_ts,
                        "updated_at": now_ts,
                        "trade_group_id": signal_row["trade_group_id"],
                    }
                    exit_order = {
                        "order_id": exit_order_id,
                        "client_order_id": f"shadow-{signal_row['trade_group_id']}-exit",
                        "symbol": symbol,
                        "side": opposite_side,
                        "type": "TAKE_PROFIT_MARKET" if sim.get("exit_reason") == "TP" else "STOP_MARKET",
                        "status": "FILLED",
                        "reduce_only": True,
                        "qty": entry_qty,
                        "price": sim.get("exit_price"),
                        "stop_price": sim.get("exit_price"),
                        "created_at": now_ts,
                        "updated_at": now_ts,
                        "trade_group_id": signal_row["trade_group_id"],
                    }
                    orders_data = [entry_order, exit_order]

                    fills_data = [
                        {
                            "trade_id": entry_order_id * 10,
                            "order_id": entry_order_id,
                            "symbol": symbol,
                            "price": sim.get("entry_price"),
                            "qty": entry_qty,
                            "fee": sim.get("fees", 0.0),
                            "fee_asset": "USDT",
                            "realized_pnl": 0.0,
                            "ts": sim.get("entry_ts"),
                            "trade_group_id": signal_row["trade_group_id"],
                        },
                        {
                            "trade_id": exit_order_id * 10,
                            "order_id": exit_order_id,
                            "symbol": symbol,
                            "price": sim.get("exit_price"),
                            "qty": entry_qty,
                            "fee": sim.get("fees", 0.0),
                            "fee_asset": "USDT",
                            "realized_pnl": sim.get("realized_pnl"),
                            "ts": sim.get("exit_ts"),
                            "trade_group_id": signal_row["trade_group_id"],
                        },
                    ]

                    position_events = [
                        {
                            "trade_group_id": signal_row["trade_group_id"],
                            "symbol": symbol,
                            "ts": sim.get("entry_ts"),
                            "side": side_str,
                            "amt": entry_qty,
                            "entry_price": sim.get("entry_price"),
                            "mark_price": sim.get("entry_price"),
                            "leverage": decision.get("leverage"),
                            "margin_type": "ISOLATED",
                            "liquidation_price": None,
                            "notional": decision.get("size_notional"),
                            "unrealized_pnl": 0.0,
                            "event_type": "OPEN",
                        },
                        {
                            "trade_group_id": signal_row["trade_group_id"],
                            "symbol": symbol,
                            "ts": sim.get("exit_ts"),
                            "side": side_str,
                            "amt": 0.0,
                            "entry_price": sim.get("entry_price"),
                            "mark_price": sim.get("exit_price"),
                            "leverage": decision.get("leverage"),
                            "margin_type": "ISOLATED",
                            "liquidation_price": None,
                            "notional": decision.get("size_notional"),
                            "unrealized_pnl": sim.get("realized_pnl"),
                            "event_type": "CLOSE",
                        },
                    ]

                TradeGroupWriter.persist_trade(signal_row, orders_data, fills_data, position_events)

                bus.publish_json(
                    "signal_updates",
                    {
                        "symbol": symbol,
                        "ts": str(latest["ts"]),
                        "decision": decision["decision"],
                        "ev_long": decision["ev_long"],
                        "ev_short": decision["ev_short"],
                        "block_reasons": decision.get("block_reasons", []),
                    },
                )

            time.sleep(settings.collect_interval_sec)
        except Exception as exc:  # noqa: BLE001
            set_collector_ok(False)
            _log_risk_event("COLLECTOR_ERROR", str(exc), severity=3)
            time.sleep(settings.collect_interval_sec)


if __name__ == "__main__":
    main()
