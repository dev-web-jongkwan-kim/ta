from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from packages.common.bus import RedisBus
from packages.common.db import execute, fetch_all
from packages.common.portfolio import get_portfolio_metrics
from packages.common.runtime import get_collector_ok, get_mode, get_userstream_ok, set_mode
from packages.common.types import StatusDTO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status", response_model=StatusDTO)
def status() -> StatusDTO:
    latest_candle = fetch_all("SELECT max(ts) FROM candles_1m")
    latest_feature = fetch_all("SELECT max(ts) FROM features_1m")
    latest_signal = fetch_all("SELECT max(ts) FROM signals")
    exposures = get_portfolio_metrics()
    last_signal = latest_signal[0][0] if latest_signal and latest_signal[0][0] else None
    return StatusDTO(
        collector_ok=get_collector_ok(),
        userstream_ok=get_userstream_ok(),
        latest_candle_ts=str(latest_candle[0][0]) if latest_candle and latest_candle[0][0] else None,
        latest_feature_ts=str(latest_feature[0][0]) if latest_feature and latest_feature[0][0] else None,
        latest_signal_ts=str(last_signal) if last_signal else None,
        queue_lag={},
        latency_ms={},
        exposures=exposures,
        mode=get_mode(),
    )


@app.get("/api/universe")
def universe() -> List[str]:
    rows = fetch_all("SELECT symbol FROM instruments ORDER BY symbol")
    return [r[0] for r in rows]


@app.get("/api/account/latest")
def account_latest() -> Dict[str, Any]:
    rows = fetch_all("SELECT * FROM account_snapshots ORDER BY ts DESC LIMIT 1")
    if not rows:
        return {}
    columns = ["ts", "equity", "wallet_balance", "unrealized_pnl", "available_margin", "used_margin", "margin_ratio"]
    return dict(zip(columns, rows[0]))


@app.get("/api/positions/latest")
def positions_latest() -> List[Dict[str, Any]]:
    rows = fetch_all("SELECT * FROM positions ORDER BY ts DESC LIMIT 200")
    columns = [
        "symbol",
        "ts",
        "side",
        "amt",
        "entry_price",
        "mark_price",
        "leverage",
        "margin_type",
        "liquidation_price",
        "notional",
        "unrealized_pnl",
        "trade_group_id",
        "event_type",
    ]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/orders")
def orders(from_ts: Optional[str] = None, to_ts: Optional[str] = None) -> List[Dict[str, Any]]:
    query = "SELECT * FROM orders"
    params: List[Any] = []
    if from_ts and to_ts:
        query += " WHERE created_at BETWEEN %s AND %s"
        params = [from_ts, to_ts]
    query += " ORDER BY created_at DESC LIMIT 500"
    rows = fetch_all(query, tuple(params) if params else None)
    columns = [
        "order_id",
        "client_order_id",
        "symbol",
        "side",
        "type",
        "status",
        "reduce_only",
        "price",
        "stop_price",
        "qty",
        "created_at",
        "updated_at",
        "trade_group_id",
    ]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/signals/latest")
def signals_latest(sort: str = "ev", limit: int = 50) -> List[Dict[str, Any]]:
    rows = fetch_all("SELECT * FROM signals ORDER BY ts DESC LIMIT %s", (limit,))
    columns = [
        "symbol",
        "ts",
        "model_id",
        "ev_long",
        "ev_short",
        "er_long",
        "er_short",
        "q05_long",
        "q05_short",
        "e_mae_long",
        "e_mae_short",
        "e_hold_long_min",
        "e_hold_short_min",
        "decision",
        "size_notional",
        "leverage",
        "sl_price",
        "tp_price",
        "block_reason_codes",
        "explain",
        "trade_group_id",
    ]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/symbol/{symbol}/series")
def symbol_series(symbol: str, from_ts: Optional[str] = None, to_ts: Optional[str] = None) -> Dict[str, Any]:
    params: List[Any] = [symbol]
    query = "SELECT * FROM candles_1m WHERE symbol = %s"
    if from_ts and to_ts:
        query += " AND ts BETWEEN %s AND %s"
        params += [from_ts, to_ts]
    query += " ORDER BY ts DESC LIMIT 500"
    candles = fetch_all(query, tuple(params))
    signals = fetch_all("SELECT * FROM signals WHERE symbol=%s ORDER BY ts DESC LIMIT 200", (symbol,))
    return {"candles": candles, "signals": signals}


@app.get("/api/training/jobs")
def training_jobs() -> List[Dict[str, Any]]:
    rows = fetch_all(
        "SELECT job_id, created_at, started_at, ended_at, status, progress, config, report_uri, report_json, error FROM training_jobs ORDER BY created_at DESC LIMIT 200"
    )
    columns = [
        "job_id",
        "created_at",
        "started_at",
        "ended_at",
        "status",
        "progress",
        "config",
        "report_uri",
        "report_json",
        "error",
    ]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/models")
def models() -> List[Dict[str, Any]]:
    rows = fetch_all("SELECT * FROM models ORDER BY created_at DESC LIMIT 200")
    columns = [
        "model_id",
        "created_at",
        "algo",
        "feature_schema_version",
        "label_spec_hash",
        "train_start",
        "train_end",
        "metrics",
        "artifact_uri",
        "is_production",
    ]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/models/{model_id}")
def model(model_id: str) -> Dict[str, Any]:
    rows = fetch_all("SELECT * FROM models WHERE model_id=%s", (model_id,))
    if not rows:
        return {}
    columns = [
        "model_id",
        "created_at",
        "algo",
        "feature_schema_version",
        "label_spec_hash",
        "train_start",
        "train_end",
        "metrics",
        "artifact_uri",
        "is_production",
    ]
    return dict(zip(columns, rows[0]))


@app.post("/api/models/{model_id}/promote")
def promote(model_id: str) -> Dict[str, Any]:
    execute("UPDATE models SET is_production=false")
    execute("UPDATE models SET is_production=true WHERE model_id=%s", (model_id,))
    return {"status": "ok"}


@app.get("/api/data-quality/summary")
def data_quality_summary() -> List[Dict[str, Any]]:
    rows = fetch_all("SELECT * FROM drift_metrics ORDER BY ts DESC LIMIT 200")
    columns = ["ts", "symbol", "schema_version", "psi", "missing_rate", "latency_ms", "outlier_count"]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/risk/state")
def risk_state() -> List[Dict[str, Any]]:
    rows = fetch_all("SELECT * FROM risk_events ORDER BY ts DESC LIMIT 200")
    columns = ["ts", "event_type", "symbol", "severity", "message", "details"]
    return [dict(zip(columns, r)) for r in rows]


@app.post("/api/trading/toggle")
def trading_toggle(mode: str) -> Dict[str, Any]:
    set_mode(mode)
    return {"status": "ok", "mode": mode}


@app.post("/api/settings")
def settings_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok", "payload": payload}


@app.get("/api/settings")
def settings_get() -> Dict[str, Any]:
    """Get current settings including filter thresholds"""
    # Default thresholds - could be stored in DB in future
    return {
        "ev_min": 0.0,
        "q05_min": -0.002,
        "mae_max": 0.01,
        "max_positions": 5,
        "daily_loss_limit": 0.02,
    }


# RL API Endpoints
@app.get("/api/rl/models")
def rl_models() -> List[Dict[str, Any]]:
    """List all RL models"""
    rows = fetch_all(
        "SELECT model_id, symbol, algorithm, train_start, train_end, metrics, model_path, is_production, created_at "
        "FROM rl_models ORDER BY created_at DESC LIMIT 100"
    )
    columns = ["model_id", "symbol", "algorithm", "train_start", "train_end", "metrics", "model_path", "is_production", "created_at"]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/rl/models/{model_id}")
def rl_model_detail(model_id: str) -> Dict[str, Any]:
    """Get RL model details"""
    rows = fetch_all(
        "SELECT model_id, symbol, algorithm, train_start, train_end, metrics, model_path, is_production, created_at "
        "FROM rl_models WHERE model_id = %s",
        (model_id,)
    )
    if not rows:
        return {}
    columns = ["model_id", "symbol", "algorithm", "train_start", "train_end", "metrics", "model_path", "is_production", "created_at"]
    return dict(zip(columns, rows[0]))


@app.post("/api/rl/models/{model_id}/promote")
def rl_promote(model_id: str) -> Dict[str, Any]:
    """Promote RL model to production"""
    # Get the symbol for this model
    rows = fetch_all("SELECT symbol FROM rl_models WHERE model_id = %s", (model_id,))
    if not rows:
        return {"status": "error", "message": "Model not found"}

    symbol = rows[0][0]
    # Clear production flag for this symbol
    execute("UPDATE rl_models SET is_production = false WHERE symbol = %s", (symbol,))
    # Set this model as production
    execute("UPDATE rl_models SET is_production = true WHERE model_id = %s", (model_id,))
    return {"status": "ok"}


@app.get("/api/rl/decisions")
def rl_decisions(
    symbol: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Get RL agent decisions"""
    query = """
        SELECT id, ts, symbol, model_id, action, action_name, confidence,
               action_probs, value_estimate, model_predictions, position_before,
               position_after, executed, pnl_result, created_at
        FROM rl_decisions
        WHERE 1=1
    """
    params: List[Any] = []

    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)
    if model_id:
        query += " AND model_id = %s"
        params.append(model_id)

    query += " ORDER BY ts DESC LIMIT %s"
    params.append(limit)

    rows = fetch_all(query, tuple(params))
    columns = [
        "id", "ts", "symbol", "model_id", "action", "action_name", "confidence",
        "action_probs", "value_estimate", "model_predictions", "position_before",
        "position_after", "executed", "pnl_result", "created_at"
    ]
    return [dict(zip(columns, r)) for r in rows]


@app.get("/api/rl/decisions/stats")
def rl_decisions_stats(symbol: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
    """Get RL decisions statistics"""
    query = """
        SELECT
            action_name,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence,
            SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed_count,
            AVG(pnl_result) FILTER (WHERE pnl_result IS NOT NULL) as avg_pnl
        FROM rl_decisions
        WHERE ts > NOW() - INTERVAL '%s hours'
    """
    params: List[Any] = [hours]

    if symbol:
        query += " AND symbol = %s"
        params.append(symbol)

    query += " GROUP BY action_name"

    rows = fetch_all(query, tuple(params))

    stats = {}
    for row in rows:
        action_name = row[0]
        stats[action_name] = {
            "count": row[1],
            "avg_confidence": row[2],
            "executed_count": row[3],
            "avg_pnl": row[4],
        }

    # Total counts
    total_query = """
        SELECT COUNT(*), SUM(CASE WHEN executed THEN 1 ELSE 0 END)
        FROM rl_decisions WHERE ts > NOW() - INTERVAL '%s hours'
    """
    total_params: List[Any] = [hours]
    if symbol:
        total_query += " AND symbol = %s"
        total_params.append(symbol)

    total_rows = fetch_all(total_query, tuple(total_params))

    return {
        "period_hours": hours,
        "symbol": symbol,
        "total_decisions": total_rows[0][0] if total_rows else 0,
        "total_executed": total_rows[0][1] if total_rows else 0,
        "by_action": stats,
    }


@app.websocket("/ws/{channel}")
async def ws_channel(ws: WebSocket, channel: str) -> None:
    await ws.accept()
    bus = RedisBus()
    last_id = "0-0"
    while True:
        entries = bus.read(channel, last_id=last_id, count=100, block_ms=1000)
        for entry_id, fields in entries:
            last_id = entry_id
            data = fields.get("data")
            if data:
                await ws.send_text(data)
