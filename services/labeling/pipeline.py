from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd

from packages.common.config import get_settings
from packages.common.db import bulk_upsert, fetch_all
from services.labeling.triple_barrier import LabelSpec, label_direction_vectorized

logger = logging.getLogger(__name__)

Timeframe = Literal["1m", "15m", "1h"]

# SQL Injection 방지를 위한 허용 테이블 목록
_VALID_LABEL_TABLES = frozenset({
    "labels_long_1m", "labels_short_1m",
    "labels_long_15m", "labels_short_15m",
    "labels_long_1h", "labels_short_1h",
})

_VALID_CANDLE_TABLES = frozenset({
    "candles_1m", "candles_15m", "candles_1h",
})

# 타임프레임별 기본 h_bars 설정 (6시간 기준)
_DEFAULT_H_BARS = {
    "1m": 360,   # 360분 = 6시간
    "15m": 24,   # 24 * 15분 = 6시간
    "1h": 6,     # 6시간
}

# 증분 로드 시 lookback 여유분 (타임프레임별 캔들 수)
_INCREMENTAL_LOAD_BUFFER = {
    "1m": 60,
    "15m": 4,
    "1h": 1,
}


@dataclass
class LabelingStats:
    """라벨링 결과 통계"""

    spec_hash: str
    symbols_processed: int
    total_new_labels: int
    total_existing: int
    by_symbol: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def total_skipped(self) -> int:
        """하위 호환성을 위한 별칭"""
        return self.total_existing


@dataclass(frozen=True)
class LabelingConfig:
    k_tp: float = 1.5
    k_sl: float = 1.0
    h_bars: int = 360
    risk_mae_atr: float = 3.0
    timeframe: Timeframe = "1m"

    def spec(self) -> LabelSpec:
        settings = get_settings()
        # 타임프레임에 맞게 h_bars 조정
        adjusted_h_bars = self.h_bars
        if self.h_bars == 360:  # 기본값인 경우 타임프레임에 맞게 조정
            adjusted_h_bars = _DEFAULT_H_BARS.get(self.timeframe, self.h_bars)

        return LabelSpec(
            k_tp=self.k_tp,
            k_sl=self.k_sl,
            h_bars=adjusted_h_bars,
            risk_mae_atr=self.risk_mae_atr,
            fee_rate=settings.taker_fee_rate,
            slippage_k=settings.slippage_k,
        )


def _get_candle_table(timeframe: Timeframe) -> str:
    return f"candles_{timeframe}"


def _get_label_table(direction: str, timeframe: Timeframe) -> str:
    return f"labels_{direction}_{timeframe}"


def _load_candles(symbol: str, after_ts: Optional[datetime] = None, timeframe: Timeframe = "1m") -> pd.DataFrame:
    """캔들 데이터 로드"""
    table = _get_candle_table(timeframe)
    if table not in _VALID_CANDLE_TABLES:
        raise ValueError(f"Invalid candle table: {table}")

    if after_ts is not None:
        rows = fetch_all(
            f"SELECT ts, high, low, close FROM {table} WHERE symbol=%s AND ts > %s ORDER BY ts",
            (symbol, after_ts),
        )
    else:
        rows = fetch_all(
            f"SELECT ts, high, low, close FROM {table} WHERE symbol=%s ORDER BY ts",
            (symbol,),
        )
    return pd.DataFrame(rows, columns=["ts", "high", "low", "close"])


def _load_premium(symbol: str, after_ts: Optional[datetime] = None) -> pd.DataFrame:
    """프리미엄 인덱스 데이터 로드"""
    if after_ts is not None:
        rows = fetch_all(
            "SELECT ts, mark_price, last_funding_rate, next_funding_time FROM premium_index WHERE symbol=%s AND ts > %s ORDER BY ts",
            (symbol, after_ts),
        )
    else:
        rows = fetch_all(
            "SELECT ts, mark_price, last_funding_rate, next_funding_time FROM premium_index WHERE symbol=%s ORDER BY ts",
            (symbol,),
        )
    return pd.DataFrame(rows, columns=["ts", "mark_price", "funding_rate", "next_funding_time"])


def _persist_label_rows(table: str, rows: List[Tuple]) -> None:
    """라벨 행을 DB에 저장"""
    if table not in _VALID_LABEL_TABLES:
        raise ValueError(f"Invalid table name: {table}")

    bulk_upsert(
        table,
        ["symbol", "ts", "spec_hash", "y", "ret_net", "mae", "mfe", "time_to_event_min"],
        rows,
        conflict_cols=["symbol", "ts", "spec_hash"],
        update_cols=["y", "ret_net", "mae", "mfe", "time_to_event_min"],
    )


def _get_max_label_timestamp(symbol: str, spec_hash: str, table: str) -> Optional[datetime]:
    """해당 심볼/스펙의 가장 최근 라벨 타임스탬프 반환"""
    if table not in _VALID_LABEL_TABLES:
        raise ValueError(f"Invalid table name: {table}")

    rows = fetch_all(
        f"SELECT MAX(ts) FROM {table} WHERE symbol = %s AND spec_hash = %s",
        (symbol, spec_hash),
    )
    return rows[0][0] if rows and rows[0][0] else None


def _count_existing_labels(symbol: str, spec_hash: str, table: str) -> int:
    """해당 심볼/스펙의 기존 라벨 개수 반환"""
    if table not in _VALID_LABEL_TABLES:
        raise ValueError(f"Invalid table name: {table}")

    rows = fetch_all(
        f"SELECT COUNT(*) FROM {table} WHERE symbol = %s AND spec_hash = %s",
        (symbol, spec_hash),
    )
    return rows[0][0] if rows else 0


def _count_existing_labels_both(symbol: str, spec_hash: str, timeframe: Timeframe = "1m") -> Tuple[int, int]:
    """해당 심볼/스펙의 long/short 기존 라벨 개수를 단일 쿼리로 반환"""
    long_table = _get_label_table("long", timeframe)
    short_table = _get_label_table("short", timeframe)

    rows = fetch_all(
        f"""
        SELECT
            (SELECT COUNT(*) FROM {long_table} WHERE symbol = %s AND spec_hash = %s) as long_count,
            (SELECT COUNT(*) FROM {short_table} WHERE symbol = %s AND spec_hash = %s) as short_count
        """,
        (symbol, spec_hash, symbol, spec_hash),
    )
    if rows:
        return (rows[0][0] or 0, rows[0][1] or 0)
    return (0, 0)


def label_symbol(symbol: str, spec: LabelSpec, timeframe: Timeframe = "1m") -> None:
    """기존 전체 라벨링 (하위 호환성 유지)"""
    label_symbol_incremental(symbol, spec, force_full=True, timeframe=timeframe)


def label_symbol_incremental(
    symbol: str, spec: LabelSpec, force_full: bool = False, timeframe: Timeframe = "1m"
) -> Dict[str, int]:
    """증분 라벨링 - 새 캔들만 처리"""
    spec_hash = spec.hash()
    h_bars = spec.h_bars
    atr_window = 14

    long_table = _get_label_table("long", timeframe)
    short_table = _get_label_table("short", timeframe)
    load_buffer = _INCREMENTAL_LOAD_BUFFER.get(timeframe, 60)

    # 기존 라벨의 최대 타임스탬프 조회
    if not force_full:
        max_ts_long = _get_max_label_timestamp(symbol, spec_hash, long_table)
        max_ts_short = _get_max_label_timestamp(symbol, spec_hash, short_table)
        if max_ts_long and max_ts_short:
            load_after_ts = min(max_ts_long, max_ts_short)
        else:
            load_after_ts = max_ts_long or max_ts_short
    else:
        max_ts_long = None
        max_ts_short = None
        load_after_ts = None

    # 증분 로드: lookback 포함
    if load_after_ts is not None:
        lookback_candles = h_bars + atr_window + load_buffer
        # 타임프레임별 분 단위 계산
        tf_minutes = {"1m": 1, "15m": 15, "1h": 60}[timeframe]
        lookback_minutes = lookback_candles * tf_minutes
        candle_load_ts = load_after_ts - timedelta(minutes=lookback_minutes)
        candles = _load_candles(symbol, after_ts=candle_load_ts, timeframe=timeframe)
        premium = _load_premium(symbol, after_ts=candle_load_ts)
        existing_long, existing_short = _count_existing_labels_both(symbol, spec_hash, timeframe)
    else:
        candles = _load_candles(symbol, timeframe=timeframe)
        premium = _load_premium(symbol)
        existing_long = 0
        existing_short = 0

    if candles.empty or premium.empty:
        logger.debug(f"{symbol}: no candles or premium data")
        return {"new_long": 0, "new_short": 0, "existing_long": existing_long, "existing_short": existing_short}

    loaded_candles = len(candles)

    # premium 데이터를 candles 타임스탬프에 맞춤
    premium = premium.set_index("ts").reindex(candles["ts"]).ffill().bfill().reset_index()
    atr = candles["high"].rolling(window=14).max() - candles["low"].rolling(window=14).min()

    # Long 라벨링
    data = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=premium["mark_price"].values,
        atr=atr.bfill().values,
        funding_rate=premium["funding_rate"].fillna(0.0).values,
        next_funding_ts=premium["next_funding_time"].ffill().values,
        spec=spec,
        direction=1,
    )

    if max_ts_long is not None:
        new_data = data[data["ts"] > max_ts_long]
    else:
        new_data = data

    rows = [
        (
            r.symbol,
            r.ts,
            r.spec_hash,
            int(r.y),
            float(r.ret_net),
            float(r.mae),
            float(r.mfe),
            int(r.time_to_event_min),
        )
        for r in new_data.itertuples(index=False)
    ]
    if rows:
        _persist_label_rows(long_table, rows)
    new_long = len(rows)

    # Short 라벨링
    data_short = label_direction_vectorized(
        symbol=symbol,
        ts_index=candles["ts"].values,
        high=candles["high"].values,
        low=candles["low"].values,
        close=candles["close"].values,
        mark=premium["mark_price"].values,
        atr=atr.bfill().values,
        funding_rate=premium["funding_rate"].fillna(0.0).values,
        next_funding_ts=premium["next_funding_time"].ffill().values,
        spec=spec,
        direction=-1,
    )

    if max_ts_short is not None:
        new_data_short = data_short[data_short["ts"] > max_ts_short]
    else:
        new_data_short = data_short

    rows_short = [
        (
            r.symbol,
            r.ts,
            r.spec_hash,
            int(r.y),
            float(r.ret_net),
            float(r.mae),
            float(r.mfe),
            int(r.time_to_event_min),
        )
        for r in new_data_short.itertuples(index=False)
    ]
    if rows_short:
        _persist_label_rows(short_table, rows_short)
    new_short = len(rows_short)

    logger.info(
        f"{symbol} ({timeframe}): loaded_candles={loaded_candles}, "
        f"new_long={new_long}, new_short={new_short}, "
        f"existing_long={existing_long}, existing_short={existing_short}"
    )

    return {
        "new_long": new_long,
        "new_short": new_short,
        "existing_long": existing_long,
        "existing_short": existing_short,
    }


@dataclass
class SymbolError:
    """심볼별 에러 정보"""
    symbol: str
    error: str


def run_labeling(
    config: Optional[LabelingConfig] = None,
    symbols: Optional[Iterable[str]] = None,
    force_full: bool = False,
    timeframe: Timeframe = "1m",
) -> LabelingStats:
    """라벨링 실행"""
    settings = get_settings()
    cfg = config or LabelingConfig(timeframe=timeframe)
    spec = cfg.spec()
    spec_hash = spec.hash()
    universe = list(symbols or settings.universe_list())

    logger.info(
        f"Starting labeling ({timeframe}): spec_hash={spec_hash}, "
        f"symbols={len(universe)}, force_full={force_full}"
    )

    stats = LabelingStats(
        spec_hash=spec_hash,
        symbols_processed=len(universe),
        total_new_labels=0,
        total_existing=0,
        by_symbol={},
    )
    errors: List[SymbolError] = []

    for i, symbol in enumerate(universe, 1):
        logger.debug(f"Processing {symbol} ({i}/{len(universe)})")
        try:
            result = label_symbol_incremental(symbol, spec, force_full=force_full, timeframe=cfg.timeframe)
            stats.by_symbol[symbol] = result
            stats.total_new_labels += result["new_long"] + result["new_short"]
            stats.total_existing += result["existing_long"] + result["existing_short"]
        except Exception as e:
            logger.error(f"{symbol}: labeling failed - {e}")
            errors.append(SymbolError(symbol=symbol, error=str(e)))
            stats.by_symbol[symbol] = {
                "new_long": 0,
                "new_short": 0,
                "existing_long": 0,
                "existing_short": 0,
                "error": str(e),
            }

    if errors:
        logger.warning(f"Labeling completed with {len(errors)} errors: {[e.symbol for e in errors]}")
    else:
        logger.info(
            f"Labeling complete ({timeframe}): new_labels={stats.total_new_labels:,}, "
            f"existing={stats.total_skipped:,}"
        )

    return stats
