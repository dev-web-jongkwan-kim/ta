from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StatusDTO(BaseModel):
    collector_ok: bool
    userstream_ok: bool
    latest_candle_ts: Optional[str] = None
    latest_feature_ts: Optional[str] = None
    latest_signal_ts: Optional[str] = None
    queue_lag: Dict[str, float] = {}
    latency_ms: Dict[str, float] = {}
    exposures: Dict[str, float] = {}
    mode: str


class SignalDTO(BaseModel):
    symbol: str
    ts: str
    ev_long: float
    ev_short: float
    er_long: Optional[float] = None
    er_short: Optional[float] = None
    q05_long: Optional[float] = None
    q05_short: Optional[float] = None
    e_mae_long: Optional[float] = None
    e_mae_short: Optional[float] = None
    e_hold_long_min: Optional[int] = None
    e_hold_short_min: Optional[int] = None
    decision: str
    size_notional: Optional[float] = None
    leverage: Optional[int] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    block_reasons: List[str] = []
    explain: Dict[str, Any] = {}


class RiskEventDTO(BaseModel):
    ts: str
    event_type: str
    symbol: Optional[str] = None
    severity: int
    message: str
    details: Dict[str, Any] = {}


class AccountSnapshotDTO(BaseModel):
    ts: str
    equity: float
    wallet_balance: float
    unrealized_pnl: float
    available_margin: float
    used_margin: float
    margin_ratio: float


class PositionDTO(BaseModel):
    symbol: str
    ts: str
    side: str
    amt: float
    entry_price: Optional[float] = None
    mark_price: Optional[float] = None
    leverage: Optional[int] = None
    margin_type: Optional[str] = None
    liquidation_price: Optional[float] = None
    notional: Optional[float] = None
    unrealized_pnl: Optional[float] = None
