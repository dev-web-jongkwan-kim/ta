from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PositionState:
    symbol: str
    side: int
    notional: float
    entry_price: float
    sl_price: float
    tp_price: float
    open_ts: str


@dataclass
class PortfolioState:
    equity: float
    used_margin: float
    positions: Dict[str, PositionState]
    daily_pnl: float
    consecutive_losses: int


def run_walkforward_backtest() -> Dict[str, float]:
    """Placeholder backtest metrics for v1."""
    return {
        "profit_factor": 1.0,
        "max_drawdown": 0.0,
        "expectancy": 0.0,
        "tail_loss": 0.0,
        "turnover": 0.0,
        "fees": 0.0,
        "slippage": 0.0,
        "funding": 0.0,
    }
