from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from packages.common.portfolio import get_portfolio_metrics


@dataclass
class RiskConfig:
    max_used_margin_pct: float
    daily_loss_limit_pct: float
    max_positions: int
    max_total_notional_pct: float
    max_directional_notional_pct: float
    missing_alert_rate: float
    missing_block_rate: float


def check_risk(action: Dict[str, Any], portfolio: Dict[str, Any], account: Dict[str, Any], cfg: RiskConfig) -> List[str]:
    reasons: List[str] = []
    equity = account.get("equity", 0.0) or 0.0
    used_margin = account.get("used_margin", 0.0) or 0.0
    margin_ratio = used_margin / equity if equity else 0.0
    if margin_ratio > cfg.max_used_margin_pct:
        reasons.append("MARGIN_LIMIT")

    metrics = get_portfolio_metrics()
    if equity and metrics["daily_loss"] >= cfg.daily_loss_limit_pct * equity:
        reasons.append("DAILY_STOP")

    if metrics["open_positions"] >= cfg.max_positions:
        reasons.append("MAX_POSITIONS")

    total_notional = metrics["total_notional"]
    if equity and total_notional > cfg.max_total_notional_pct * equity:
        reasons.append("MAX_EXPOSURE")

    directional = max(metrics["long_notional"], metrics["short_notional"])
    if equity and directional > cfg.max_directional_notional_pct * equity:
        reasons.append("MAX_DIRECTIONAL_EXPOSURE")

    if portfolio.get("userstream_ok") is False:
        reasons.append("USERSTREAM_DOWN")

    if portfolio.get("data_stale") is True:
        reasons.append("DATA_STALE")

    if portfolio.get("order_failure_spike") is True:
        reasons.append("ORDER_FAILURE_SPIKE")

    missing_rate = portfolio.get("missing_rate", 0.0)
    if missing_rate >= cfg.missing_block_rate:
        reasons.append("MISSING_BLOCK")
    elif missing_rate >= cfg.missing_alert_rate:
        reasons.append("MISSING_DATA")

    drift_status = portfolio.get("drift_status", "ok")
    if drift_status == "block":
        reasons.append("DRIFT_BLOCK")
    elif drift_status == "alert":
        reasons.append("DRIFT_ALERT")

    return reasons
