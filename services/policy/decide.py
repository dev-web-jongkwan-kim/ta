from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from packages.common.config import get_settings


@dataclass
class PolicyConfig:
    ev_min: float
    q05_min: float
    mae_max: float
    max_positions: int


def _estimate_costs(e_hold_min: float) -> float:
    settings = get_settings()
    fee = settings.taker_fee_rate * 2
    slippage = settings.slippage_k * 0.0001 * 2
    funding = 0.0
    if e_hold_min:
        funding = 0.0
    return fee + slippage + funding


def decide(symbol: str, preds: Dict[str, float], state: Dict[str, Any], cfg: PolicyConfig) -> Dict[str, Any]:
    reasons: List[str] = []

    ev_long = preds.get("er_long", 0.0) - _estimate_costs(preds.get("e_hold_long", 0))
    ev_short = preds.get("er_short", 0.0) - _estimate_costs(preds.get("e_hold_short", 0))

    best = max(ev_long, ev_short, 0.0)
    decision = "FLAT"
    if best == ev_long:
        decision = "LONG"
    elif best == ev_short:
        decision = "SHORT"

    if best <= cfg.ev_min:
        reasons.append("EV_MIN")
    if decision == "LONG" and preds.get("q05_long", 0.0) < cfg.q05_min:
        reasons.append("Q05_MIN")
    if decision == "SHORT" and preds.get("q05_short", 0.0) < cfg.q05_min:
        reasons.append("Q05_MIN")
    if decision == "LONG" and preds.get("e_mae_long", 0.0) > cfg.mae_max:
        reasons.append("MAE_MAX")
    if decision == "SHORT" and preds.get("e_mae_short", 0.0) > cfg.mae_max:
        reasons.append("MAE_MAX")

    size_notional = state.get("equity", 0) * 0.05
    leverage = 3

    return {
        "symbol": symbol,
        "decision": decision,
        "ev_long": ev_long,
        "ev_short": ev_short,
        "size_notional": size_notional,
        "leverage": leverage,
        "sl_price": state.get("sl_price"),
        "tp_price": state.get("tp_price"),
        "block_reasons": reasons,
    }
