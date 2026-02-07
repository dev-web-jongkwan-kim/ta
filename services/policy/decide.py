from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from packages.common.config import get_settings


@dataclass
class PolicyConfig:
    ev_min: float
    q05_min: float
    mae_max: float
    max_positions: int


# 필터 임계값
EXTREME_PUMP_THRESHOLD = 0.015   # 1시간 1.5% 이상 상승
EXTREME_DUMP_THRESHOLD = -0.015  # 1시간 1.5% 이상 하락
HIGH_VOLATILITY_THRESHOLD = 90   # ATR 상위 10%
HIGH_VOL_EV_MIN = 0.002          # 고변동성에서 EV 기준
MAX_CONSECUTIVE_LOSSES = 3       # 연패 임계값
RECOVERY_EV_MIN = 0.003          # 연패 후 EV 기준


def _estimate_costs(e_hold_min: float) -> float:
    settings = get_settings()
    fee = settings.taker_fee_rate * 2
    slippage = settings.slippage_k * 0.0001 * 2
    funding = 0.0
    if e_hold_min:
        funding = 0.0
    return fee + slippage + funding


def _market_direction_filter(
    decision: str,
    features: Optional[Dict[str, float]]
) -> Tuple[bool, Optional[str]]:
    """극단적 시장 방향에서 역방향 거래 차단"""
    if features is None:
        return True, None

    btc_ret_60 = features.get("btc_ret_60", 0.0)

    # 급등장에서 SHORT 차단
    if decision == "SHORT" and btc_ret_60 > EXTREME_PUMP_THRESHOLD:
        return False, "EXTREME_PUMP"

    # 급락장에서 LONG 차단
    if decision == "LONG" and btc_ret_60 < EXTREME_DUMP_THRESHOLD:
        return False, "EXTREME_DUMP"

    return True, None


def _volatility_filter(
    decision: str,
    best_ev: float,
    features: Optional[Dict[str, float]]
) -> Tuple[bool, Optional[str]]:
    """고변동성에서 EV 기준 상향"""
    if features is None:
        return True, None

    atr_percentile = features.get("atr_percentile", 50.0)

    if atr_percentile > HIGH_VOLATILITY_THRESHOLD:
        if best_ev < HIGH_VOL_EV_MIN:
            return False, "HIGH_VOL_LOW_EV"

    return True, None


def _consecutive_loss_filter(
    best_ev: float,
    consecutive_losses: int
) -> Tuple[bool, Optional[str]]:
    """연속 손실 후 보수적 진입"""
    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
        if best_ev < RECOVERY_EV_MIN:
            return False, "RECOVERY_MODE"

    return True, None


def decide(
    symbol: str,
    preds: Dict[str, float],
    state: Dict[str, Any],
    cfg: PolicyConfig,
    features: Optional[Dict[str, float]] = None,
    consecutive_losses: int = 0,
    current_equity: Optional[float] = None,
) -> Dict[str, Any]:
    reasons: List[str] = []

    ev_long = preds.get("er_long", 0.0) - _estimate_costs(preds.get("e_hold_long", 0))
    ev_short = preds.get("er_short", 0.0) - _estimate_costs(preds.get("e_hold_short", 0))

    best = max(ev_long, ev_short, 0.0)
    decision = "FLAT"
    if best == ev_long and ev_long > 0:
        decision = "LONG"
    elif best == ev_short and ev_short > 0:
        decision = "SHORT"

    # 기존 필터
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

    # 새로운 필터들 (decision이 FLAT이 아닐 때만)
    if decision != "FLAT":
        # 필터 1: 시장 방향 필터
        ok, reason = _market_direction_filter(decision, features)
        if not ok and reason:
            reasons.append(reason)

        # 필터 2: 변동성 필터
        ok, reason = _volatility_filter(decision, best, features)
        if not ok and reason:
            reasons.append(reason)

        # 필터 3: 연속 손실 필터
        ok, reason = _consecutive_loss_filter(best, consecutive_losses)
        if not ok and reason:
            reasons.append(reason)

    settings = get_settings()
    leverage = settings.leverage

    # 복리 계산: 현재 자본의 10%를 포지션 사이즈로 사용
    if current_equity is not None and current_equity > 0:
        position_size = current_equity * 0.10  # 자본의 10%
    else:
        position_size = settings.initial_capital * 0.10  # 기본값: 초기자본의 10%

    size_notional = position_size * leverage

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
