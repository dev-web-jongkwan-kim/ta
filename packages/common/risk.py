from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from packages.common.db import execute


def log_risk_event(
    event_type: str,
    message: str,
    symbol: str = "",
    severity: int = 2,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    execute(
        """
        INSERT INTO risk_events (ts, event_type, symbol, severity, message, details)
        VALUES (now(), %s, %s, %s, %s, %s)
        """,
        (event_type, symbol, severity, message, json.dumps(details or {})),
    )


# ============================================================================
# Kelly Criterion Position Sizing
# ============================================================================

@dataclass
class KellyResult:
    """Kelly criterion 계산 결과"""
    kelly_fraction: float  # 원래 Kelly 비율
    half_kelly: float  # Half Kelly (실제 사용 권장)
    position_pct: float  # 최종 포지션 비율 (caps 적용 후)


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.5,  # Half Kelly (default)
    max_position_pct: float = 0.25,  # 최대 25%
    min_position_pct: float = 0.01,  # 최소 1%
) -> KellyResult:
    """Kelly Criterion 기반 포지션 크기 계산

    Args:
        win_rate: 승률 (0-1)
        avg_win: 평균 이익 (양수)
        avg_loss: 평균 손실 (양수로 입력)
        kelly_fraction: Kelly 비율 (0.5 = Half Kelly)
        max_position_pct: 최대 포지션 비율
        min_position_pct: 최소 포지션 비율

    Returns:
        KellyResult with kelly_fraction, half_kelly, position_pct
    """
    if avg_loss <= 0 or avg_win <= 0:
        return KellyResult(0, 0, 0)

    if win_rate <= 0 or win_rate >= 1:
        return KellyResult(0, 0, 0)

    # Kelly formula: f* = (p * b - q) / b
    # p = win_rate, q = 1 - p, b = avg_win / avg_loss
    b = avg_win / avg_loss
    q = 1 - win_rate
    full_kelly = (win_rate * b - q) / b

    if full_kelly <= 0:
        return KellyResult(full_kelly, 0, 0)

    # Half Kelly
    half_kelly = full_kelly * kelly_fraction

    # Position cap 적용
    position_pct = max(min_position_pct, min(half_kelly, max_position_pct))

    return KellyResult(
        kelly_fraction=full_kelly,
        half_kelly=half_kelly,
        position_pct=position_pct
    )


def calculate_kelly_from_trades(
    returns: pd.Series,
    kelly_fraction: float = 0.5,
    max_position_pct: float = 0.25,
) -> KellyResult:
    """거래 수익률 시리즈에서 Kelly 계산

    Args:
        returns: 거래별 수익률 시리즈
        kelly_fraction: Kelly 비율 (0.5 = Half Kelly)
        max_position_pct: 최대 포지션 비율

    Returns:
        KellyResult
    """
    if len(returns) < 10:
        return KellyResult(0, 0, 0)

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return KellyResult(0, 0, 0)

    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    return calculate_kelly_fraction(
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        kelly_fraction=kelly_fraction,
        max_position_pct=max_position_pct,
    )


# ============================================================================
# ATR-Based Position Sizing
# ============================================================================

def calculate_atr_position_size(
    capital: float,
    entry_price: float,
    atr: float,
    risk_per_trade: float = 0.02,  # 2% risk per trade
    atr_multiplier: float = 2.0,  # SL at 2x ATR
) -> Dict[str, float]:
    """ATR 기반 포지션 크기 계산

    Args:
        capital: 총 자본금
        entry_price: 진입 가격
        atr: ATR 값
        risk_per_trade: 거래당 리스크 비율 (기본 2%)
        atr_multiplier: SL ATR 배수 (기본 2x)

    Returns:
        Dict with position_size, position_value, stop_loss_price, risk_amount
    """
    if atr <= 0 or entry_price <= 0 or capital <= 0:
        return {
            "position_size": 0,
            "position_value": 0,
            "stop_loss_price": 0,
            "risk_amount": 0,
        }

    risk_amount = capital * risk_per_trade
    stop_distance = atr * atr_multiplier
    stop_loss_price = entry_price - stop_distance  # Long 기준

    # 포지션 가치 = 리스크 금액 / (SL 거리 / 진입가)
    position_value = risk_amount / (stop_distance / entry_price)
    position_size = position_value / entry_price

    return {
        "position_size": position_size,
        "position_value": position_value,
        "stop_loss_price": stop_loss_price,
        "risk_amount": risk_amount,
    }


# ============================================================================
# Correlation-Based Position Filter
# ============================================================================

def should_open_position(
    new_symbol: str,
    current_positions: List[str],
    correlation_matrix: pd.DataFrame,
    max_correlation: float = 0.7,
) -> Dict[str, Any]:
    """상관관계 기반 포지션 진입 필터

    Args:
        new_symbol: 진입 예정 심볼
        current_positions: 현재 보유 중인 심볼 리스트
        correlation_matrix: 심볼간 상관관계 매트릭스
        max_correlation: 최대 허용 상관관계

    Returns:
        Dict with allowed, reason, correlated_symbols
    """
    if not current_positions:
        return {"allowed": True, "reason": "No existing positions", "correlated_symbols": []}

    if new_symbol not in correlation_matrix.index:
        return {"allowed": True, "reason": "Symbol not in correlation matrix", "correlated_symbols": []}

    correlated_symbols = []
    for existing in current_positions:
        if existing in correlation_matrix.columns:
            corr = correlation_matrix.loc[new_symbol, existing]
            if abs(corr) > max_correlation:
                correlated_symbols.append((existing, corr))

    if correlated_symbols:
        return {
            "allowed": False,
            "reason": f"High correlation with existing positions",
            "correlated_symbols": correlated_symbols,
        }

    return {"allowed": True, "reason": "Correlation check passed", "correlated_symbols": []}


def build_correlation_matrix(
    returns_df: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """수익률 DataFrame으로부터 상관관계 매트릭스 생성

    Args:
        returns_df: 심볼별 수익률 DataFrame (columns = symbols)
        window: 롤링 윈도우 (기본 60)

    Returns:
        상관관계 매트릭스 DataFrame
    """
    return returns_df.tail(window).corr()


# ============================================================================
# Dynamic TP/SL Calculation
# ============================================================================

def get_dynamic_barriers(
    atr_percentile: float,
    base_k_tp: float = 1.5,
    base_k_sl: float = 1.0,
) -> Dict[str, float]:
    """ATR percentile 기반 동적 TP/SL 배수 계산

    Args:
        atr_percentile: ATR 백분위 (0-100)
        base_k_tp: 기본 TP 배수
        base_k_sl: 기본 SL 배수

    Returns:
        Dict with k_tp, k_sl, regime
    """
    if atr_percentile > 80:
        # 고변동성: 넓은 배리어
        k_tp = base_k_tp * 1.5
        k_sl = base_k_sl * 1.2
        regime = "high_volatility"
    elif atr_percentile < 20:
        # 저변동성: 좁은 배리어
        k_tp = base_k_tp * 0.7
        k_sl = base_k_sl * 0.6
        regime = "low_volatility"
    else:
        # 일반
        k_tp = base_k_tp
        k_sl = base_k_sl
        regime = "normal"

    return {
        "k_tp": k_tp,
        "k_sl": k_sl,
        "regime": regime,
    }


# ============================================================================
# Integrated Position Sizing
# ============================================================================

def calculate_position_size(
    capital: float,
    entry_price: float,
    atr: float,
    win_rate: float = 0.5,
    avg_win: float = 0.01,
    avg_loss: float = 0.01,
    use_kelly: bool = True,
    use_atr: bool = True,
    max_position_pct: float = 0.25,
) -> Dict[str, Any]:
    """통합 포지션 크기 계산

    Kelly와 ATR 기반 사이징 중 더 보수적인 값 선택

    Args:
        capital: 총 자본금
        entry_price: 진입 가격
        atr: ATR 값
        win_rate: 승률
        avg_win: 평균 이익
        avg_loss: 평균 손실
        use_kelly: Kelly 사용 여부
        use_atr: ATR 사용 여부
        max_position_pct: 최대 포지션 비율

    Returns:
        Dict with position_size, position_value, method, kelly_pct, atr_pct
    """
    kelly_pct = max_position_pct
    atr_pct = max_position_pct

    # Kelly 계산
    if use_kelly and win_rate > 0 and avg_win > 0 and avg_loss > 0:
        kelly_result = calculate_kelly_fraction(
            win_rate, avg_win, avg_loss,
            kelly_fraction=0.5,
            max_position_pct=max_position_pct
        )
        kelly_pct = kelly_result.position_pct

    # ATR 계산
    if use_atr and atr > 0:
        atr_result = calculate_atr_position_size(
            capital, entry_price, atr,
            risk_per_trade=0.02,
            atr_multiplier=2.0
        )
        atr_value = atr_result["position_value"]
        atr_pct = min(atr_value / capital, max_position_pct) if capital > 0 else 0

    # 더 보수적인 값 선택
    final_pct = min(kelly_pct, atr_pct)
    position_value = capital * final_pct
    position_size = position_value / entry_price if entry_price > 0 else 0

    method = "kelly" if kelly_pct <= atr_pct else "atr"

    return {
        "position_size": position_size,
        "position_value": position_value,
        "position_pct": final_pct,
        "method": method,
        "kelly_pct": kelly_pct,
        "atr_pct": atr_pct,
    }
