"""
RL Decision Storage
모든 RL 결정을 DB에 저장하고 조회
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from packages.common.db import execute, fetch_all
from services.rl.agent import RLDecision


def save_decision(decision: RLDecision, model_id: str, executed: bool = False) -> int:
    """
    RL 결정을 DB에 저장

    Args:
        decision: RLDecision 객체
        model_id: 사용된 RL 모델 ID
        executed: 실제 실행 여부

    Returns:
        저장된 레코드 ID
    """
    query = """
        INSERT INTO rl_decisions (
            ts, symbol, model_id, action, action_name, confidence,
            action_probs, value_estimate, model_predictions, observation,
            position_before, position_after, executed, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
        )
        RETURNING id
    """

    params = (
        decision.ts,
        decision.symbol,
        model_id,
        decision.action,
        decision.action_name,
        decision.confidence,
        json.dumps(decision.action_probs.tolist() if decision.action_probs is not None else None),
        decision.value,
        json.dumps(decision.model_predictions),
        json.dumps(decision.observation.tolist() if hasattr(decision.observation, 'tolist') else decision.observation),
        decision.position_before,
        decision.position_after,
        executed,
    )

    result = fetch_all(query, params)
    return result[0][0] if result else 0


def update_decision_result(decision_id: int, pnl_result: float) -> None:
    """결정 결과 업데이트 (손익 기록)"""
    execute(
        "UPDATE rl_decisions SET pnl_result = %s WHERE id = %s",
        (pnl_result, decision_id)
    )


def mark_executed(decision_id: int) -> None:
    """결정이 실행되었음을 표시"""
    execute(
        "UPDATE rl_decisions SET executed = true WHERE id = %s",
        (decision_id,)
    )


def get_recent_decisions(
    symbol: Optional[str] = None,
    model_id: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """최근 결정 조회"""
    query = """
        SELECT id, ts, symbol, model_id, action, action_name, confidence,
               action_probs, value_estimate, model_predictions,
               position_before, position_after, executed, pnl_result, created_at
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
        "action_probs", "value_estimate", "model_predictions",
        "position_before", "position_after", "executed", "pnl_result", "created_at"
    ]
    return [dict(zip(columns, r)) for r in rows]


def get_decision_stats(symbol: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
    """결정 통계"""
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

    return {
        "period_hours": hours,
        "symbol": symbol,
        "by_action": stats,
    }


def get_production_model(symbol: str) -> Optional[Dict[str, Any]]:
    """특정 심볼의 production RL 모델 조회"""
    rows = fetch_all(
        """
        SELECT model_id, symbol, algorithm, train_start, train_end,
               metrics, model_path, is_production, created_at
        FROM rl_models
        WHERE symbol = %s AND is_production = true
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (symbol,)
    )

    if not rows:
        return None

    columns = [
        "model_id", "symbol", "algorithm", "train_start", "train_end",
        "metrics", "model_path", "is_production", "created_at"
    ]
    return dict(zip(columns, rows[0]))


def get_all_production_models() -> List[Dict[str, Any]]:
    """모든 production RL 모델 조회"""
    rows = fetch_all(
        """
        SELECT model_id, symbol, algorithm, train_start, train_end,
               metrics, model_path, is_production, created_at
        FROM rl_models
        WHERE is_production = true
        ORDER BY symbol
        """
    )

    columns = [
        "model_id", "symbol", "algorithm", "train_start", "train_end",
        "metrics", "model_path", "is_production", "created_at"
    ]
    return [dict(zip(columns, r)) for r in rows]
