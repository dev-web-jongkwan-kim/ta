from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from packages.common.db import fetch_all
from services.labeling.pipeline import LabelingConfig
from services.training.train import TrainConfig, run_training_job

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    targets: Tuple[str, ...] = ("er_long", "q05_long", "e_mae_long", "e_hold_long")
    feature_schema_version: int = 1
    algo: str = "lgbm"
    purge_bars: int = 0
    embargo_pct: float = 0.0
    label_config: Optional[Dict[str, Any]] = None

    def to_label_config(self) -> LabelingConfig:
        conf = self.label_config or {}
        return LabelingConfig(**conf)


def _check_label_availability(
    spec_hash: str,
    train_start: str,
    train_end: str,
    min_labels: int = 10000,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    min_val_labels: int = 1000,
) -> Dict[str, Any]:
    """라벨 가용성 검증 (long/short 모두, train/val 기간 모두 확인)

    Args:
        spec_hash: 라벨 스펙 해시
        train_start: 학습 시작일
        train_end: 학습 종료일
        min_labels: 학습 기간 최소 필요 라벨 수
        val_start: 검증 시작일 (선택)
        val_end: 검증 종료일 (선택)
        min_val_labels: 검증 기간 최소 필요 라벨 수

    Returns:
        {"available": bool, "train": {...}, "val": {...} or None}
    """
    # 학습 기간 라벨 카운트
    rows = fetch_all(
        """
        SELECT
            (SELECT COUNT(*) FROM labels_long_1m WHERE spec_hash = %s AND ts BETWEEN %s AND %s) as long_count,
            (SELECT COUNT(*) FROM labels_short_1m WHERE spec_hash = %s AND ts BETWEEN %s AND %s) as short_count
        """,
        (spec_hash, train_start, train_end, spec_hash, train_start, train_end),
    )
    train_long = rows[0][0] if rows else 0
    train_short = rows[0][1] if rows else 0
    train_min = min(train_long, train_short)
    train_available = train_min >= min_labels

    logger.debug(
        f"Train label check: spec_hash={spec_hash}, "
        f"long={train_long:,}, short={train_short:,}, min_required={min_labels:,}"
    )

    result: Dict[str, Any] = {
        "available": train_available,
        "train": {
            "long_count": train_long,
            "short_count": train_short,
            "min_count": train_min,
            "required": min_labels,
            "available": train_available,
        },
        "val": None,
    }

    # 검증 기간 라벨 카운트 (옵션)
    if val_start and val_end:
        rows_val = fetch_all(
            """
            SELECT
                (SELECT COUNT(*) FROM labels_long_1m WHERE spec_hash = %s AND ts BETWEEN %s AND %s) as long_count,
                (SELECT COUNT(*) FROM labels_short_1m WHERE spec_hash = %s AND ts BETWEEN %s AND %s) as short_count
            """,
            (spec_hash, val_start, val_end, spec_hash, val_start, val_end),
        )
        val_long = rows_val[0][0] if rows_val else 0
        val_short = rows_val[0][1] if rows_val else 0
        val_min = min(val_long, val_short)
        val_available = val_min >= min_val_labels

        logger.debug(
            f"Val label check: spec_hash={spec_hash}, "
            f"long={val_long:,}, short={val_short:,}, min_required={min_val_labels:,}"
        )

        result["val"] = {
            "long_count": val_long,
            "short_count": val_short,
            "min_count": val_min,
            "required": min_val_labels,
            "available": val_available,
        }
        result["available"] = train_available and val_available

    return result


def run_training_pipeline(
    cfg: PipelineConfig, min_labels: int = 10000, min_val_labels: int = 1000
) -> Dict[str, Any]:
    """학습 파이프라인 실행

    Args:
        cfg: 파이프라인 설정
        min_labels: 학습 기간 최소 필요 라벨 수
        min_val_labels: 검증 기간 최소 필요 라벨 수

    Returns:
        학습 결과 딕셔너리

    Raises:
        ValueError: 라벨이 부족한 경우
    """
    label_cfg = cfg.to_label_config()
    spec = label_cfg.spec()
    label_spec_hash = spec.hash()  # run_labeling 대신 직접 계산

    logger.info(
        f"Starting training pipeline: spec_hash={label_spec_hash}, "
        f"train={cfg.train_start}~{cfg.train_end}, val={cfg.val_start}~{cfg.val_end}"
    )

    # 라벨 가용성 검증 (train/val 기간 모두)
    check = _check_label_availability(
        label_spec_hash,
        cfg.train_start,
        cfg.train_end,
        min_labels,
        val_start=cfg.val_start,
        val_end=cfg.val_end,
        min_val_labels=min_val_labels,
    )

    if not check["available"]:
        train_info = check["train"]
        val_info = check["val"]

        error_parts = []
        if not train_info["available"]:
            error_parts.append(
                f"학습 기간 라벨 부족: long={train_info['long_count']:,}, "
                f"short={train_info['short_count']:,} (최소 {min_labels:,}개 필요)"
            )
        if val_info and not val_info["available"]:
            error_parts.append(
                f"검증 기간 라벨 부족: long={val_info['long_count']:,}, "
                f"short={val_info['short_count']:,} (최소 {min_val_labels:,}개 필요)"
            )

        raise ValueError(
            ". ".join(error_parts) + ". "
            "라벨링을 먼저 실행하세요: python -m services.labeling.run_labeling"
        )

    train_info = check["train"]
    val_info = check["val"]
    logger.info(
        f"Label check passed: train(long={train_info['long_count']:,}, short={train_info['short_count']:,})"
        + (f", val(long={val_info['long_count']:,}, short={val_info['short_count']:,})" if val_info else "")
    )

    spec_meta = asdict(spec)
    spec_meta["spec_hash"] = label_spec_hash
    train_cfg = TrainConfig(
        label_spec_hash=label_spec_hash,
        feature_schema_version=cfg.feature_schema_version,
        train_start=cfg.train_start,
        train_end=cfg.train_end,
        val_start=cfg.val_start,
        val_end=cfg.val_end,
        targets=cfg.targets,
        algo=cfg.algo,
        purge_bars=cfg.purge_bars,
        embargo_pct=cfg.embargo_pct,
        label_spec=spec_meta,
    )
    training_result = run_training_job(train_cfg)
    training_result["label_spec_hash"] = label_spec_hash
    training_result["target_config"] = cfg.targets

    logger.info(f"Training pipeline complete: spec_hash={label_spec_hash}")

    return training_result
