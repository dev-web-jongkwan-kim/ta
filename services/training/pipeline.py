from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from packages.common.config import get_settings
from services.labeling.pipeline import LabelingConfig, run_labeling
from services.training.train import TrainConfig, run_training_job


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


def run_training_pipeline(cfg: PipelineConfig) -> Dict[str, Any]:
    label_cfg = cfg.to_label_config()
    spec = label_cfg.spec()
    label_spec_hash = run_labeling(label_cfg)
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
    return training_result
