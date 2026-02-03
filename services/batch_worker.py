from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from packages.common.db import execute, fetch_all
from services.labeling.pipeline import LabelingConfig, run_labeling
from services.training.pipeline import PipelineConfig, run_training_pipeline

logger = logging.getLogger(__name__)


def _run_labeling_job(config: Dict[str, Any]) -> Dict[str, Any]:
    """라벨링 작업 실행"""
    label_config = config.get("label_config", {})
    symbols = config.get("symbols")
    force_full = config.get("force_full", False)

    logger.info(f"Running labeling job: force_full={force_full}, symbols={symbols}")

    stats = run_labeling(
        LabelingConfig(**label_config) if label_config else None,
        symbols=symbols,
        force_full=force_full,
    )

    return {
        "status": "ok",
        "spec_hash": stats.spec_hash,
        "symbols_processed": stats.symbols_processed,
        "total_new_labels": stats.total_new_labels,
        "total_existing": stats.total_existing,
    }


def _run_training_job(config: Dict[str, Any]) -> Dict[str, Any]:
    """학습 작업 실행"""
    # config 복사하여 원본 변형 방지
    cfg_dict = dict(config)
    if "targets" in cfg_dict and isinstance(cfg_dict["targets"], list):
        cfg_dict["targets"] = tuple(cfg_dict["targets"])

    cfg = PipelineConfig(**cfg_dict)
    logger.info(f"Running training job: train={cfg.train_start}~{cfg.train_end}, algo={cfg.algo}")

    return run_training_pipeline(cfg)


def main() -> None:
    logger.info("Batch worker started")

    while True:
        rows = fetch_all("SELECT job_id, config FROM training_jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1")
        if not rows:
            time.sleep(5)
            continue

        job_id, raw_config = rows[0]
        if isinstance(raw_config, str):
            raw_config = json.loads(raw_config)

        # config를 복사하여 원본 변형 방지
        config = dict(raw_config)
        job_type = config.pop("job_type", "training")

        logger.info(f"Starting job {job_id}: type={job_type}")
        execute("UPDATE training_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))

        try:
            if job_type == "labeling":
                result = _run_labeling_job(config)
                report_payload = {
                    "spec_hash": result["spec_hash"],
                    "symbols_processed": result["symbols_processed"],
                    "total_new_labels": result["total_new_labels"],
                    "total_existing": result["total_existing"],
                }
            else:  # training
                result = _run_training_job(config)
                report_payload = result.get("report", {})

            execute(
                "UPDATE training_jobs SET status='completed', ended_at=now(), report_uri=%s, report_json=%s, error=NULL WHERE job_id=%s",
                (result.get("report_uri"), json.dumps(report_payload), job_id),
            )
            logger.info(f"Job {job_id} completed successfully")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Job {job_id} failed: {exc}")
            execute(
                "UPDATE training_jobs SET status='failed', ended_at=now(), error=%s WHERE job_id=%s",
                (str(exc), job_id),
            )
        time.sleep(1)


if __name__ == "__main__":
    main()
