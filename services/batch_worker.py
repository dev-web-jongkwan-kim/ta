from __future__ import annotations

import json
import time
from typing import Any, Dict

from packages.common.db import execute, fetch_all
from services.training.pipeline import PipelineConfig, run_training_pipeline


def main() -> None:
    while True:
        rows = fetch_all("SELECT job_id, config FROM training_jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1")
        if not rows:
            time.sleep(5)
            continue
        job_id, config = rows[0]
        if isinstance(config, str):
            config = json.loads(config)
        if "targets" in config and isinstance(config["targets"], list):
            config["targets"] = tuple(config["targets"])
        execute("UPDATE training_jobs SET status='running', started_at=now() WHERE job_id=%s", (job_id,))
        cfg = PipelineConfig(**config)
        try:
            result = run_training_pipeline(cfg)
            report_payload = result.get("report", {})
            execute(
                "UPDATE training_jobs SET status='completed', ended_at=now(), report_uri=%s, report_json=%s, error=NULL WHERE job_id=%s",
                (result.get("report_uri"), json.dumps(report_payload), job_id),
            )
        except Exception as exc:  # noqa: BLE001
            execute(
                "UPDATE training_jobs SET status='failed', ended_at=now(), error=%s WHERE job_id=%s",
                (str(exc), job_id),
            )
        time.sleep(1)


if __name__ == "__main__":
    main()
