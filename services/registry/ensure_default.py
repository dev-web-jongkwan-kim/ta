from __future__ import annotations

import json
import uuid

from packages.common.db import execute, fetch_all


def ensure_default_model() -> str:
    rows = fetch_all("SELECT model_id FROM models WHERE is_production = true LIMIT 1")
    if rows:
        return str(rows[0][0])
    model_id = str(uuid.uuid4())
    execute(
        """
        INSERT INTO models (model_id, algo, feature_schema_version, label_spec_hash, train_start, train_end, metrics, artifact_uri, is_production)
        VALUES (%s,%s,%s,%s,now(),now(),%s,%s,%s)
        """,
        (
            model_id,
            "dummy",
            1,
            "dummy",
            json.dumps({}),
            "s3://models/dummy",
            True,
        ),
    )
    return model_id
