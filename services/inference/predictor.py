from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from packages.common.db import fetch_all
from services.registry.storage import download_model


class Predictor:
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._model_id: Optional[str] = None

    def _latest_model(self) -> Tuple[Optional[str], Optional[str]]:
        rows = fetch_all(
            "SELECT model_id, artifact_uri FROM models WHERE is_production = true ORDER BY created_at DESC LIMIT 1"
        )
        if not rows:
            return None, None
        return rows[0][0], rows[0][1]

    def ensure_loaded(self) -> None:
        model_id, artifact_uri = self._latest_model()
        if not model_id or model_id == self._model_id:
            return
        object_name = artifact_uri.split("/", 3)[-1] if artifact_uri else ""
        try:
            self._models = download_model(object_name) if object_name else {}
        except Exception:  # noqa: BLE001
            self._models = {}
        self._model_id = model_id

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        self.ensure_loaded()
        if not self._models:
            return {
                "er_long": 0.0,
                "er_short": 0.0,
                "q05_long": 0.0,
                "q05_short": 0.0,
                "e_mae_long": 0.0,
                "e_mae_short": 0.0,
                "e_hold_long": 0.0,
                "e_hold_short": 0.0,
            }
        df = pd.DataFrame([features])
        preds: Dict[str, float] = {}
        for key, model in self._models.items():
            value = float(model.predict(df)[0])
            preds[key] = value
            if key.endswith("_long"):
                preds[key.replace("_long", "_short")] = -value
        return preds
