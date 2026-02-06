from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from packages.common.db import fetch_all
from services.registry.storage import download_model

logger = logging.getLogger(__name__)


def _ensemble_predict(
    models: Dict[str, Any],
    X: pd.DataFrame,
    weights: Tuple[float, float] = (0.6, 0.4),
) -> np.ndarray:
    """앙상블 예측 (lgbm + catboost)"""
    lgbm_model = models.get("lgbm")
    catboost_model = models.get("catboost")

    if lgbm_model is None:
        return np.zeros(len(X))

    lgbm_pred = lgbm_model.predict(X)

    if catboost_model is None:
        return lgbm_pred

    cat_pred = catboost_model.predict(X)
    return weights[0] * lgbm_pred + weights[1] * cat_pred


class Predictor:
    def __init__(self) -> None:
        self._models: Dict[str, Any] = {}
        self._model_id: Optional[str] = None
        self._feature_names: List[str] = []

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
            # Extract feature names from the first model
            if self._models:
                first_key = next(iter(self._models))
                model_or_dict = self._models[first_key]
                if isinstance(model_or_dict, dict) and "lgbm" in model_or_dict:
                    self._feature_names = list(model_or_dict["lgbm"].feature_name_)
                elif hasattr(model_or_dict, "feature_name_"):
                    self._feature_names = list(model_or_dict.feature_name_)
                logger.info(f"Loaded model {model_id} with {len(self._feature_names)} features")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Failed to load model: {e}")
            self._models = {}
            self._feature_names = []
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

        # Create DataFrame with exactly the features the model expects
        if self._feature_names:
            # Fill missing features with 0.0
            aligned_features = {name: features.get(name, 0.0) for name in self._feature_names}
            df = pd.DataFrame([aligned_features])[self._feature_names]
        else:
            df = pd.DataFrame([features])

        # Replace NaN and inf values
        df = df.fillna(0.0).replace([np.inf, -np.inf], 0.0)

        preds: Dict[str, float] = {}

        # First pass: predict with main models (not meta)
        for key, model_or_dict in self._models.items():
            if key == "meta":
                continue  # Skip meta model in first pass
            try:
                # 중첩 딕셔너리 구조 처리 (lgbm + catboost 앙상블)
                if isinstance(model_or_dict, dict):
                    value = float(_ensemble_predict(model_or_dict, df)[0])
                else:
                    value = float(model_or_dict.predict(df)[0])
                preds[key] = value
            except Exception as e:  # noqa: BLE001
                logger.error(f"Prediction failed for {key}: {e}")
                preds[key] = 0.0

        # Fallback: if short models don't exist, use negation of long (backward compat)
        for long_key in ["er_long", "q05_long", "e_mae_long", "e_hold_long"]:
            short_key = long_key.replace("_long", "_short")
            if short_key not in preds and long_key in preds:
                preds[short_key] = -preds[long_key]

        # Second pass: predict with meta model (needs primary_pred)
        if "meta" in self._models:
            try:
                meta_model = self._models["meta"]
                # Add primary_pred (er_long prediction) to features
                meta_features = features.copy()
                meta_features["primary_pred"] = preds.get("er_long", 0.0)

                # Get meta model feature names
                if hasattr(meta_model, "feature_name_"):
                    meta_feature_names = list(meta_model.feature_name_)
                else:
                    meta_feature_names = self._feature_names + ["primary_pred"]

                aligned_meta = {name: meta_features.get(name, 0.0) for name in meta_feature_names}
                df_meta = pd.DataFrame([aligned_meta])[meta_feature_names]
                df_meta = df_meta.fillna(0.0).replace([np.inf, -np.inf], 0.0)

                meta_pred = float(meta_model.predict(df_meta)[0])
                preds["meta"] = meta_pred
            except Exception as e:  # noqa: BLE001
                logger.error(f"Meta prediction failed: {e}")
                preds["meta"] = 0.0

        return preds
