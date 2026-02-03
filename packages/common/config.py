from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    mode: str = "shadow"  # off|shadow|live
    app_env: str = "local"

    database_url: str = "postgresql://ta:ta@localhost:5432/ta"
    redis_url: str = "redis://localhost:6379/0"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minio"
    minio_secret_key: str = "minio123"
    minio_bucket: str = "models"
    minio_secure: bool = False

    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True

    universe: str = "BTCUSDT,ETHUSDT"

    collect_interval_sec: int = 5
    flush_interval_sec: int = 2
    flush_batch_size: int = 500
    universe_refresh_sec: int = 3600
    next_public_api_base: str | None = None
    drift_reference_hours: int = 168
    drift_current_hours: int = 1
    drift_alert_psi: float = 0.2
    drift_block_psi: float = 0.4
    drift_latency_threshold_ms: float = 120000
    drift_missing_alert_rate: float = 0.005
    drift_missing_block_rate: float = 0.02

    max_used_margin_pct: float = 0.35
    daily_loss_limit_pct: float = 0.02
    max_positions: int = 6
    max_total_notional_pct: float = 1.2
    max_directional_notional_pct: float = 0.8

    ev_min: float = 0.0
    q05_min: float = -0.002
    mae_max: float = 0.01

    taker_fee_rate: float = 0.0004
    slippage_k: float = 0.15

    data_stale_sec: int = 120
    userstream_stale_sec: int = 120
    order_failure_spike: int = 5

    def universe_list(self) -> List[str]:
        return [s.strip().upper() for s in self.universe.split(",") if s.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
