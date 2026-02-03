from __future__ import annotations

import time
from typing import Optional

import httpx

from packages.common.config import get_settings
from packages.common.runtime import set_userstream_ok


class UserStreamManager:
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.binance_api_key
        self.base_url = "https://testnet.binancefuture.com" if settings.binance_testnet else "https://fapi.binance.com"
        self.listen_key: Optional[str] = None

    def start(self) -> None:
        if not self.api_key:
            set_userstream_ok(False)
            return
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.post(f"{self.base_url}/fapi/v1/listenKey", headers={"X-MBX-APIKEY": self.api_key})
                if resp.status_code != 200:
                    set_userstream_ok(False)
                    return
                self.listen_key = resp.json().get("listenKey")
                set_userstream_ok(True)
        except Exception:  # noqa: BLE001
            set_userstream_ok(False)

    def keepalive(self) -> None:
        if not self.api_key or not self.listen_key:
            set_userstream_ok(False)
            return
        with httpx.Client(timeout=10) as client:
            resp = client.put(
                f"{self.base_url}/fapi/v1/listenKey",
                headers={"X-MBX-APIKEY": self.api_key},
                params={"listenKey": self.listen_key},
            )
            if resp.status_code == 200:
                set_userstream_ok(True)
            else:
                set_userstream_ok(False)
