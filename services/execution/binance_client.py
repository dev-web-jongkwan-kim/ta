from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict

import httpx

from packages.common.config import get_settings


class BinanceClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret.encode("utf-8")
        self.base_url = "https://testnet.binancefuture.com" if settings.binance_testnet else "https://fapi.binance.com"

    def _sign(self, params: Dict[str, Any]) -> str:
        query = "&".join([f"{k}={params[k]}" for k in sorted(params)])
        return hmac.new(self.api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()

    def signed_request(self, method: str, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=10) as client:
            resp = client.request(method, url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        return self.signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

    def place_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.signed_request("POST", "/fapi/v1/order", params)
