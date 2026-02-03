from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from packages.common.runtime import get_mode
from services.execution.binance_client import BinanceClient


@dataclass
class OrderRequest:
    symbol: str
    side: str  # BUY/SELL
    notional: float
    leverage: int
    sl_price: float
    tp_price: float
    entry_price: float


def place_entry_with_brackets(req: OrderRequest) -> Dict[str, Any]:
    mode = get_mode()
    if mode != "live":
        return {"status": "skipped", "reason": f"mode={mode}"}

    client = BinanceClient()
    client.set_leverage(req.symbol, req.leverage)
    qty = req.notional / max(req.entry_price, 1e-8)

    try:
        entry = client.place_order(
            {
                "symbol": req.symbol,
                "side": req.side,
                "type": "MARKET",
                "quantity": round(qty, 6),
            }
        )
        sl = client.place_order(
            {
                "symbol": req.symbol,
                "side": "SELL" if req.side == "BUY" else "BUY",
                "type": "STOP_MARKET",
                "stopPrice": req.sl_price,
                "closePosition": "true",
                "reduceOnly": "true",
            }
        )
        tp = client.place_order(
            {
                "symbol": req.symbol,
                "side": "SELL" if req.side == "BUY" else "BUY",
                "type": "TAKE_PROFIT_MARKET",
                "stopPrice": req.tp_price,
                "closePosition": "true",
                "reduceOnly": "true",
            }
        )
        return {"status": "ok", "entry": entry, "sl": sl, "tp": tp}
    except Exception as exc:  # noqa: BLE001
        client.place_order(
            {
                "symbol": req.symbol,
                "side": "SELL" if req.side == "BUY" else "BUY",
                "type": "MARKET",
                "quantity": round(qty, 6),
                "reduceOnly": "true",
            }
        )
        return {"status": "failed", "error": str(exc)}
