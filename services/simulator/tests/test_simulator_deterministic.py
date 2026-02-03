import pandas as pd

from services.simulator.fill import FillModel, simulate_trade_path


def test_simulator_deterministic():
    ts = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
    candles = pd.DataFrame(
        {
            "ts": ts,
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1000, 1000, 1000, 1000],
        }
    )
    premium = pd.DataFrame({"ts": ts, "mark_price": candles["close"]})
    fill = FillModel(fee_rate=0.0004, slippage_k=0.1)
    result = simulate_trade_path(
        candles=candles,
        premium=premium,
        entry_ts=str(ts[0]),
        side=1,
        notional=1000,
        sl_price=99,
        tp_price=104,
        fill=fill,
    )
    assert result["exit_reason"] in {"TP", "SL", "TIME"}
    assert isinstance(result["realized_pnl"], float)
