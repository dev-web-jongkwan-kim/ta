import pandas as pd

from services.features.compute import compute_features_for_symbol


def test_features_time_alignment_no_future_leakage():
    ts = pd.date_range("2024-01-01", periods=100, freq="1min", tz="UTC")
    candles = pd.DataFrame(
        {
            "symbol": "BTCUSDT",
            "ts": ts,
            "open": range(100, 200),
            "high": range(101, 201),
            "low": range(99, 199),
            "close": range(100, 200),
            "volume": [1000] * 100,
        }
    )
    premium = pd.DataFrame({"ts": ts, "mark_price": candles["close"], "last_funding_rate": 0.0001})
    btc = candles.copy()

    base = compute_features_for_symbol("BTCUSDT", candles, premium, btc)

    candles_future = candles.copy()
    candles_future.loc[80:, "close"] = candles_future.loc[80:, "close"] * 10
    premium_future = premium.copy()
    premium_future.loc[80:, "mark_price"] = premium_future.loc[80:, "mark_price"] * 10

    mutated = compute_features_for_symbol("BTCUSDT", candles_future, premium_future, btc)

    pd.testing.assert_frame_equal(
        base.loc[:70, ["atr", "funding_z", "btc_regime"]],
        mutated.loc[:70, ["atr", "funding_z", "btc_regime"]],
        check_dtype=False,
    )
