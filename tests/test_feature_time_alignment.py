import pandas as pd
import pytest

from services.features.compute import compute_features_for_symbol


def _mk_candles(n: int = 50) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(100 + pd.Series(range(n)) * 0.25)
    high = close + 1.0
    low = close - 1.0
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series([1000.0] * n)
    return pd.DataFrame(
        {"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _mk_premium(candles: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": candles["ts"],
            "mark_price": candles["close"],
            "last_funding_rate": 0.0,
            "next_funding_time": candles["ts"] + pd.Timedelta(minutes=1),
        }
    )


def _mk_btc(candles: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ts": candles["ts"], "high": candles["close"], "low": candles["close"], "close": candles["close"], "volume": candles["volume"]})


def test_feature_time_alignment():
    candles = _mk_candles()
    premium = _mk_premium(candles)
    btc = _mk_btc(candles)
    features = compute_features_for_symbol("BTCUSDT", candles, premium, btc)
    assert len(features) == len(candles)
    features = features.sort_values("ts").reset_index(drop=True)

    expected_ret1 = candles["close"].pct_change(1)
    expected_ret5 = candles["close"].pct_change(5)
    expected_ret15 = candles["close"].pct_change(15)

    high = candles["high"]
    low = candles["low"]
    prev_close = candles["close"].shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    expected_atr = tr.rolling(window=14, min_periods=14).mean()

    for i, row in features.iterrows():
        feats = row["features"]
        if i > 0:
            assert feats["ret_1"] == pytest.approx(expected_ret1.iloc[i], rel=1e-9, abs=1e-12)
        else:
            assert pd.isna(feats["ret_1"])
        if i >= 5:
            assert feats["ret_5"] == pytest.approx(expected_ret5.iloc[i], rel=1e-9, abs=1e-12)
        else:
            assert pd.isna(feats["ret_5"])
        if i >= 15:
            assert feats["ret_15"] == pytest.approx(expected_ret15.iloc[i], rel=1e-9, abs=1e-12)
        else:
            assert pd.isna(feats["ret_15"])
        if i >= 13:
            assert feats["atr"] == pytest.approx(expected_atr.iloc[i], rel=1e-9, abs=1e-10)
        else:
            assert pd.isna(feats["atr"])
