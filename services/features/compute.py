from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureRow:
    symbol: str
    ts: str
    schema_version: int
    features: Dict[str, Any]
    atr: Optional[float]
    funding_z: Optional[float]
    btc_regime: Optional[int]


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def _zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window, min_periods=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window, min_periods=window).mean() / atr)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return 100 * dx.rolling(window=window, min_periods=window).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD, Signal, Histogram"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


def compute_features_for_symbol(
    symbol: str,
    candles: pd.DataFrame,
    premium: pd.DataFrame,
    btc_candles: pd.DataFrame,
    open_interest: pd.DataFrame | None = None,
    long_short_ratio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return features indexed by ts with JSONB dict + key summary columns (atr, funding_z, btc_regime)."""
    df = candles.copy().sort_values("ts").reset_index(drop=True)
    df["symbol"] = symbol
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)

    df["atr"] = _atr(df, window=14)
    ema = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_dist_atr"] = (df["close"] - ema) / df["atr"]
    df["adx"] = _adx(df, window=14)
    roll_mean = df["close"].rolling(window=20, min_periods=20).mean()
    roll_std = df["close"].rolling(window=20, min_periods=20).std(ddof=0)
    df["bb_z"] = (df["close"] - roll_mean) / roll_std.replace(0, np.nan)
    df["vol_z"] = _zscore(df["volume"], window=60)

    # 신규 기술적 지표 피처
    df["rsi"] = _rsi(df["close"], window=14)
    df["rsi_z"] = _zscore(df["rsi"], window=60)  # RSI의 Z-score
    macd_line, macd_signal, macd_hist = _macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["macd_z"] = _zscore(df["macd"], window=60)
    df["natr"] = df["atr"] / df["close"] * 100  # Normalized ATR (%)
    df["obv"] = _obv(df["close"], df["volume"])
    df["obv_z"] = _zscore(df["obv"], window=60)  # OBV Z-score

    premium_sorted = premium.sort_values("ts") if not premium.empty else premium
    if not premium_sorted.empty:
        premium_sorted = premium_sorted.set_index("ts").reindex(df["ts"], method="ffill")
        df["funding_rate"] = premium_sorted["last_funding_rate"].values
        df["funding_z"] = _zscore(df["funding_rate"].fillna(0), window=120)
        if "last_price" in premium_sorted.columns and "index_price" in premium_sorted.columns:
            df["basis"] = (premium_sorted["last_price"] - premium_sorted["index_price"]) / premium_sorted[
                "index_price"
            ]
        else:
            df["basis"] = np.nan
    else:
        df["funding_rate"] = np.nan
        df["funding_z"] = np.nan
        df["basis"] = np.nan

    btc = btc_candles.copy().sort_values("ts").reset_index(drop=True)
    btc["btc_ret_60"] = btc["close"].pct_change(60)
    btc["btc_vol_60"] = btc["close"].pct_change().rolling(window=60, min_periods=60).std(ddof=0)
    btc_series = btc.set_index("ts")["btc_ret_60"].reindex(df["ts"], method="ffill")
    btc_vol = btc.set_index("ts")["btc_vol_60"].reindex(df["ts"], method="ffill")
    df["btc_ret_60"] = btc_series.values
    df["btc_vol_60"] = btc_vol.values
    df["btc_regime"] = np.where(df["btc_vol_60"] > df["btc_vol_60"].median(), 1, 0)

    # Open Interest 피처
    if open_interest is not None and not open_interest.empty:
        oi_sorted = open_interest.sort_values("ts").set_index("ts")
        oi_reindex = oi_sorted["open_interest"].reindex(df["ts"], method="ffill")
        df["oi"] = oi_reindex.values
        df["oi_change_1h"] = df["oi"].pct_change(60)   # 1시간 OI 변화율
        df["oi_change_4h"] = df["oi"].pct_change(240)  # 4시간 OI 변화율
        df["oi_z"] = _zscore(df["oi"], window=120)     # OI Z-score
    else:
        df["oi"] = np.nan
        df["oi_change_1h"] = np.nan
        df["oi_change_4h"] = np.nan
        df["oi_z"] = np.nan

    # Long/Short 비율 피처
    if long_short_ratio is not None and not long_short_ratio.empty:
        ls_sorted = long_short_ratio.sort_values("ts").set_index("ts")
        ls_reindex = ls_sorted.reindex(df["ts"], method="ffill")
        # Global Long/Short Ratio
        df["ls_ratio"] = ls_reindex["long_short_ratio"].values if "long_short_ratio" in ls_reindex.columns else np.nan
        df["ls_ratio_z"] = _zscore(df["ls_ratio"].fillna(1.0), window=120)
        # Long/Short 극단값 (>2 또는 <0.5 = 극단적 편향)
        df["ls_extreme"] = np.where((df["ls_ratio"] > 2.0) | (df["ls_ratio"] < 0.5), 1, 0)
        # Top Trader Long/Short Ratio
        df["top_ls_ratio"] = ls_reindex["top_long_short_ratio"].values if "top_long_short_ratio" in ls_reindex.columns else np.nan
        df["top_ls_ratio_z"] = _zscore(df["top_ls_ratio"].fillna(1.0), window=120)
        # Taker Buy/Sell Ratio
        df["taker_ratio"] = ls_reindex["taker_buy_sell_ratio"].values if "taker_buy_sell_ratio" in ls_reindex.columns else np.nan
        df["taker_ratio_z"] = _zscore(df["taker_ratio"].fillna(1.0), window=60)
    else:
        df["ls_ratio"] = np.nan
        df["ls_ratio_z"] = np.nan
        df["ls_extreme"] = 0
        df["top_ls_ratio"] = np.nan
        df["top_ls_ratio_z"] = np.nan
        df["taker_ratio"] = np.nan
        df["taker_ratio_z"] = np.nan

    feature_cols = [
        # 기존 가격 피처 (3개)
        "ret_1",
        "ret_5",
        "ret_15",
        # 변동성 피처 (4개)
        "atr",
        "natr",  # 신규: Normalized ATR
        "bb_z",
        "vol_z",
        # 추세/모멘텀 피처 (8개)
        "ema_dist_atr",
        "adx",
        "rsi",           # 신규: RSI
        "rsi_z",         # 신규: RSI Z-score
        "macd",          # 신규: MACD
        "macd_signal",   # 신규: MACD Signal
        "macd_hist",     # 신규: MACD Histogram
        "macd_z",        # 신규: MACD Z-score
        # 거래량 피처 (2개)
        "obv",           # 신규: OBV
        "obv_z",         # 신규: OBV Z-score
        # 펀딩/베이시스 피처 (3개)
        "funding_rate",
        "funding_z",
        "basis",
        # BTC 레짐 피처 (3개)
        "btc_ret_60",
        "btc_vol_60",
        "btc_regime",
        # Open Interest 피처 (3개)
        "oi_change_1h",
        "oi_change_4h",
        "oi_z",
        # Long/Short 비율 피처 (7개)
        "ls_ratio",
        "ls_ratio_z",
        "ls_extreme",
        "top_ls_ratio",
        "top_ls_ratio_z",
        "taker_ratio",
        "taker_ratio_z",
    ]

    df["features"] = df[feature_cols].apply(lambda row: row.to_dict(), axis=1)
    df["schema_version"] = 2  # 스키마 버전 업데이트
    return df[["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"]]
