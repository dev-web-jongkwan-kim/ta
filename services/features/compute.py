from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

Timeframe = Literal["1m", "15m", "1h"]


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


def _percentile(series: pd.Series, window: int = 100) -> pd.Series:
    """시계열의 롤링 백분위 계산"""
    def pct_rank(x):
        if len(x) < 2:
            return np.nan
        return (x.rank().iloc[-1] - 1) / (len(x) - 1) * 100
    return series.rolling(window=window, min_periods=window).apply(pct_rank, raw=False)


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


def _rolling_corr(s1: pd.Series, s2: pd.Series, window: int = 60) -> pd.Series:
    """두 시계열의 롤링 상관관계"""
    return s1.rolling(window=window, min_periods=window).corr(s2)


def _trend_strength(adx: pd.Series, rsi: pd.Series) -> pd.Series:
    """추세 강도: ADX와 RSI 결합

    ADX > 25이고 RSI가 극단에 있을 때 강한 추세
    반환값: -100 ~ 100 (양수=상승추세, 음수=하락추세)
    """
    # RSI를 -50 ~ 50 스케일로 변환
    rsi_centered = rsi - 50
    # ADX를 0~1로 정규화 (25 이상이면 1에 가까움)
    adx_norm = (adx / 50).clip(0, 1)
    return rsi_centered * adx_norm * 2


def compute_features_for_symbol(
    symbol: str,
    candles: pd.DataFrame,
    premium: pd.DataFrame,
    btc_candles: pd.DataFrame,
    eth_candles: pd.DataFrame | None = None,
    open_interest: pd.DataFrame | None = None,
    long_short_ratio: pd.DataFrame | None = None,
    timeframe: Timeframe = "1m",
) -> pd.DataFrame:
    """Return features indexed by ts with JSONB dict + key summary columns (atr, funding_z, btc_regime).

    Args:
        symbol: 심볼명
        candles: 캔들 데이터
        premium: 프리미엄 인덱스 데이터
        btc_candles: BTC 캔들 (시장 레짐용)
        eth_candles: ETH 캔들 (상관관계 피처용)
        open_interest: OI 데이터
        long_short_ratio: L/S 비율 데이터
        timeframe: 타임프레임 (1m, 15m, 1h)
    """
    # 타임프레임별 윈도우 조정 계수
    tf_multiplier = {"1m": 1, "15m": 15, "1h": 60}[timeframe]

    df = candles.copy().sort_values("ts").reset_index(drop=True)
    df["symbol"] = symbol

    # 기본 수익률 피처
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)

    # 변동성 피처
    df["atr"] = _atr(df, window=14)
    ema = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_dist_atr"] = (df["close"] - ema) / df["atr"]
    df["adx"] = _adx(df, window=14)
    roll_mean = df["close"].rolling(window=20, min_periods=20).mean()
    roll_std = df["close"].rolling(window=20, min_periods=20).std(ddof=0)
    df["bb_z"] = (df["close"] - roll_mean) / roll_std.replace(0, np.nan)

    # 윈도우를 타임프레임에 맞게 조정 (60분 = 1시간 기준)
    vol_window = max(4, 60 // tf_multiplier)
    df["vol_z"] = _zscore(df["volume"], window=vol_window)

    # 기술적 지표 피처
    df["rsi"] = _rsi(df["close"], window=14)
    rsi_z_window = max(4, 60 // tf_multiplier)
    df["rsi_z"] = _zscore(df["rsi"], window=rsi_z_window)
    macd_line, macd_signal, macd_hist = _macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["macd_z"] = _zscore(df["macd"], window=rsi_z_window)
    df["natr"] = df["atr"] / df["close"] * 100
    df["obv"] = _obv(df["close"], df["volume"])
    df["obv_z"] = _zscore(df["obv"], window=rsi_z_window)

    # === 새로운 피처 1: 변동성 레짐 필터 (atr_percentile) ===
    atr_pct_window = max(10, 100 // tf_multiplier)
    df["atr_percentile"] = _percentile(df["atr"], window=atr_pct_window)

    # === 새로운 피처 2: 모멘텀 강도 (trend_strength) ===
    df["trend_strength"] = _trend_strength(df["adx"], df["rsi"])

    # === 새로운 피처 3: 시간대 피처 ===
    df["hour_of_day"] = pd.to_datetime(df["ts"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["ts"]).dt.dayofweek
    # 아시아 세션: UTC 00:00-08:00 (KST 09:00-17:00)
    df["is_asia_session"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] < 8)).astype(int)
    # 미국 세션: UTC 13:00-21:00 (EST 08:00-16:00)
    df["is_us_session"] = ((df["hour_of_day"] >= 13) & (df["hour_of_day"] < 21)).astype(int)

    # 펀딩 피처
    premium_sorted = premium.sort_values("ts") if not premium.empty else premium
    if not premium_sorted.empty:
        premium_sorted = premium_sorted.set_index("ts").reindex(df["ts"], method="ffill")
        df["funding_rate"] = premium_sorted["last_funding_rate"].values
        funding_z_window = max(8, 120 // tf_multiplier)
        df["funding_z"] = _zscore(df["funding_rate"].fillna(0), window=funding_z_window)
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

    # BTC 레짐 피처
    btc = btc_candles.copy().sort_values("ts").reset_index(drop=True)
    btc_ret_window = max(4, 60 // tf_multiplier)
    btc["btc_ret_60"] = btc["close"].pct_change(btc_ret_window)
    btc["btc_vol_60"] = btc["close"].pct_change().rolling(window=btc_ret_window, min_periods=btc_ret_window).std(ddof=0)
    btc_series = btc.set_index("ts")["btc_ret_60"].reindex(df["ts"], method="ffill")
    btc_vol = btc.set_index("ts")["btc_vol_60"].reindex(df["ts"], method="ffill")
    df["btc_ret_60"] = btc_series.values
    df["btc_vol_60"] = btc_vol.values
    df["btc_regime"] = np.where(df["btc_vol_60"] > df["btc_vol_60"].median(), 1, 0)

    # === 새로운 피처 4: 심볼 간 상관관계 ===
    # BTC 선행 수익률 (BTC가 먼저 움직이면 따라오는 패턴)
    if symbol != "BTCUSDT":
        btc["btc_ret_lead"] = btc["close"].pct_change(1).shift(1)  # 1캔들 전 BTC 수익률
        btc_lead = btc.set_index("ts")["btc_ret_lead"].reindex(df["ts"], method="ffill")
        df["btc_lead_ret"] = btc_lead.values
    else:
        df["btc_lead_ret"] = 0.0

    # ETH/BTC 스프레드 (있는 경우)
    if eth_candles is not None and not eth_candles.empty and symbol not in ["BTCUSDT", "ETHUSDT"]:
        eth = eth_candles.copy().sort_values("ts").reset_index(drop=True)
        eth_indexed = eth.set_index("ts")["close"].reindex(df["ts"], method="ffill")
        btc_indexed = btc.set_index("ts")["close"].reindex(df["ts"], method="ffill")
        eth_btc_ratio = eth_indexed / btc_indexed.replace(0, np.nan)
        eth_btc_ratio_pct = eth_btc_ratio.pct_change(btc_ret_window)
        df["eth_btc_spread"] = eth_btc_ratio_pct.values
    else:
        df["eth_btc_spread"] = 0.0

    # === 새로운 피처 5: Rolling Correlation (심볼-BTC 상관관계) ===
    corr_window = max(30, 60 // tf_multiplier)  # 60분 기준
    btc_ret_series = btc.set_index("ts")["close"].pct_change(1).reindex(df["ts"], method="ffill")

    if symbol != "BTCUSDT":
        df["btc_corr_60"] = _rolling_corr(df["ret_1"], btc_ret_series, window=corr_window)
        # 상관관계 레짐 브레이크 (상관관계 < 0.3이면 디커플링)
        df["corr_regime_break"] = (df["btc_corr_60"].abs() < 0.3).astype(int)
    else:
        df["btc_corr_60"] = 1.0
        df["corr_regime_break"] = 0

    # ETH 상관관계 (BTC, ETH 제외)
    if eth_candles is not None and not eth_candles.empty and symbol not in ["BTCUSDT", "ETHUSDT"]:
        eth_ret_series = eth.set_index("ts")["close"].pct_change(1).reindex(df["ts"], method="ffill")
        df["eth_corr_60"] = _rolling_corr(df["ret_1"], eth_ret_series, window=corr_window)
    else:
        df["eth_corr_60"] = 0.0 if symbol == "BTCUSDT" else 1.0 if symbol == "ETHUSDT" else 0.0

    # Open Interest 피처
    oi_1h_window = max(1, 60 // tf_multiplier)
    oi_4h_window = max(1, 240 // tf_multiplier)
    oi_z_window = max(8, 120 // tf_multiplier)

    if open_interest is not None and not open_interest.empty:
        oi_sorted = open_interest.sort_values("ts").set_index("ts")
        oi_reindex = oi_sorted["open_interest"].reindex(df["ts"], method="ffill")
        df["oi"] = oi_reindex.values
        df["oi_change_1h"] = df["oi"].pct_change(oi_1h_window)
        df["oi_change_4h"] = df["oi"].pct_change(oi_4h_window)
        df["oi_z"] = _zscore(df["oi"], window=oi_z_window)
    else:
        df["oi"] = np.nan
        df["oi_change_1h"] = np.nan
        df["oi_change_4h"] = np.nan
        df["oi_z"] = np.nan

    # Long/Short 비율 피처
    ls_z_window = max(8, 120 // tf_multiplier)
    taker_z_window = max(4, 60 // tf_multiplier)

    if long_short_ratio is not None and not long_short_ratio.empty:
        ls_sorted = long_short_ratio.sort_values("ts").set_index("ts")
        ls_reindex = ls_sorted.reindex(df["ts"], method="ffill")
        df["ls_ratio"] = ls_reindex["long_short_ratio"].values if "long_short_ratio" in ls_reindex.columns else np.nan
        df["ls_ratio_z"] = _zscore(df["ls_ratio"].fillna(1.0), window=ls_z_window)
        df["ls_extreme"] = np.where((df["ls_ratio"] > 2.0) | (df["ls_ratio"] < 0.5), 1, 0)
        df["top_ls_ratio"] = ls_reindex["top_long_short_ratio"].values if "top_long_short_ratio" in ls_reindex.columns else np.nan
        df["top_ls_ratio_z"] = _zscore(df["top_ls_ratio"].fillna(1.0), window=ls_z_window)
        df["taker_ratio"] = ls_reindex["taker_buy_sell_ratio"].values if "taker_buy_sell_ratio" in ls_reindex.columns else np.nan
        df["taker_ratio_z"] = _zscore(df["taker_ratio"].fillna(1.0), window=taker_z_window)
    else:
        df["ls_ratio"] = np.nan
        df["ls_ratio_z"] = np.nan
        df["ls_extreme"] = 0
        df["top_ls_ratio"] = np.nan
        df["top_ls_ratio_z"] = np.nan
        df["taker_ratio"] = np.nan
        df["taker_ratio_z"] = np.nan

    feature_cols = [
        # 가격 피처 (3개)
        "ret_1", "ret_5", "ret_15",
        # 변동성 피처 (5개)
        "atr", "natr", "bb_z", "vol_z", "atr_percentile",
        # 추세/모멘텀 피처 (9개)
        "ema_dist_atr", "adx", "rsi", "rsi_z",
        "macd", "macd_signal", "macd_hist", "macd_z",
        "trend_strength",
        # 거래량 피처 (2개)
        "obv", "obv_z",
        # 시간대 피처 (4개)
        "hour_of_day", "day_of_week", "is_asia_session", "is_us_session",
        # 펀딩/베이시스 피처 (3개)
        "funding_rate", "funding_z", "basis",
        # BTC 레짐 피처 (3개)
        "btc_ret_60", "btc_vol_60", "btc_regime",
        # 심볼 상관관계 피처 (5개)
        "btc_lead_ret", "eth_btc_spread",
        "btc_corr_60", "corr_regime_break", "eth_corr_60",
        # Open Interest 피처 (3개)
        "oi_change_1h", "oi_change_4h", "oi_z",
        # Long/Short 비율 피처 (7개)
        "ls_ratio", "ls_ratio_z", "ls_extreme",
        "top_ls_ratio", "top_ls_ratio_z",
        "taker_ratio", "taker_ratio_z",
    ]

    df["features"] = df[feature_cols].apply(lambda row: row.to_dict(), axis=1)
    df["schema_version"] = 4  # 스키마 버전 업데이트 (멀티TF + Rolling Corr)
    return df[["symbol", "ts", "schema_version", "features", "atr", "funding_z", "btc_regime"]]


def compute_multi_tf_features(
    symbol: str,
    features_1m: pd.DataFrame,
    features_15m: pd.DataFrame,
    features_1h: pd.DataFrame,
) -> pd.DataFrame:
    """멀티 타임프레임 피처 조합

    1분 타임스탬프를 기준으로 15분/1시간 피처를 forward-fill로 조인

    Args:
        symbol: 심볼명
        features_1m: 1분 피처 DataFrame
        features_15m: 15분 피처 DataFrame
        features_1h: 1시간 피처 DataFrame

    Returns:
        조합된 피처 DataFrame
    """
    if features_1m.empty:
        return pd.DataFrame()

    df = features_1m.copy()
    df = df.set_index("ts").sort_index()

    # 15분 피처 조인 (forward-fill)
    if not features_15m.empty:
        f15 = features_15m.set_index("ts").sort_index()
        f15_features = f15["features"].reindex(df.index, method="ffill")
        df["features_15m"] = f15_features
    else:
        df["features_15m"] = None

    # 1시간 피처 조인 (forward-fill)
    if not features_1h.empty:
        f1h = features_1h.set_index("ts").sort_index()
        f1h_features = f1h["features"].reindex(df.index, method="ffill")
        df["features_1h"] = f1h_features
    else:
        df["features_1h"] = None

    df = df.reset_index()
    df["symbol"] = symbol

    # 멀티TF 일치 피처 계산
    def calc_multi_tf_agreement(row):
        """멀티 타임프레임 추세 일치 여부"""
        try:
            f1m = row["features"] if isinstance(row["features"], dict) else {}
            f15m = row["features_15m"] if isinstance(row["features_15m"], dict) else {}
            f1h = row["features_1h"] if isinstance(row["features_1h"], dict) else {}

            # RSI 기반 추세 방향
            rsi_1m = f1m.get("rsi", 50)
            rsi_15m = f15m.get("rsi", 50) if f15m else 50
            rsi_1h = f1h.get("rsi", 50) if f1h else 50

            # 모두 같은 방향이면 1, 아니면 0
            bullish = (rsi_1m > 50) and (rsi_15m > 50) and (rsi_1h > 50)
            bearish = (rsi_1m < 50) and (rsi_15m < 50) and (rsi_1h < 50)

            if bullish:
                return 1
            elif bearish:
                return -1
            return 0
        except Exception:
            return 0

    df["multi_tf_agreement"] = df.apply(calc_multi_tf_agreement, axis=1)

    return df
