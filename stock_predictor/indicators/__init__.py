"""Technical indicators module."""

import pandas as pd

from stock_predictor.indicators.bollinger import (
    calculate_bollinger_bands,
    get_bollinger_signal,
)
from stock_predictor.indicators.macd import calculate_macd, get_macd_signal
from stock_predictor.indicators.moving_averages import (
    calculate_ema,
    calculate_sma,
    get_ma_signal,
)
from stock_predictor.indicators.rsi import calculate_rsi, get_rsi_signal
from stock_predictor.config import IndicatorConfig

__all__ = [
    "calculate_rsi",
    "get_rsi_signal",
    "calculate_sma",
    "calculate_ema",
    "get_ma_signal",
    "calculate_macd",
    "get_macd_signal",
    "calculate_bollinger_bands",
    "get_bollinger_signal",
    "compute_all_indicators",
    "get_all_signals",
]


def compute_all_indicators(
    df: pd.DataFrame, config: IndicatorConfig | None = None
) -> pd.DataFrame:
    """
    Compute all technical indicators for a DataFrame.

    Args:
        df: DataFrame with at least 'Close' column
        config: Indicator configuration (uses defaults if None)

    Returns:
        DataFrame with all indicator columns added
    """
    if config is None:
        config = IndicatorConfig()

    result = df.copy()
    close = df["Close"]

    result["rsi"] = calculate_rsi(close, config.rsi_period)

    for period in config.sma_periods:
        result[f"sma_{period}"] = calculate_sma(close, period)

    for period in config.ema_periods:
        result[f"ema_{period}"] = calculate_ema(close, period)

    macd_df = calculate_macd(
        close, config.macd_fast, config.macd_slow, config.macd_signal
    )
    result["macd"] = macd_df["macd"]
    result["macd_signal"] = macd_df["signal"]
    result["macd_histogram"] = macd_df["histogram"]

    bb_df = calculate_bollinger_bands(close, config.bollinger_period, config.bollinger_std)
    result["bb_upper"] = bb_df["upper"]
    result["bb_middle"] = bb_df["middle"]
    result["bb_lower"] = bb_df["lower"]
    result["bb_bandwidth"] = bb_df["bandwidth"]
    result["bb_percent_b"] = bb_df["percent_b"]

    return result


def get_all_signals(row: pd.Series, config: IndicatorConfig | None = None) -> dict:
    """
    Get trading signals from all indicators for a single row.

    Args:
        row: Series containing Close price and all indicator values
        config: Indicator configuration

    Returns:
        Dictionary with signal values for each indicator
    """
    if config is None:
        config = IndicatorConfig()

    signals = {}

    signals["rsi"] = get_rsi_signal(row.get("rsi", float("nan")))

    short_ma = row.get(f"sma_{config.sma_periods[0]}", float("nan"))
    long_ma = row.get(f"sma_{config.sma_periods[1]}", float("nan"))
    signals["moving_averages"] = get_ma_signal(row["Close"], short_ma, long_ma)

    signals["macd"] = get_macd_signal(
        row.get("macd", float("nan")),
        row.get("macd_signal", float("nan")),
        row.get("macd_histogram", float("nan")),
    )

    signals["bollinger"] = get_bollinger_signal(
        row["Close"],
        row.get("bb_upper", float("nan")),
        row.get("bb_middle", float("nan")),
        row.get("bb_lower", float("nan")),
    )

    return signals
