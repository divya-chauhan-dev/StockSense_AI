"""Simple Moving Average (SMA) and Exponential Moving Average (EMA) indicators."""

import pandas as pd


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        close: Series of closing prices
        period: Lookback period

    Returns:
        Series of SMA values
    """
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    EMA gives more weight to recent prices compared to SMA.

    Args:
        close: Series of closing prices
        period: Lookback period

    Returns:
        Series of EMA values
    """
    return close.ewm(span=period, adjust=False).mean()


def get_ma_signal(
    close: float,
    short_ma: float,
    long_ma: float,
) -> float:
    """
    Generate trading signal from moving average crossover.

    Bullish: Price above both MAs, short MA above long MA
    Bearish: Price below both MAs, short MA below long MA

    Args:
        close: Current closing price
        short_ma: Short-term moving average value
        long_ma: Long-term moving average value

    Returns:
        Signal in range [-1, 1]
    """
    if pd.isna(short_ma) or pd.isna(long_ma):
        return 0.0

    price_vs_short = (close - short_ma) / short_ma
    price_vs_long = (close - long_ma) / long_ma
    ma_crossover = (short_ma - long_ma) / long_ma

    signal = 0.0

    if close > short_ma > long_ma:
        signal = min(1.0, (price_vs_long + ma_crossover) * 2)
    elif close < short_ma < long_ma:
        signal = max(-1.0, (price_vs_long + ma_crossover) * 2)
    else:
        signal = (price_vs_short + price_vs_long) / 2

    return max(-1.0, min(1.0, signal))
