"""Relative Strength Index (RSI) indicator."""

import pandas as pd


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index.

    RSI measures momentum by comparing recent gains to recent losses.
    Values range from 0-100:
    - RSI < 30: Oversold (potential buy signal)
    - RSI > 70: Overbought (potential sell signal)

    Args:
        close: Series of closing prices
        period: Lookback period (default 14)

    Returns:
        Series of RSI values
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def get_rsi_signal(rsi: float, oversold: float = 30, overbought: float = 70) -> float:
    """
    Convert RSI value to a trading signal.

    Args:
        rsi: Current RSI value
        oversold: Threshold for oversold condition
        overbought: Threshold for overbought condition

    Returns:
        Signal in range [-1, 1]:
        - Positive values indicate bullish (buy) signal
        - Negative values indicate bearish (sell) signal
        - Values near 0 indicate neutral
    """
    if pd.isna(rsi):
        return 0.0

    if rsi <= oversold:
        return (oversold - rsi) / oversold
    elif rsi >= overbought:
        return -(rsi - overbought) / (100 - overbought)
    else:
        midpoint = (oversold + overbought) / 2
        if rsi < midpoint:
            return (midpoint - rsi) / (midpoint - oversold) * 0.3
        else:
            return -(rsi - midpoint) / (overbought - midpoint) * 0.3
