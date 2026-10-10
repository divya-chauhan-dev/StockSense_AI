"""MACD (Moving Average Convergence Divergence) indicator."""

import pandas as pd


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Calculate MACD indicator.

    MACD shows the relationship between two EMAs and includes:
    - MACD line: Fast EMA - Slow EMA
    - Signal line: EMA of MACD line
    - Histogram: MACD line - Signal line

    Args:
        close: Series of closing prices
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)

    Returns:
        DataFrame with columns: macd, signal, histogram
    """
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def get_macd_signal(macd: float, signal: float, histogram: float) -> float:
    """
    Generate trading signal from MACD values.

    Bullish: MACD above signal line (positive histogram)
    Bearish: MACD below signal line (negative histogram)
    Signal strength based on histogram magnitude.

    Args:
        macd: MACD line value
        signal: Signal line value
        histogram: MACD histogram value

    Returns:
        Signal in range [-1, 1]
    """
    if pd.isna(macd) or pd.isna(signal) or pd.isna(histogram):
        return 0.0

    if histogram > 0:
        signal_strength = min(1.0, histogram / abs(signal) if signal != 0 else 0.5)
    else:
        signal_strength = max(-1.0, histogram / abs(signal) if signal != 0 else -0.5)

    return max(-1.0, min(1.0, signal_strength))
