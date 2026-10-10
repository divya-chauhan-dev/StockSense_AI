"""Bollinger Bands indicator."""

import pandas as pd


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of:
    - Middle band: SMA of closing prices
    - Upper band: Middle + (std_dev * standard deviation)
    - Lower band: Middle - (std_dev * standard deviation)

    Args:
        close: Series of closing prices
        period: Lookback period for SMA and std dev (default 20)
        std_dev: Number of standard deviations (default 2.0)

    Returns:
        DataFrame with columns: upper, middle, lower, bandwidth, percent_b
    """
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    bandwidth = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower)

    return pd.DataFrame(
        {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        }
    )


def get_bollinger_signal(
    close: float,
    upper: float,
    middle: float,
    lower: float,
) -> float:
    """
    Generate trading signal from Bollinger Bands.

    Price near lower band: Potentially oversold (buy signal)
    Price near upper band: Potentially overbought (sell signal)

    Args:
        close: Current closing price
        upper: Upper band value
        middle: Middle band value
        lower: Lower band value

    Returns:
        Signal in range [-1, 1]
    """
    if pd.isna(upper) or pd.isna(lower) or pd.isna(close):
        return 0.0

    band_width = upper - lower
    if band_width == 0:
        return 0.0

    position = (close - lower) / band_width

    if position <= 0:
        return 1.0
    elif position >= 1:
        return -1.0
    elif position < 0.5:
        return (0.5 - position) * 2
    else:
        return -(position - 0.5) * 2
