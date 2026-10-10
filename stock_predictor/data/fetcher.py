"""Stock data fetching using Yahoo Finance."""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


class StockDataFetcher:
    """Fetches OHLCV stock data from Yahoo Finance."""

    def fetch(
        self,
        ticker: str,
        start_date: str | datetime,
        end_date: str | datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a ticker.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data (YYYY-MM-DD or datetime)
            end_date: End date for data (defaults to today)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume

        Raises:
            ValueError: If no data is returned or data is incomplete
        """
        if end_date is None:
            end_date = datetime.now()

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)

        return df

    def fetch_latest(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Fetch recent data using a period string.

        Args:
            ticker: Stock symbol
            period: Period string (e.g., '1mo', '3mo', '6mo', '1y', '2y', '5y')

        Returns:
            DataFrame with OHLCV data
        """
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'")

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[required_cols].copy()
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)

        return df

    def validate_data(self, df: pd.DataFrame, min_rows: int = 100) -> bool:
        """
        Validate that data meets minimum requirements.

        Args:
            df: OHLCV DataFrame
            min_rows: Minimum number of rows required

        Returns:
            True if data is valid
        """
        if len(df) < min_rows:
            return False

        if df.isnull().any().any():
            null_pct = df.isnull().sum().sum() / df.size
            if null_pct > 0.05:
                return False

        return True
