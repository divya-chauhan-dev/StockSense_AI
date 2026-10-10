"""Main API for the stock prediction system."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_predictor.config import Config
from stock_predictor.data import StockDataFetcher
from stock_predictor.indicators import compute_all_indicators
from stock_predictor.models import FeatureEngineer, PricePredictor
from stock_predictor.engine import DecisionEngine, Decision


class StockAnalyzer:
    """
    Main API for stock market analysis and prediction.

    Combines ML-based price prediction with technical indicators
    to generate actionable trading recommendations.

    Example:
        >>> analyzer = StockAnalyzer()
        >>> analyzer.train('AAPL', '2020-01-01', '2024-01-01')
        >>> result = analyzer.analyze('AAPL', horizon=5)
        >>> print(result['recommendation'])
        'BUY'
    """

    def __init__(self, config: Config | None = None):
        """
        Initialize the stock analyzer.

        Args:
            config: System configuration (uses defaults if None)
        """
        self.config = config or Config()
        self.fetcher = StockDataFetcher()
        self.feature_engineer = FeatureEngineer(self.config.indicators)
        self.predictor = PricePredictor(self.config.model)
        self.decision_engine = DecisionEngine(self.config.decision)

    def train(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime | None = None,
        horizon: int | None = None,
    ) -> dict:
        """
        Train the ML model on historical data.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            start: Training start date
            end: Training end date (defaults to today)
            horizon: Prediction horizon in days (defaults to config)

        Returns:
            Dictionary with training metrics
        """
        horizon = horizon or self.config.default_horizon

        df = self.fetcher.fetch(ticker, start, end)

        if not self.fetcher.validate_data(df):
            raise ValueError(f"Insufficient data for ticker '{ticker}'")

        X, y = self.feature_engineer.prepare_train_data(df, horizon=horizon)

        metrics = self.predictor.train(X, y)
        metrics["ticker"] = ticker
        metrics["samples"] = len(X)
        metrics["horizon"] = horizon

        return metrics

    def analyze(
        self,
        ticker: str,
        horizon: int | None = None,
        period: str = "1y",
    ) -> dict:
        """
        Analyze a stock and generate a recommendation.

        Args:
            ticker: Stock symbol
            horizon: Prediction horizon (for context; model should be trained)
            period: Data period for indicator calculation

        Returns:
            Dictionary with recommendation and analysis details
        """
        if not self.predictor.is_fitted:
            raise RuntimeError("Model must be trained before analysis. Call train() first.")

        df = self.fetcher.fetch_latest(ticker, period)
        df_with_indicators = compute_all_indicators(df, self.config.indicators)

        X = self.feature_engineer.prepare_predict_data(df)
        ml_prediction = self.predictor.predict(X)

        latest_row = df_with_indicators.iloc[-1]
        decision = self.decision_engine.analyze(latest_row, ml_prediction)

        return {
            "ticker": ticker,
            "date": df.index[-1].strftime("%Y-%m-%d"),
            "price": float(df["Close"].iloc[-1]),
            "recommendation": decision.recommendation.value,
            "confidence": round(decision.confidence, 3),
            "score": round(decision.score, 3),
            "ml_prediction": decision.ml_prediction,
            "indicator_signals": {
                k: round(v, 3) for k, v in decision.indicator_signals.items()
            },
            "signal_breakdown": {
                k: round(v, 3) for k, v in decision.signal_breakdown.items()
            },
        }

    def get_indicators(self, ticker: str, period: str = "3mo") -> pd.DataFrame:
        """
        Get technical indicator values for a stock.

        Args:
            ticker: Stock symbol
            period: Data period

        Returns:
            DataFrame with OHLCV and indicator values
        """
        df = self.fetcher.fetch_latest(ticker, period)
        return compute_all_indicators(df, self.config.indicators)

    def save_model(self, path: str | Path) -> None:
        """
        Save the trained model to disk.

        Args:
            path: File path for saving
        """
        self.predictor.save(path)

    def load_model(self, path: str | Path) -> None:
        """
        Load a trained model from disk.

        Args:
            path: File path to load from
        """
        self.predictor.load(path)

    def batch_analyze(
        self,
        tickers: list[str],
        period: str = "1y",
    ) -> list[dict]:
        """
        Analyze multiple stocks.

        Args:
            tickers: List of stock symbols
            period: Data period for each stock

        Returns:
            List of analysis results
        """
        results = []
        for ticker in tickers:
            try:
                result = self.analyze(ticker, period=period)
                results.append(result)
            except Exception as e:
                results.append({
                    "ticker": ticker,
                    "error": str(e),
                })
        return results
