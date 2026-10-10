"""Feature engineering for ML model training."""

import pandas as pd
import numpy as np

from stock_predictor.indicators import compute_all_indicators
from stock_predictor.config import IndicatorConfig


class FeatureEngineer:
    """Prepares features for ML model training and prediction."""

    def __init__(self, config: IndicatorConfig | None = None):
        """
        Initialize feature engineer.

        Args:
            config: Indicator configuration
        """
        self.config = config or IndicatorConfig()
        self._feature_columns: list[str] | None = None

    def create_features(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
        include_target: bool = True,
    ) -> pd.DataFrame:
        """
        Create features and target variable from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns
            horizon: Prediction horizon in days
            include_target: Whether to include target variable

        Returns:
            DataFrame with features and optionally target
        """
        result = compute_all_indicators(df, self.config)

        result["return_1d"] = result["Close"].pct_change(1)
        result["return_3d"] = result["Close"].pct_change(3)
        result["return_5d"] = result["Close"].pct_change(5)
        result["return_10d"] = result["Close"].pct_change(10)

        result["volatility_10d"] = result["return_1d"].rolling(10).std()
        result["volatility_20d"] = result["return_1d"].rolling(20).std()

        result["volume_ma_5"] = result["Volume"].rolling(5).mean()
        result["volume_ma_20"] = result["Volume"].rolling(20).mean()
        result["volume_ratio"] = result["Volume"] / result["volume_ma_20"]

        result["high_low_range"] = (result["High"] - result["Low"]) / result["Close"]
        result["close_open_range"] = (result["Close"] - result["Open"]) / result["Open"]

        if include_target:
            result["future_return"] = result["Close"].shift(-horizon) / result["Close"] - 1
            result["target"] = (result["future_return"] > 0).astype(int)

        result = result.replace([np.inf, -np.inf], np.nan)

        return result

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Get list of feature column names.

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude = [
            "Open", "High", "Low", "Close", "Volume",
            "future_return", "target",
            "volume_ma_5", "volume_ma_20",
        ]
        features = [col for col in df.columns if col not in exclude]
        self._feature_columns = features
        return features

    def prepare_train_data(
        self,
        df: pd.DataFrame,
        horizon: int = 5,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with features and target.

        Args:
            df: OHLCV DataFrame
            horizon: Prediction horizon

        Returns:
            Tuple of (X features, y target)
        """
        featured = self.create_features(df, horizon=horizon, include_target=True)
        feature_cols = self.get_feature_columns(featured)

        featured = featured.dropna(subset=feature_cols + ["target"])

        X = featured[feature_cols]
        y = featured["target"]

        return X, y

    def prepare_predict_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for prediction (no target needed).

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with features for the latest row
        """
        featured = self.create_features(df, include_target=False)

        if self._feature_columns is None:
            self._feature_columns = self.get_feature_columns(featured)

        available_cols = [c for c in self._feature_columns if c in featured.columns]
        return featured[available_cols].iloc[[-1]]
