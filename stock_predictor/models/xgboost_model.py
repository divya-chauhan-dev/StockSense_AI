"""XGBoost model for price direction prediction."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from stock_predictor.config import ModelConfig


class PricePredictor:
    """XGBoost-based price direction predictor."""

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize the predictor.

        Args:
            config: Model configuration
        """
        self.config = config or ModelConfig()
        self.model: xgb.XGBClassifier | None = None
        self.scaler = StandardScaler()
        self._is_fitted = False

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validate: bool = True,
    ) -> dict:
        """
        Train the model on historical data.

        Args:
            X: Feature DataFrame
            y: Target Series (0/1 for down/up)
            validate: Whether to run cross-validation

        Returns:
            Dictionary with training metrics
        """
        X_scaled = self.scaler.fit_transform(X)

        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_child_weight=self.config.min_child_weight,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        metrics = {}

        if validate:
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(
                self.model, X_scaled, y, cv=tscv, scoring="accuracy"
            )
            metrics["cv_accuracy_mean"] = cv_scores.mean()
            metrics["cv_accuracy_std"] = cv_scores.std()

        self.model.fit(X_scaled, y)
        self._is_fitted = True

        train_pred = self.model.predict(X_scaled)
        metrics["train_accuracy"] = (train_pred == y).mean()

        if hasattr(self.model, "feature_importances_"):
            importance = pd.Series(
                self.model.feature_importances_,
                index=X.columns,
            ).sort_values(ascending=False)
            metrics["top_features"] = importance.head(10).to_dict()

        return metrics

    def predict(self, X: pd.DataFrame) -> dict:
        """
        Make a prediction for new data.

        Args:
            X: Feature DataFrame

        Returns:
            Dictionary with prediction and probability
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]

        direction = "up" if prediction == 1 else "down"
        probability = probabilities[1] if prediction == 1 else probabilities[0]

        return {
            "direction": direction,
            "probability": float(probability),
            "raw_prediction": int(prediction),
            "probabilities": {
                "down": float(probabilities[0]),
                "up": float(probabilities[1]),
            },
        }

    def save(self, path: str | Path) -> None:
        """
        Save the model to disk.

        Args:
            path: File path for saving
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save untrained model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model": self.model,
            "scaler": self.scaler,
            "config": self.config,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str | Path) -> None:
        """
        Load a model from disk.

        Args:
            path: File path to load from
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.scaler = state["scaler"]
        self.config = state["config"]
        self._is_fitted = True

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted
