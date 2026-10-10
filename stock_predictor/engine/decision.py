"""Decision engine for combining ML predictions with technical indicators."""

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from stock_predictor.config import DecisionConfig
from stock_predictor.indicators import get_all_signals


class Recommendation(Enum):
    """Trading recommendation."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class Decision:
    """Result of the decision engine analysis."""

    recommendation: Recommendation
    confidence: float
    score: float
    ml_prediction: dict
    indicator_signals: dict
    signal_breakdown: dict


class DecisionEngine:
    """
    Combines ML predictions with technical indicators to produce recommendations.

    The engine uses configurable weights to blend different signal sources
    into a final score, which is then mapped to a Buy/Hold/Sell recommendation.
    """

    def __init__(self, config: DecisionConfig | None = None):
        """
        Initialize the decision engine.

        Args:
            config: Decision engine configuration
        """
        self.config = config or DecisionConfig()

    def analyze(
        self,
        row: pd.Series,
        ml_prediction: dict,
    ) -> Decision:
        """
        Analyze a data point and produce a recommendation.

        Args:
            row: Series containing Close price and indicator values
            ml_prediction: Dict with 'direction' and 'probability' keys

        Returns:
            Decision object with recommendation and details
        """
        indicator_signals = get_all_signals(row)

        ml_signal = self._convert_ml_to_signal(ml_prediction)

        signal_breakdown = {
            "ml_prediction": ml_signal,
            **indicator_signals,
        }

        weighted_score = self._compute_weighted_score(signal_breakdown)

        recommendation = self._score_to_recommendation(weighted_score)

        confidence = self._compute_confidence(signal_breakdown, weighted_score)

        return Decision(
            recommendation=recommendation,
            confidence=confidence,
            score=weighted_score,
            ml_prediction=ml_prediction,
            indicator_signals=indicator_signals,
            signal_breakdown=signal_breakdown,
        )

    def _convert_ml_to_signal(self, ml_prediction: dict) -> float:
        """
        Convert ML prediction to a signal in [-1, 1].

        Args:
            ml_prediction: Dict with direction and probability

        Returns:
            Signal value
        """
        direction = ml_prediction.get("direction", "hold")
        probability = ml_prediction.get("probability", 0.5)

        confidence_factor = (probability - 0.5) * 2

        if direction == "up":
            return confidence_factor
        elif direction == "down":
            return -confidence_factor
        else:
            return 0.0

    def _compute_weighted_score(self, signals: dict) -> float:
        """
        Compute weighted average of all signals.

        Args:
            signals: Dictionary of signal values

        Returns:
            Weighted score in [-1, 1]
        """
        weights = self.config.weights
        total_weight = 0.0
        weighted_sum = 0.0

        for signal_name, signal_value in signals.items():
            weight = weights.get(signal_name, 0.0)
            if weight > 0 and signal_value is not None:
                weighted_sum += weight * signal_value
                total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _score_to_recommendation(self, score: float) -> Recommendation:
        """
        Convert score to recommendation.

        Args:
            score: Weighted score

        Returns:
            Recommendation enum value
        """
        if score >= self.config.buy_threshold:
            return Recommendation.BUY
        elif score <= self.config.sell_threshold:
            return Recommendation.SELL
        else:
            return Recommendation.HOLD

    def _compute_confidence(self, signals: dict, weighted_score: float) -> float:
        """
        Compute confidence based on signal agreement.

        Higher confidence when signals agree, lower when mixed.

        Args:
            signals: Dictionary of signal values
            weighted_score: Final weighted score

        Returns:
            Confidence in [0, 1]
        """
        signal_values = [v for v in signals.values() if v is not None]
        if not signal_values:
            return 0.0

        same_direction = 0
        for sig in signal_values:
            if (sig > 0 and weighted_score > 0) or (sig < 0 and weighted_score < 0):
                same_direction += 1
            elif sig == 0:
                same_direction += 0.5

        agreement = same_direction / len(signal_values)

        score_strength = abs(weighted_score)

        confidence = (agreement * 0.6) + (score_strength * 0.4)

        return min(1.0, max(0.0, confidence))
