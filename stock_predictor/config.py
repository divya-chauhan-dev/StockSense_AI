"""Configuration settings for the stock predictor system."""

from dataclasses import dataclass, field


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""

    rsi_period: int = 14
    sma_periods: list[int] = field(default_factory=lambda: [20, 50])
    ema_periods: list[int] = field(default_factory=lambda: [12, 26])
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0


@dataclass
class ModelConfig:
    """Configuration for the XGBoost model."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


@dataclass
class DecisionConfig:
    """Configuration for the decision engine."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "ml_prediction": 0.40,
            "rsi": 0.15,
            "macd": 0.15,
            "moving_averages": 0.15,
            "bollinger": 0.15,
        }
    )
    buy_threshold: float = 0.3
    sell_threshold: float = -0.3


@dataclass
class Config:
    """Main configuration container."""

    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    default_horizon: int = 5
