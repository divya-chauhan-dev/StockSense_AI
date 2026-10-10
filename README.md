# AI-Driven Stock Market Decision Support System

An intelligent stock market decision support system that integrates XGBoost-based price prediction with classical technical analysis indicators to generate actionable Buy/Hold/Sell recommendations.

## Features

- **ML-Based Prediction**: XGBoost model trained on historical OHLCV data to predict short-term price movements
- **Technical Indicators**: RSI, SMA/EMA, MACD, and Bollinger Bands for multi-dimensional market analysis
- **Weighted Decision Engine**: Configurable fusion of ML predictions and technical signals
- **Confidence Scoring**: Recommendations include confidence levels based on signal agreement
- **Yahoo Finance Integration**: Automatic data fetching for any stock ticker

## Project Structure

```
FND/
├── app.py                   # Flask web application
├── example.py               # CLI usage example
├── pyproject.toml           # Package configuration
├── templates/
│   └── index.html           # Web UI template
├── static/
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       └── app.js           # Frontend JavaScript
└── stock_predictor/
    ├── __init__.py          # Package exports
    ├── config.py            # Configuration settings
    ├── api.py               # Main StockAnalyzer API
    ├── data/
    │   └── fetcher.py       # Yahoo Finance data fetching
    ├── indicators/
    │   ├── rsi.py           # Relative Strength Index
    │   ├── moving_averages.py # SMA and EMA
    │   ├── macd.py          # MACD indicator
    │   └── bollinger.py     # Bollinger Bands
    ├── models/
    │   ├── feature_engineer.py # Feature engineering
    │   └── xgboost_model.py # XGBoost predictor
    └── engine/
        └── decision.py      # Weighted decision engine
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Step 1: Clone or Copy the Project

```bash
# If using git
git clone <repository-url>
cd FND

# Or simply copy the FND folder to your machine
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install yfinance pandas numpy xgboost scikit-learn flask
```

Or install the package in development mode:

```bash
pip install -e .
pip install flask
```

## Quick Start

### Option 1: Run the Web Application (Recommended)

```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

**Features of the Web UI:**
- Search for any stock ticker
- View interactive price charts with technical indicators
- Get AI-powered Buy/Hold/Sell recommendations
- See signal breakdown from each indicator
- View RSI and MACD charts

![Web UI Screenshot](docs/screenshot.png)

### Option 2: Use Python API

```python
from stock_predictor import StockAnalyzer

# Initialize the analyzer
analyzer = StockAnalyzer()

# Train the model on historical data
analyzer.train(
    ticker='AAPL',
    start='2020-01-01',
    end='2024-01-01',
    horizon=5  # Predict 5 days ahead
)

# Get recommendation for a stock
result = analyzer.analyze('AAPL')

print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Current Price: ${result['price']:.2f}")
```

### Run the Example Script

```bash
python example.py
```

## API Reference

### StockAnalyzer

The main class for stock analysis.

#### Methods

| Method | Description |
|--------|-------------|
| `train(ticker, start, end, horizon)` | Train the ML model on historical data |
| `analyze(ticker, horizon, period)` | Analyze a stock and get recommendation |
| `get_indicators(ticker, period)` | Get technical indicator values |
| `save_model(path)` | Save trained model to disk |
| `load_model(path)` | Load trained model from disk |
| `batch_analyze(tickers, period)` | Analyze multiple stocks |

#### Example: Full Workflow

```python
from stock_predictor import StockAnalyzer, Config
from stock_predictor.config import DecisionConfig

# Custom configuration
config = Config()
config.decision.weights = {
    'ml_prediction': 0.50,  # Increase ML weight
    'rsi': 0.15,
    'macd': 0.15,
    'moving_averages': 0.10,
    'bollinger': 0.10,
}

analyzer = StockAnalyzer(config=config)

# Train
metrics = analyzer.train('MSFT', '2019-01-01', '2024-01-01', horizon=5)
print(f"Training accuracy: {metrics['train_accuracy']:.2%}")

# Save model for later use
analyzer.save_model('models/msft_model.pkl')

# Analyze
result = analyzer.analyze('MSFT')
print(result)
```

#### Example: Load Saved Model

```python
from stock_predictor import StockAnalyzer

analyzer = StockAnalyzer()
analyzer.load_model('models/msft_model.pkl')

# Now you can analyze without retraining
result = analyzer.analyze('MSFT')
```

## Understanding the Output

### Recommendation Types

| Recommendation | Score Range | Meaning |
|----------------|-------------|---------|
| **BUY** | > 0.3 | Bullish signals dominate; consider buying |
| **HOLD** | -0.3 to 0.3 | Mixed signals; maintain current position |
| **SELL** | < -0.3 | Bearish signals dominate; consider selling |

### Signal Breakdown

Each analysis includes individual signals from:

- **ml_prediction**: XGBoost model prediction (-1 to +1)
- **rsi**: RSI momentum signal (-1 to +1)
- **macd**: MACD trend signal (-1 to +1)
- **moving_averages**: SMA crossover signal (-1 to +1)
- **bollinger**: Bollinger Band position signal (-1 to +1)

### Sample Output

```python
{
    'ticker': 'AAPL',
    'date': '2024-05-10',
    'price': 182.45,
    'recommendation': 'BUY',
    'confidence': 0.72,
    'score': 0.45,
    'ml_prediction': {
        'direction': 'up',
        'probability': 0.68
    },
    'indicator_signals': {
        'rsi': 0.25,
        'moving_averages': 0.35,
        'macd': 0.18,
        'bollinger': 0.12
    }
}
```

## Technical Indicators Explained

| Indicator | Description | Bullish Signal | Bearish Signal |
|-----------|-------------|----------------|----------------|
| **RSI** | Measures momentum (0-100) | RSI < 30 (oversold) | RSI > 70 (overbought) |
| **SMA/EMA** | Trend-following averages | Price > MAs, short > long | Price < MAs, short < long |
| **MACD** | Trend momentum | MACD > Signal line | MACD < Signal line |
| **Bollinger Bands** | Volatility bands | Price near lower band | Price near upper band |

## Configuration

### Default Weights

```python
weights = {
    'ml_prediction': 0.40,    # 40% weight to ML model
    'rsi': 0.15,              # 15% to RSI
    'macd': 0.15,             # 15% to MACD
    'moving_averages': 0.15,  # 15% to moving averages
    'bollinger': 0.15,        # 15% to Bollinger Bands
}
```

### Customizing Configuration

```python
from stock_predictor import Config
from stock_predictor.config import IndicatorConfig, ModelConfig, DecisionConfig

config = Config(
    indicators=IndicatorConfig(
        rsi_period=14,
        sma_periods=[20, 50],
        ema_periods=[12, 26],
        bollinger_period=20,
        bollinger_std=2.0,
    ),
    model=ModelConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
    ),
    decision=DecisionConfig(
        buy_threshold=0.3,
        sell_threshold=-0.3,
    ),
    default_horizon=5,
)

analyzer = StockAnalyzer(config=config)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| yfinance | >= 0.2.0 | Stock data fetching |
| pandas | >= 2.0.0 | Data manipulation |
| numpy | >= 1.24.0 | Numerical operations |
| xgboost | >= 2.0.0 | ML model |
| scikit-learn | >= 1.3.0 | Preprocessing, cross-validation |
| flask | >= 2.0.0 | Web application framework |

### Frontend Libraries (CDN - No Installation Required)

| Library | Purpose |
|---------|---------|
| Chart.js | Interactive charts |
| Luxon | Date handling for charts |

## Limitations and Disclaimer

- **Not Financial Advice**: This system is for educational and research purposes only. Do not use it as the sole basis for investment decisions.
- **Past Performance**: Historical accuracy does not guarantee future results.
- **Market Conditions**: The model may perform differently in various market conditions (trending, volatile, sideways).
- **Data Quality**: Predictions depend on the quality and availability of Yahoo Finance data.

## Troubleshooting

### Common Issues

**1. No data returned for ticker**
```
ValueError: No data returned for ticker 'XYZ'
```
- Verify the ticker symbol is correct
- Check if the stock is listed on Yahoo Finance

**2. Insufficient data for training**
```
ValueError: Insufficient data for ticker 'AAPL'
```
- Use a longer date range (at least 1 year of data recommended)
- Ensure the stock has enough trading history

**3. Model not trained error**
```
RuntimeError: Model must be trained before analysis
```
- Call `analyzer.train()` before `analyzer.analyze()`
- Or load a pre-trained model with `analyzer.load_model()`

## Web Application API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/api/analyze` | POST | Analyze a stock (body: `{"ticker": "AAPL"}`) |
| `/api/indicators/<ticker>` | GET | Get indicator values for a stock |

### Example API Usage

```bash
# Analyze a stock
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Get indicators
curl http://localhost:5000/api/indicators/AAPL
```

## License

This project is provided for educational purposes.

## Contributing

Contributions are welcome! Areas for improvement:
- Add LSTM model option
- Implement backtesting framework
- Add more technical indicators
- Add user authentication
- Deploy to cloud (Heroku, AWS, etc.)
