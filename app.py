"""Flask Web Application for Stock Market Decision Support System."""

from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import json

from stock_predictor import StockAnalyzer
from stock_predictor.indicators import compute_all_indicators

app = Flask(__name__)

analyzer = StockAnalyzer()
trained_tickers = {}


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    """Analyze a stock and return recommendations."""
    data = request.get_json()
    ticker = data.get('ticker', '').upper()

    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        if ticker not in trained_tickers:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3 * 365)

            metrics = analyzer.train(
                ticker=ticker,
                start=start_date,
                end=end_date,
                horizon=5,
            )
            trained_tickers[ticker] = metrics

        result = analyzer.analyze(ticker)

        df = analyzer.fetcher.fetch_latest(ticker, period="6mo")
        df_indicators = compute_all_indicators(df)

        chart_data = {
            'dates': df_indicators.index.strftime('%Y-%m-%d').tolist(),
            'open': df_indicators['Open'].tolist(),
            'high': df_indicators['High'].tolist(),
            'low': df_indicators['Low'].tolist(),
            'close': df_indicators['Close'].tolist(),
            'volume': df_indicators['Volume'].tolist(),
            'rsi': df_indicators['rsi'].fillna(0).tolist(),
            'macd': df_indicators['macd'].fillna(0).tolist(),
            'macd_signal': df_indicators['macd_signal'].fillna(0).tolist(),
            'macd_histogram': df_indicators['macd_histogram'].fillna(0).tolist(),
            'bb_upper': df_indicators['bb_upper'].fillna(0).tolist(),
            'bb_lower': df_indicators['bb_lower'].fillna(0).tolist(),
            'sma_20': df_indicators['sma_20'].fillna(0).tolist(),
            'sma_50': df_indicators['sma_50'].fillna(0).tolist(),
        }

        return jsonify({
            'success': True,
            'result': result,
            'chart_data': chart_data,
            'training_metrics': trained_tickers.get(ticker, {}),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/indicators/<ticker>')
def get_indicators(ticker):
    """Get technical indicators for a stock."""
    ticker = ticker.upper()

    try:
        df = analyzer.fetcher.fetch_latest(ticker, period="3mo")
        df_indicators = compute_all_indicators(df)

        latest = df_indicators.iloc[-1]

        return jsonify({
            'success': True,
            'indicators': {
                'rsi': round(latest['rsi'], 2) if not pd.isna(latest['rsi']) else None,
                'macd': round(latest['macd'], 4) if not pd.isna(latest['macd']) else None,
                'sma_20': round(latest['sma_20'], 2) if not pd.isna(latest['sma_20']) else None,
                'sma_50': round(latest['sma_50'], 2) if not pd.isna(latest['sma_50']) else None,
                'bb_upper': round(latest['bb_upper'], 2) if not pd.isna(latest['bb_upper']) else None,
                'bb_lower': round(latest['bb_lower'], 2) if not pd.isna(latest['bb_lower']) else None,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import pandas as pd
    app.run(debug=True, port=5000)
