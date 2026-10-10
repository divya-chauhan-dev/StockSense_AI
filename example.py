"""Example usage of the Stock Predictor system."""

from stock_predictor import StockAnalyzer, Config


def main():
    """Demonstrate the stock prediction system."""

    print("=" * 60)
    print("Stock Market Decision Support System - Demo")
    print("=" * 60)

    analyzer = StockAnalyzer()

    ticker = "AAPL"
    print(f"\n[1] Training model on {ticker} historical data...")
    print("    Period: 2020-01-01 to 2024-01-01")
    print("    Horizon: 5 days")

    metrics = analyzer.train(
        ticker=ticker,
        start="2020-01-01",
        end="2024-01-01",
        horizon=5,
    )

    print(f"\n    Training Results:")
    print(f"    - Samples: {metrics['samples']}")
    print(f"    - Train Accuracy: {metrics['train_accuracy']:.2%}")
    if "cv_accuracy_mean" in metrics:
        print(f"    - CV Accuracy: {metrics['cv_accuracy_mean']:.2%} (+/- {metrics['cv_accuracy_std']:.2%})")

    if "top_features" in metrics:
        print("\n    Top Features:")
        for feat, importance in list(metrics["top_features"].items())[:5]:
            print(f"      - {feat}: {importance:.4f}")

    print(f"\n[2] Analyzing {ticker} for trading recommendation...")

    result = analyzer.analyze(ticker)

    print(f"\n    Analysis Results for {result['ticker']}:")
    print(f"    Date: {result['date']}")
    print(f"    Price: ${result['price']:.2f}")
    print(f"\n    >>> RECOMMENDATION: {result['recommendation']} <<<")
    print(f"    Confidence: {result['confidence']:.1%}")
    print(f"    Score: {result['score']:.3f}")

    print(f"\n    ML Prediction:")
    ml = result["ml_prediction"]
    print(f"      Direction: {ml['direction']}")
    print(f"      Probability: {ml['probability']:.1%}")

    print(f"\n    Indicator Signals (range: -1 to +1):")
    for indicator, signal in result["indicator_signals"].items():
        direction = "bullish" if signal > 0.1 else "bearish" if signal < -0.1 else "neutral"
        print(f"      - {indicator}: {signal:+.3f} ({direction})")

    print("\n[3] Technical indicator snapshot (last 5 days):")
    indicators_df = analyzer.get_indicators(ticker, period="1mo")
    cols = ["Close", "rsi", "macd", "bb_percent_b"]
    print(indicators_df[cols].tail().to_string())

    print("\n[4] Batch analysis of multiple stocks...")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    results = analyzer.batch_analyze(tickers)

    print("\n    Summary:")
    print(f"    {'Ticker':<8} {'Price':>10} {'Recommendation':<12} {'Confidence':>10}")
    print("    " + "-" * 45)
    for r in results:
        if "error" not in r:
            print(f"    {r['ticker']:<8} ${r['price']:>9.2f} {r['recommendation']:<12} {r['confidence']:>9.1%}")
        else:
            print(f"    {r['ticker']:<8} Error: {r['error']}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
