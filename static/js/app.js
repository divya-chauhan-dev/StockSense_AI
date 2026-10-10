let priceChart = null;
let rsiChart = null;
let macdChart = null;

function setTicker(ticker) {
    document.getElementById('tickerInput').value = ticker;
    analyzeStock();
}

async function analyzeStock() {
    const tickerInput = document.getElementById('tickerInput');
    const ticker = tickerInput.value.trim().toUpperCase();

    if (!ticker) {
        showError('Please enter a stock ticker symbol');
        return;
    }

    showLoading(true);
    hideError();

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ ticker }),
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to analyze stock');
        }

        displayResults(data);
    } catch (error) {
        showError(error.message);
        document.getElementById('resultsSection').classList.add('hidden');
    } finally {
        showLoading(false);
    }
}

function displayResults(data) {
    const { result, chart_data, training_metrics } = data;

    document.getElementById('metricTicker').textContent = result.ticker;
    document.getElementById('metricPrice').textContent = `$${result.price.toFixed(2)}`;
    document.getElementById('metricDate').textContent = result.date;
    document.getElementById('metricAccuracy').textContent =
        training_metrics.cv_accuracy_mean ? `${(training_metrics.cv_accuracy_mean * 100).toFixed(1)}%` : '-';

    updateRecommendation(result);
    updateMLPrediction(result.ml_prediction);
    updateSignals(result.signal_breakdown);
    createCharts(chart_data);

    document.getElementById('resultsSection').classList.remove('hidden');
}

function updateRecommendation(result) {
    const box = document.getElementById('recommendationBox');
    const icon = document.getElementById('recIcon');
    const text = document.getElementById('recText');
    const confidence = document.getElementById('recConfidence');
    const score = document.getElementById('recScore');

    box.className = 'recommendation-box ' + result.recommendation.toLowerCase();

    const icons = { BUY: '🟢', SELL: '🔴', HOLD: '🟡' };
    icon.textContent = icons[result.recommendation] || '⚪';
    text.textContent = result.recommendation;
    confidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
    score.textContent = result.score.toFixed(3);
}

function updateMLPrediction(ml) {
    const direction = document.getElementById('mlDirection');
    const probability = document.getElementById('mlProbability');

    const emoji = ml.direction === 'up' ? '📈' : '📉';
    direction.textContent = `${emoji} ${ml.direction.toUpperCase()}`;
    direction.className = 'ml-direction ' + ml.direction;
    probability.innerHTML = `<span>Probability:</span> <span>${(ml.probability * 100).toFixed(1)}%</span>`;
}

function updateSignals(signals) {
    const list = document.getElementById('signalsList');
    list.innerHTML = '';

    const signalNames = {
        ml_prediction: 'ML Prediction',
        rsi: 'RSI',
        macd: 'MACD',
        moving_averages: 'Moving Averages',
        bollinger: 'Bollinger Bands'
    };

    for (const [key, value] of Object.entries(signals)) {
        const item = document.createElement('div');
        item.className = 'signal-item';

        const valueClass = value > 0.1 ? 'positive' : value < -0.1 ? 'negative' : 'neutral';
        const sign = value > 0 ? '+' : '';

        item.innerHTML = `
            <span class="signal-name">${signalNames[key] || key}</span>
            <span class="signal-value ${valueClass}">${sign}${value.toFixed(3)}</span>
        `;

        list.appendChild(item);
    }
}

function createCharts(data) {
    createPriceChart(data);
    createRSIChart(data);
    createMACDChart(data);
}

function createPriceChart(data) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    if (priceChart) {
        priceChart.destroy();
    }

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Close Price',
                    data: data.close,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                },
                {
                    label: 'SMA 20',
                    data: data.sma_20,
                    borderColor: '#f59e0b',
                    borderWidth: 1.5,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                },
                {
                    label: 'SMA 50',
                    data: data.sma_50,
                    borderColor: '#8b5cf6',
                    borderWidth: 1.5,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                },
                {
                    label: 'BB Upper',
                    data: data.bb_upper,
                    borderColor: 'rgba(156, 163, 175, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0,
                },
                {
                    label: 'BB Lower',
                    data: data.bb_lower,
                    borderColor: 'rgba(156, 163, 175, 0.5)',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    fill: '-1',
                    backgroundColor: 'rgba(156, 163, 175, 0.1)',
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                    },
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: $${context.parsed.y.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false,
                    },
                    ticks: {
                        maxTicksLimit: 10,
                    },
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(0);
                        }
                    }
                },
            },
        },
    });
}

function createRSIChart(data) {
    const ctx = document.getElementById('rsiChart').getContext('2d');

    if (rsiChart) {
        rsiChart.destroy();
    }

    rsiChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'RSI',
                    data: data.rsi,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                annotation: {
                    annotations: {
                        overbought: {
                            type: 'line',
                            yMin: 70,
                            yMax: 70,
                            borderColor: '#ef4444',
                            borderWidth: 1,
                            borderDash: [5, 5],
                        },
                        oversold: {
                            type: 'line',
                            yMin: 30,
                            yMax: 30,
                            borderColor: '#10b981',
                            borderWidth: 1,
                            borderDash: [5, 5],
                        },
                    },
                },
            },
            scales: {
                x: {
                    display: false,
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    },
                },
            },
        },
    });

    addHorizontalLine(rsiChart, 70, '#ef4444');
    addHorizontalLine(rsiChart, 30, '#10b981');
}

function createMACDChart(data) {
    const ctx = document.getElementById('macdChart').getContext('2d');

    if (macdChart) {
        macdChart.destroy();
    }

    const histogramColors = data.macd_histogram.map(v => v >= 0 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(239, 68, 68, 0.6)');

    macdChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'Histogram',
                    data: data.macd_histogram,
                    backgroundColor: histogramColors,
                    borderWidth: 0,
                    order: 2,
                },
                {
                    label: 'MACD',
                    data: data.macd,
                    type: 'line',
                    borderColor: '#2563eb',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    order: 1,
                },
                {
                    label: 'Signal',
                    data: data.macd_signal,
                    type: 'line',
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    order: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 10,
                        font: {
                            size: 10,
                        },
                    },
                },
            },
            scales: {
                x: {
                    display: false,
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    },
                },
            },
        },
    });
}

function addHorizontalLine(chart, yValue, color) {
    const originalDraw = chart.draw;
    chart.draw = function() {
        originalDraw.apply(this, arguments);

        const ctx = this.ctx;
        const yAxis = this.scales.y;
        const xAxis = this.scales.x;

        const y = yAxis.getPixelForValue(yValue);

        ctx.save();
        ctx.beginPath();
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.moveTo(xAxis.left, y);
        ctx.lineTo(xAxis.right, y);
        ctx.stroke();
        ctx.restore();
    };
    chart.draw();
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    const btn = document.getElementById('analyzeBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoader = btn.querySelector('.btn-loader');

    if (show) {
        overlay.classList.remove('hidden');
        btn.disabled = true;
        btnText.textContent = 'Analyzing...';
        btnLoader.classList.remove('hidden');
    } else {
        overlay.classList.add('hidden');
        btn.disabled = false;
        btnText.textContent = 'Analyze';
        btnLoader.classList.add('hidden');
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    document.getElementById('errorMessage').classList.add('hidden');
}

document.getElementById('tickerInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        analyzeStock();
    }
});
