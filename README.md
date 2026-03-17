# 📈 AI-Based Stock Price Prediction
## Real-Time Data Series Analysis using LSTM

---

## 🗂️ Project Structure

```
stock_prediction/
├── model.py            ← Main pipeline (run this)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python model.py
```

### 3. Switch to real Yahoo Finance data
Inside `model.py`, find **STEP 1: DATA LOADING** and:
- **Uncomment** the `yfinance` block
- **Comment out** the simulated data block

---

## 🧠 Model Architecture

```
Input (60 timesteps × 15 features)
        │
Bidirectional LSTM (128 units) → BatchNorm → Dropout(0.2)
        │
LSTM (64 units)                → BatchNorm → Dropout(0.2)
        │
Dense (32, ReLU)
        │
Dense (1)  ← Predicted Close Price
```

**Loss function:** Huber loss (robust to outliers)  
**Optimizer:** Adam with learning rate scheduling  
**Callbacks:** EarlyStopping + ReduceLROnPlateau

---

## 📊 Features Used (15 total)

| Category            | Features                                |
|---------------------|-----------------------------------------|
| OHLCV               | Open, High, Low, Close, Volume          |
| Moving Averages     | SMA_10, SMA_50, EMA_20                  |
| Momentum            | RSI (14-day)                            |
| Trend               | MACD, Signal Line                       |
| Volatility          | Bollinger Bands (Upper, Lower)          |
| Returns             | Daily Return, Log Return                |

---

## ⚙️ Configuration (top of model.py)

| Parameter      | Default  | Description                     |
|----------------|----------|---------------------------------|
| `TICKER`       | `AAPL`   | Stock symbol                    |
| `START_DATE`   | 2018     | Historical data start           |
| `END_DATE`     | 2024     | Historical data end             |
| `SEQUENCE_LEN` | 60       | Look-back window (days)         |
| `TEST_SPLIT`   | 0.2      | Fraction of data for testing    |
| `EPOCHS`       | 50       | Max training epochs             |
| `LSTM_UNITS`   | [128,64] | Units in each LSTM layer        |
| `FUTURE_DAYS`  | 30       | Days to forecast ahead          |

---

## 📈 Outputs

| File                              | Description                  |
|-----------------------------------|------------------------------|
| `stock_prediction_results.png`    | 5-panel visualization chart  |
| `lstm_stock_model.keras`          | Trained model (saved)        |

---

## 📋 Evaluation Metrics

- **RMSE** — Root Mean Squared Error (lower is better)
- **MAE**  — Mean Absolute Error (lower is better)
- **R²**   — Coefficient of Determination (closer to 1 is better)
- **MAPE** — Mean Absolute Percentage Error (lower is better)

---
