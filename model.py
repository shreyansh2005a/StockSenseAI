import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                   # non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# DEFAULT CONFIG  — every key can be overridden by the UI
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "ticker"      : "AAPL",
    "start_date"  : "2019-01-01",
    "end_date"    : "2026-03-01",   # up-to-date so model sees recent prices
    "sequence_len": 90,             # longer look-back reduces lag
    "test_split"  : 0.2,
    "epochs"      : 100,
    "batch_size"  : 32,
    "lstm_units"  : [256, 128],     # larger network handles wider price range
    "dropout_rate": 0.2,
    "future_days" : 30,
    "use_real"    : True,
    "output_dir"  : ".",            # saves chart & model next to this file
}

# 17 features — Momentum & Volatility fix the one-step lag problem
FEATURE_COLS = [
    "Close", "Open", "High", "Low", "Volume",
    "SMA_10", "SMA_50", "EMA_20",
    "RSI", "MACD", "Signal",
    "BB_Upper", "BB_Lower",
    "Return", "Log_Return",
    "Momentum",   # 5-day price difference
    "Volatility", # 10-day rolling std
]


# ─────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_data(ticker: str, start: str, end: str, use_real: bool = True):
    """
    Returns (df, is_real_data).
    Tries Yahoo Finance first; falls back to GBM simulation.
    """
    if use_real:
        try:
            import yfinance as yf
            print(f"[INFO] Downloading {ticker} from Yahoo Finance …")
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("yfinance returned empty data.")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.dropna(inplace=True)
            print(f"[INFO] Loaded {len(df)} rows from Yahoo Finance.")
            return df, True
        except Exception as e:
            print(f"[WARN] yfinance failed: {e}  →  using simulation.")

    # Geometric Brownian Motion fallback
    print(f"[INFO] Simulating {ticker} data ({start} → {end})")
    np.random.seed(42)
    dates  = pd.bdate_range(start=start, end=end)
    n      = len(dates)
    prices = 150.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, n)))
    df = pd.DataFrame({
        "Open"  : prices * np.random.uniform(0.995, 1.005, n),
        "High"  : prices * np.random.uniform(1.000, 1.020, n),
        "Low"   : prices * np.random.uniform(0.980, 1.000, n),
        "Close" : prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, n).astype(float),
    }, index=dates)
    return df, False


# ─────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df["SMA_10"]    = close.rolling(10).mean()
    df["SMA_50"]    = close.rolling(50).mean()
    df["EMA_20"]    = close.ewm(span=20, adjust=False).mean()

    delta           = close.diff()
    gain            = delta.clip(lower=0).rolling(14).mean()
    loss            = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]       = 100 - 100 / (1 + gain / (loss + 1e-9))

    ema12           = close.ewm(span=12, adjust=False).mean()
    ema26           = close.ewm(span=26, adjust=False).mean()
    df["MACD"]      = ema12 - ema26
    df["Signal"]    = df["MACD"].ewm(span=9, adjust=False).mean()

    sma20           = close.rolling(20).mean()
    std20           = close.rolling(20).std()
    df["BB_Upper"]  = sma20 + 2 * std20
    df["BB_Lower"]  = sma20 - 2 * std20

    df["Return"]    = close.pct_change()
    df["Log_Return"]= np.log(close / close.shift(1))

    # Critical: these two features teach the model price DIRECTION
    df["Momentum"]  = close - close.shift(5)
    df["Volatility"]= close.rolling(10).std()

    df.dropna(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, sequence_len: int, test_split: float):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FEATURE_COLS].values)

    X, y = [], []
    for i in range(sequence_len, len(scaled)):
        X.append(scaled[i - sequence_len : i])
        y.append(scaled[i, 0])     # index 0 = Close

    X, y  = np.array(X), np.array(y)
    split = int(len(X) * (1 - test_split))

    return (
        X[:split], X[split:],
        y[:split], y[split:],
        scaler,
        split + sequence_len        
    )


# ─────────────────────────────────────────────────────────────
# STEP 4: BUILD MODEL
# ─────────────────────────────────────────────────────────────
def build_model(sequence_len: int, lstm_units: list, dropout_rate: float):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    )
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        Bidirectional(
            LSTM(lstm_units[0], return_sequences=True),
            input_shape=(sequence_len, len(FEATURE_COLS))
        ),
        BatchNormalization(),
        Dropout(dropout_rate),

        LSTM(lstm_units[1], return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"]
    )
    return model


# ─────────────────────────────────────────────────────────────
# STEP 5: TRAIN
# ─────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train,
                epochs, batch_size, ckpt_path):
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )

    callbacks = [
        EarlyStopping(patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-7, verbose=1),
        ModelCheckpoint(ckpt_path, save_best_only=True, verbose=0),
    ]

    return model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
        shuffle=False           # preserve temporal order
    )


# ─────────────────────────────────────────────────────────────
# STEP 6: EVALUATE
# ─────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, scaler):
    from sklearn.metrics import (mean_squared_error,
                                  mean_absolute_error, r2_score)

    n     = len(FEATURE_COLS)
    y_s   = model.predict(X_test, verbose=0).flatten()

    def inv(arr):
        d = np.zeros((len(arr), n)); d[:, 0] = arr
        return scaler.inverse_transform(d)[:, 0]

    y_real = inv(y_test)
    y_pred = inv(y_s)

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAE" : float(mean_absolute_error(y_real, y_pred)),
        "R2"  : float(r2_score(y_real, y_pred)),
        "MAPE": float(np.mean(np.abs((y_real - y_pred) /
                                      (y_real + 1e-9))) * 100),
    }

    print(f"  {'RMSE':>8}: ${metrics['RMSE']:.2f}")
    print(f"  {'MAE':>8}: ${metrics['MAE']:.2f}")
    print(f"  {'R²':>8}: {metrics['R2']:.4f}")
    print(f"  {'MAPE':>8}: {metrics['MAPE']:.2f}%")

    return y_real, y_pred, metrics


# ─────────────────────────────────────────────────────────────
# STEP 7: FUTURE FORECAST
# ─────────────────────────────────────────────────────────────
def forecast_future(model, scaled_all, scaler, sequence_len, future_days):
    n   = len(FEATURE_COLS)
    seq = scaled_all[-sequence_len:].copy()
    out = []

    for _ in range(future_days):
        inp  = seq[-sequence_len:].reshape(1, sequence_len, n)
        pred = model.predict(inp, verbose=0)[0, 0]
        out.append(pred)
        row    = seq[-1].copy()
        row[0] = pred
        seq    = np.vstack([seq, row])

    dummy = np.zeros((future_days, n))
    dummy[:, 0] = out
    return scaler.inverse_transform(dummy)[:, 0]


# ─────────────────────────────────────────────────────────────
# STEP 8: VISUALIZE
# ─────────────────────────────────────────────────────────────
def plot_results(df, history, y_real, y_pred,
                 future_prices, split_idx, cfg):

    ticker      = cfg["ticker"]
    future_days = cfg["future_days"]
    output_dir  = cfg["output_dir"]
    metrics     = cfg.get("_metrics", {})

    C = dict(
        bg="#0d1117", panel="#161b22", grid="#30363d", text="#e6edf3",
        blue="#58a6ff", green="#3fb950", red="#f85149", yellow="#d29922"
    )

    fig = plt.figure(figsize=(20, 16), facecolor=C["bg"])
    fig.suptitle(
        f"AI Stock Prediction — {ticker}  |  Bidirectional LSTM  |  17 Features",
        fontsize=17, color="white", fontweight="bold", y=0.98
    )
    gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    def style(ax, title):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text"], labelsize=9)
        for s in ax.spines.values():
            s.set_edgecolor(C["grid"])
        ax.grid(color=C["grid"], linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_title(title, color=C["text"], fontsize=11, pad=8)

    # 1. Price chart
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, "Actual vs Predicted Close Price")
    dates = df.index
    ax1.plot(dates[:split_idx], df["Close"].values[:split_idx],
             color=C["blue"], lw=1.1, alpha=0.75, label="Train (Actual)")
    test_dates = dates[split_idx : split_idx + len(y_real)]
    ax1.plot(test_dates, y_real, color=C["green"], lw=1.6, label="Test (Actual)")
    ax1.plot(test_dates, y_pred, color=C["red"],   lw=1.6,
             linestyle="--", label="Test (Predicted)")
    future_dates = pd.bdate_range(
        start=dates[-1] + pd.Timedelta(days=1), periods=future_days)
    ax1.plot(future_dates, future_prices, color=C["yellow"], lw=2,
             linestyle=":", marker="o", markersize=3,
             label=f"{future_days}-Day Forecast")
    ax1.axvline(dates[split_idx], color=C["grid"],
                lw=1.4, linestyle="--", alpha=0.9)
    ax1.legend(facecolor=C["panel"], edgecolor=C["grid"],
               labelcolor=C["text"], fontsize=9)
    ax1.set_ylabel("Price (USD)", color=C["text"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 2. Loss
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, "Training vs Validation Loss")
    ep = range(1, len(history.history["loss"]) + 1)
    ax2.plot(ep, history.history["loss"],     color=C["blue"], lw=1.5, label="Train")
    ax2.plot(ep, history.history["val_loss"], color=C["red"],  lw=1.5, label="Val")
    ax2.legend(facecolor=C["panel"], edgecolor=C["grid"],
               labelcolor=C["text"], fontsize=9)
    ax2.set_xlabel("Epoch", color=C["text"])
    ax2.set_ylabel("Huber Loss", color=C["text"])

    # 3. Scatter
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, "Actual vs Predicted (Scatter)")
    ax3.scatter(y_real, y_pred, color=C["blue"], alpha=0.35, s=10)
    lo, hi = min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())
    ax3.plot([lo, hi], [lo, hi], color=C["green"], lw=1.5,
             linestyle="--", label="Perfect fit")
    ax3.set_xlabel("Actual Price", color=C["text"])
    ax3.set_ylabel("Predicted Price", color=C["text"])
    ax3.legend(facecolor=C["panel"], edgecolor=C["grid"],
               labelcolor=C["text"], fontsize=9)

    # 4. Residuals
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, "Prediction Residuals")
    res = y_real - y_pred
    ax4.bar(range(len(res)), res,
            color=[C["green"] if r >= 0 else C["red"] for r in res],
            width=1, alpha=0.8)
    ax4.axhline(0, color=C["text"], lw=0.8, linestyle="--")
    ax4.set_xlabel("Test Sample Index", color=C["text"])
    ax4.set_ylabel("Residual (USD)", color=C["text"])

    # 5. Metrics card
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(C["panel"]); ax5.axis("off")
    ax5.text(0.5, 0.93, "Model Performance Metrics",
             color=C["text"], ha="center", fontsize=12,
             fontweight="bold", transform=ax5.transAxes)
    items = [
        ("RMSE", f"${metrics.get('RMSE', 0):.2f}",   C["red"]),
        ("MAE",  f"${metrics.get('MAE',  0):.2f}",   C["yellow"]),
        ("R²",   f"{metrics.get('R2',    0):.4f}",   C["green"]),
        ("MAPE", f"{metrics.get('MAPE',  0):.2f}%",  C["blue"]),
    ]
    for i, (lbl, val, col) in enumerate(items):
        y0 = 0.72 - i * 0.17
        ax5.text(0.12, y0, lbl, color=C["text"], fontsize=11,
                 transform=ax5.transAxes, va="center")
        ax5.text(0.58, y0, val, color=col, fontsize=15,
                 fontweight="bold", transform=ax5.transAxes, va="center")

    out = os.path.join(output_dir, "stock_prediction_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[INFO] Chart saved → {out}")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE  — called directly OR from app.py (UI)
# ─────────────────────────────────────────────────────────────
def run_pipeline(config: dict = None):
    """
    Run the full LSTM pipeline.

    Parameters
    ----------
    config : dict
        UI can pass any of these keys:
            ticker, start_date, end_date, sequence_len,
            test_split, epochs, batch_size, lstm_units,
            dropout_rate, future_days, use_real, output_dir

    Returns
    -------
    dict:  metrics, future_prices, future_dates,
           chart_path, df, y_real, y_pred,
           split_idx, history, scaler, is_real_data
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    print("=" * 65)
    print(f"  StockSense AI  |  {cfg['ticker']}  |  LSTM Pipeline")
    print("=" * 65)

    # Always start fresh — delete stale checkpoints
    for f in ["best_model.keras",
              os.path.join(cfg["output_dir"], "best_model.keras")]:
        if os.path.exists(f):
            os.remove(f)
            print(f"[INFO] Removed old checkpoint: {f}")

    # 1. Load data
    df, is_real = load_data(
        cfg["ticker"], cfg["start_date"],
        cfg["end_date"], cfg["use_real"]
    )

    # 2. Feature engineering
    df = add_technical_indicators(df)
    print(f"[INFO] {len(FEATURE_COLS)} features, {len(df)} trading days")

    # 3. Preprocess
    X_train, X_test, y_train, y_test, scaler, split_idx = preprocess(
        df, cfg["sequence_len"], cfg["test_split"]
    )
    print(f"[INFO] Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

    # 4. Build
    model = build_model(
        cfg["sequence_len"], cfg["lstm_units"], cfg["dropout_rate"]
    )
    model.summary()

    # 5. Train
    print("\n[INFO] Training …")
    ckpt = os.path.join(cfg["output_dir"], "best_model.keras")
    history = train_model(
        model, X_train, y_train,
        cfg["epochs"], cfg["batch_size"], ckpt
    )

    # 6. Evaluate
    print("\n[INFO] Evaluation Metrics:")
    y_real, y_pred, metrics = evaluate_model(model, X_test, y_test, scaler)
    cfg["_metrics"] = metrics

    # 7. Forecast
    scaled_all    = scaler.transform(df[FEATURE_COLS].values)
    future_prices = forecast_future(
        model, scaled_all, scaler,
        cfg["sequence_len"], cfg["future_days"]
    )
    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=cfg["future_days"]
    )

    top5 = ["$" + str(round(p, 2)) for p in future_prices[:5]]
    print(f"\n[INFO] {cfg['future_days']}-day forecast (first 5): {top5} …")

    # 8. Plot & save chart
    chart_path = plot_results(
        df, history, y_real, y_pred,
        future_prices, split_idx, cfg
    )

    # 9. Save model
    model_out = os.path.join(cfg["output_dir"], "lstm_stock_model.keras")
    model.save(model_out)
    print(f"[INFO] Model saved → {model_out}")
    print("\n✅  Pipeline complete!")

    return {
        "metrics"      : metrics,
        "future_prices": future_prices,
        "future_dates" : future_dates,
        "chart_path"   : chart_path,
        "df"           : df,
        "y_real"       : y_real,
        "y_pred"       : y_pred,
        "split_idx"    : split_idx,
        "history"      : history,
        "scaler"       : scaler,
        "is_real_data" : is_real,
    }


if __name__ == "__main__":
    run_pipeline()
