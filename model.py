import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

DEFAULT_CONFIG = {
    "ticker": "AAPL",
    "start_date": "2019-01-01",
    "end_date": "2026-03-01",
    "sequence_len": 60,
    "test_split": 0.2,
    "epochs": 100,
    "batch_size": 32,
    "lstm_units": [128, 64],
    "dropout_rate": 0.2,
    "future_days": 30,
    "use_real": True,
    "output_dir": ".",
}

FEATURE_COLS = [
    "close_ratio",
    "open_ratio",
    "high_ratio",
    "low_ratio",
    "log_ret",
    "log_ret_5",
    "log_ret_20",
    "Volume_scaled",
    "RSI",
    "MACD_norm",
    "Signal_norm",
    "BB_Width",
    "BB_Position",
    "SMA_10_ratio",
    "SMA_50_ratio",
    "EMA_20_ratio",
    "Volatility_norm",
    "ATR_norm",
    "Volume_change",
]
N_FEAT = len(FEATURE_COLS)


def load_data(ticker, start, end, use_real=True):
    if use_real:
        try:
            import yfinance as yf
            df = yf.download(ticker, start=start, end=end,progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError("Empty data.")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            df.dropna(inplace=True)
            return df, True
        except Exception:
            pass

    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    prices = 150.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, n)))
    df = pd.DataFrame({
        "Open": prices * np.random.uniform(0.995, 1.005, n),
        "High": prices * np.random.uniform(1.000, 1.020, n),
        "Low": prices * np.random.uniform(0.980, 1.000, n),
        "Close": prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, n).astype(float),
    }, index=dates)
    return df, False


def add_technical_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    sma20 = close.rolling(20).mean()
    df["SMA20"] = sma20

    df["close_ratio"] = close / sma20
    df["open_ratio"] = df["Open"] / sma20
    df["high_ratio"] = high / sma20
    df["low_ratio"] = low / sma20

    df["log_ret"] = np.log(close / close.shift(1))
    df["log_ret_5"] = np.log(close / close.shift(5))
    df["log_ret_20"] = np.log(close / close.shift(20))

    sma10 = close.rolling(10).mean()
    sma50 = close.rolling(50).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    df["SMA_10"] = sma10
    df["SMA_50"] = sma50
    df["EMA_20"] = ema20
    df["SMA_10_ratio"] = sma10 / sma20 - 1
    df["SMA_50_ratio"] = sma50 / sma20 - 1
    df["EMA_20_ratio"] = ema20 / sma20 - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = (100 - 100 / (1 + gain / (loss + 1e-9))) / 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["MACD"] = macd
    df["Signal"] = sig
    df["MACD_norm"] = macd / (sma20 + 1e-9)
    df["Signal_norm"] = sig / (sma20 + 1e-9)

    std20 = close.rolling(20).std() + 1e-9
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Width"] = (upper - lower) / (sma20 + 1e-9)
    df["BB_Position"] = (close - lower) / (upper - lower + 1e-9)

    df["Volatility_norm"] = close.rolling(10).std() / (sma20 + 1e-9)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_norm"] = df["ATR"] / (sma20 + 1e-9)

    df["Momentum"] = close - close.shift(5)
    df["Volume_change"] = volume.pct_change()

    df.dropna(inplace=True)
    return df


def preprocess(df, sequence_len, test_split):
    from sklearn.preprocessing import RobustScaler, MinMaxScaler

    n_rows = len(df)
    split_r = int(n_rows * (1 - test_split))

    df_tr = df.iloc[:split_r]
    df_te = df.iloc[split_r:]

    stat_cols = [c for c in FEATURE_COLS if c != "Volume_scaled"]

    stat_scaler = RobustScaler()
    stat_scaler.fit(df_tr[stat_cols].values)

    vol_scaler = MinMaxScaler(feature_range=(0, 1))
    vol_scaler.fit(df_tr["Volume"].values.reshape(-1, 1))

    def _scale(src):
        out = pd.DataFrame(index=src.index)
        sc = stat_scaler.transform(src[stat_cols].values)
        sc = np.clip(sc, -4, 4)
        for j, col in enumerate(stat_cols):
            out[col] = sc[:, j]
        out["Volume_scaled"] = vol_scaler.transform(
            src["Volume"].values.reshape(-1, 1)).flatten()
        out["Close"] = src["Close"].values
        out["SMA20"] = src["SMA20"].values
        return out

    df_scaled = pd.concat([_scale(df_tr), _scale(df_te)])

    feat_arr = df_scaled[FEATURE_COLS].values
    sma20_arr = df_scaled["SMA20"].values
    close_arr = df_scaled["Close"].values

    X, y, sma20_y = [], [], []
    for i in range(sequence_len, len(feat_arr)):
        X.append(feat_arr[i - sequence_len: i])
        y.append(feat_arr[i, 0])
        sma20_y.append(sma20_arr[i])

    X = np.array(X)
    y = np.array(y)
    sma20_y = np.array(sma20_y)

    seq_split = split_r - sequence_len

    return (
        X[:seq_split], X[seq_split:],
        y[:seq_split], y[seq_split:],
        sma20_y[:seq_split], sma20_y[seq_split:],
        stat_scaler, vol_scaler,
        split_r,
        df_scaled,
        close_arr, sma20_arr,
    )


def _inv(scaled_ratio, sma20, stat_scaler):
    stat_cols_n = len([c for c in FEATURE_COLS if c != "Volume_scaled"])
    dummy = np.zeros((len(scaled_ratio), stat_cols_n))
    dummy[:, 0] = scaled_ratio
    ratios = stat_scaler.inverse_transform(dummy)[:, 0]
    return ratios * sma20
# ─────────────────────────────────────────────────────────────────────────────
def build_model(sequence_len, lstm_units, dropout_rate):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, Bidirectional, LSTM,
        BatchNormalization, Dropout, Dense,
        MultiHeadAttention, LayerNormalization,
        GlobalAveragePooling1D, Add,
    )
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(sequence_len, N_FEAT))

    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu')(inp)
    x = Conv1D(64, kernel_size=5, padding='causal', activation='relu')(x)

    x = Bidirectional(LSTM(lstm_units[0], return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    x = Bidirectional(LSTM(lstm_units[1], return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    attn = MultiHeadAttention(num_heads=4, key_dim=lstm_units[1])(x, x)
    x    = LayerNormalization()(Add()([attn, x]))
    x    = GlobalAveragePooling1D()(x)

    x   = Dense(128, activation='relu')(x)
    x   = Dropout(dropout_rate)(x)
    x   = Dense(64,  activation='relu')(x)
    out = Dense(1)(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
        loss='huber',
        metrics=['mae'],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train, epochs, batch_size, ckpt_path):
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    return model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        verbose=1,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, verbose=1),
            ModelCheckpoint(ckpt_path, save_best_only=True, verbose=0),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, sma20_test, stat_scaler):
    """
    WALK-FORWARD (teacher-forced) evaluation.
    Each prediction uses the TRUE prior 60-day window — no compounding.
    This gives honest 1-step-ahead metrics.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_pred_sc = model.predict(X_test, verbose=0).flatten()

    y_real = _inv(y_test,     sma20_test, stat_scaler)
    y_pred = _inv(y_pred_sc,  sma20_test, stat_scaler)

    # Directional accuracy
    real_dir = np.sign(np.diff(y_real))
    pred_dir = np.sign(np.diff(y_pred))
    dir_acc  = float(np.mean(real_dir == pred_dir)) * 100

    metrics = {
        "RMSE"    : float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAE"     : float(mean_absolute_error(y_real, y_pred)),
        "R2"      : float(r2_score(y_real, y_pred)),
        "MAPE"    : float(np.mean(np.abs((y_real - y_pred) /
                                         (np.abs(y_real) + 1e-9))) * 100),
        "DIR_ACC" : dir_acc,
    }
    print(f"[INFO] Evaluation Metrics (1-step walk-forward):")
    print(f"  {'RMSE':>12}: ${metrics['RMSE']:.2f}")
    print(f"  {'MAE':>12}: ${metrics['MAE']:.2f}")
    print(f"  {'R²':>12}: {metrics['R2']:.4f}")
    print(f"  {'MAPE':>12}: {metrics['MAPE']:.2f}%")
    print(f"  {'Direction':>12}: {metrics['DIR_ACC']:.1f}%  (random=50%)")
    return y_real, y_pred, metrics


# ─────────────────────────────────────────────────────────────────────────────
def forecast_future(model, df_scaled, stat_scaler, sequence_len, future_days):
    feat_arr   = df_scaled[FEATURE_COLS].values.copy()
    sma20_arr  = df_scaled["SMA20"].values.copy()
    close_arr  = df_scaled["Close"].values.copy()

    seq        = feat_arr[-sequence_len:].copy()
    price_buf  = list(close_arr[-20:])
    last_sma20 = sma20_arr[-1]

    stat_cols_n = len([c for c in FEATURE_COLS if c != "Volume_scaled"])
    preds = []

    for _ in range(future_days):
        inp = seq[-sequence_len:].reshape(1, sequence_len, N_FEAT)
        pred_sc = float(model.predict(inp, verbose=0)[0, 0])
        dummy = np.zeros((1, stat_cols_n))
        dummy[0, 0] = pred_sc
        pred_ratio  = stat_scaler.inverse_transform(dummy)[0, 0]
        pred_price  = pred_ratio * last_sma20
        preds.append(pred_price)
        price_buf.append(pred_price)
        new_sma20 = np.mean(price_buf[-20:])
        new_row    = seq[-1].copy()
        new_row[0] = pred_sc
        last_price = price_buf[-2]
        raw_log_r  = np.log(pred_price / (last_price + 1e-9))
        # scale log_ret using stat_scaler col index for log_ret (col 4)
        dummy2 = np.zeros((1, stat_cols_n))
        dummy2[0, 4] = raw_log_r
        new_row[4] = stat_scaler.transform(dummy2)[0, 4]

        seq        = np.vstack([seq, new_row])
        last_sma20 = new_sma20

    return np.array(preds)


# ─────────────────────────────────────────────────────────────────────────────
def plot_results(df, history, y_real, y_pred, future_prices, split_idx, cfg):
    ticker      = cfg["ticker"]
    future_days = cfg["future_days"]
    output_dir  = cfg["output_dir"]
    m           = cfg.get("_metrics", {})

    C = dict(bg="#0d1117", panel="#161b22", grid="#30363d", text="#e6edf3",
             blue="#58a6ff", green="#3fb950", red="#f85149", yellow="#d29922",
             purple="#bc8cff")

    fig = plt.figure(figsize=(22, 17), facecolor=C["bg"])
    fig.suptitle(
        f"StockSense AI — {ticker}  |  BiLSTM+Attention  |  1-Step Walk-Forward",
        fontsize=16, color="white", fontweight="bold", y=0.98)
    gs = GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.3)

    def style(ax, title):
        ax.set_facecolor(C["panel"])
        ax.tick_params(colors=C["text"], labelsize=9)
        for s in ax.spines.values(): s.set_edgecolor(C["grid"])
        ax.grid(color=C["grid"], linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_title(title, color=C["text"], fontsize=11, pad=8)

    # ── Panel 1: price chart ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    style(ax1, "Actual vs Predicted Close Price  (1-step walk-forward on test)")
    dates = df.index
    ax1.plot(dates[:split_idx], df["Close"].values[:split_idx],
             color=C["blue"], lw=1.0, alpha=0.7, label="Train (Actual)")
    test_dates = dates[split_idx: split_idx + len(y_real)]
    ax1.plot(test_dates, y_real, color=C["green"],  lw=1.6, label="Test (Actual)")
    ax1.plot(test_dates, y_pred, color=C["red"],    lw=1.6,
             linestyle="--", label="Test (Predicted, 1-step)")
    fd = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_days)
    ax1.plot(fd, future_prices, color=C["yellow"], lw=2,
             linestyle=":", marker="o", markersize=3,
             label=f"{future_days}-Day Forecast (speculative)")
    ax1.axvline(dates[split_idx], color=C["grid"], lw=1.4, linestyle="--", alpha=0.9)
    ax1.legend(facecolor=C["panel"], edgecolor=C["grid"],
               labelcolor=C["text"], fontsize=9)
    ax1.set_ylabel("Price (USD)", color=C["text"])
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ── Panel 2: loss ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style(ax2, "Training vs Validation Loss")
    ep = range(1, len(history.history["loss"]) + 1)
    ax2.plot(ep, history.history["loss"],     color=C["blue"], lw=1.5, label="Train")
    ax2.plot(ep, history.history["val_loss"], color=C["red"],  lw=1.5, label="Val")
    ax2.legend(facecolor=C["panel"], edgecolor=C["grid"],
               labelcolor=C["text"], fontsize=9)
    ax2.set_xlabel("Epoch", color=C["text"])
    ax2.set_ylabel("Huber Loss", color=C["text"])

    # ── Panel 3: scatter ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style(ax3, "Actual vs Predicted (Scatter)")
    ax3.scatter(y_real, y_pred, color=C["blue"], alpha=0.35, s=10)
    lo = min(y_real.min(), y_pred.min())
    hi = max(y_real.max(), y_pred.max())
    ax3.plot([lo, hi], [lo, hi], color=C["green"], lw=1.5,
             linestyle="--", label="Perfect fit")
    ax3.set_xlabel("Actual Price (USD)", color=C["text"])
    ax3.set_ylabel("Predicted Price (USD)", color=C["text"])
    ax3.legend(facecolor=C["panel"], edgecolor=C["grid"],
               labelcolor=C["text"], fontsize=9)

    # ── Panel 4: residuals ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    style(ax4, "Prediction Residuals (1-step)")
    res = y_real - y_pred
    ax4.bar(range(len(res)), res,
            color=[C["green"] if r >= 0 else C["red"] for r in res],
            width=1, alpha=0.8)
    ax4.axhline(0, color=C["text"], lw=0.8, linestyle="--")
    ax4.set_xlabel("Test Sample Index", color=C["text"])
    ax4.set_ylabel("Residual (USD)", color=C["text"])

    # ── Panel 5: metrics card ─────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor(C["panel"]); ax5.axis("off")
    ax5.text(0.5, 0.96, "Model Performance Metrics",
             color=C["text"], ha="center", fontsize=12,
             fontweight="bold", transform=ax5.transAxes)
    rows = [
        ("RMSE",     f"${m.get('RMSE',0):.2f}",         C["red"]),
        ("MAE",      f"${m.get('MAE', 0):.2f}",          C["yellow"]),
        ("R²",       f"{m.get('R2',  0):.4f}",           C["green"]),
        ("MAPE",     f"{m.get('MAPE',0):.2f}%",          C["blue"]),
        ("Dir.Acc.", f"{m.get('DIR_ACC',0):.1f}%",       C["purple"]),
    ]
    for i, (lbl, val, col) in enumerate(rows):
        y0 = 0.78 - i * 0.15
        ax5.text(0.08, y0, lbl, color=C["text"], fontsize=10,
                 transform=ax5.transAxes, va="center")
        ax5.text(0.52, y0, val, color=col, fontsize=14,
                 fontweight="bold", transform=ax5.transAxes, va="center")
    ax5.text(0.5, 0.02,
             "1-step walk-forward • dir.acc. > 52% = tradeable edge",
             color="#6e7f8d", ha="center", fontsize=8,
             transform=ax5.transAxes)

    out = os.path.join(output_dir, "stock_prediction_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C["bg"])
    print(f"[INFO] Chart saved → {out}")
    plt.close(fig)
    return out

def run_pipeline(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    print("=" * 65)
    print(f"  StockSense AI  |  {cfg['ticker']}  |  BiLSTM Pipeline")
    print("=" * 65)

    for f in ["best_model.keras",
              os.path.join(cfg["output_dir"], "best_model.keras")]:
        if os.path.exists(f):
            os.remove(f)

    df, is_real = load_data(
        cfg["ticker"], cfg["start_date"], cfg["end_date"], cfg["use_real"])
    df = add_technical_indicators(df)
    print(f"[INFO] {N_FEAT} features, {len(df)} trading days")

    (X_train, X_test,
     y_train, y_test,
     sma20_train, sma20_test,
     stat_scaler, vol_scaler,
     split_idx,
     df_scaled,
     close_arr, sma20_arr) = preprocess(df, cfg["sequence_len"], cfg["test_split"])

    print(f"[INFO] Train: {X_train.shape}  |  Test: {X_test.shape}")

    model = build_model(cfg["sequence_len"], cfg["lstm_units"], cfg["dropout_rate"])
    model.summary()

    print("\n[INFO] Training ...")
    ckpt    = os.path.join(cfg["output_dir"], "best_model.keras")
    history = train_model(model, X_train, y_train,
                          cfg["epochs"], cfg["batch_size"], ckpt)

    y_real, y_pred, metrics = evaluate_model(
        model, X_test, y_test, sma20_test, stat_scaler)
    cfg["_metrics"] = metrics

    future_prices = forecast_future(
        model, df_scaled, stat_scaler,
        cfg["sequence_len"], cfg["future_days"])

    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=cfg["future_days"])

    print(f"\n[INFO] Forecast (first 5): "
          f"{['$'+str(round(p,2)) for p in future_prices[:5]]} ...")

    chart_path = plot_results(
        df, history, y_real, y_pred, future_prices, split_idx, cfg)

    model_out = os.path.join(cfg["output_dir"], "lstm_stock_model.keras")
    model.save(model_out)
    print(f"[INFO] Model saved → {model_out}")
    print("\n Pipeline complete!")

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
        "scaler"       : stat_scaler,
        "is_real_data" : is_real,
    }


if __name__ == "__main__":
    run_pipeline()
