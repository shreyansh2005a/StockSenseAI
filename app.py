import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
import model as stock_model

st.set_page_config(
    page_title="StockSense AI", page_icon="📈",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:#080c10;color:#c9d1d9}
#MainMenu,footer{visibility:hidden}
.block-container{padding:1.5rem 2rem 2rem;max-width:1400px}
section[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e2936}
section[data-testid="stSidebar"] *{color:#c9d1d9 !important}
.hero-banner{background:linear-gradient(135deg,#0d1117 0%,#0f1e2e 60%,#091422 100%);border:1px solid #1e3a52;border-radius:12px;padding:1.8rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden}
.hero-banner::before{content:"";position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,#1a6fa822 0%,transparent 70%);border-radius:50%}
.hero-title{font-family:'Share Tech Mono',monospace;font-size:2.2rem;color:#58a6ff;letter-spacing:2px;margin:0;line-height:1}
.hero-sub{font-size:.88rem;color:#6e7f8d;margin-top:.35rem;letter-spacing:1px;text-transform:uppercase}
.hero-tags{display:flex;gap:6px;margin-top:.9rem;flex-wrap:wrap}
.hero-tag{display:inline-block;background:#0e2a3e;border:1px solid #1a6fa8;color:#58a6ff;font-family:'Share Tech Mono',monospace;font-size:.7rem;padding:3px 10px;border-radius:4px;letter-spacing:1px}
.metric-row{display:flex;gap:1rem;margin-bottom:1.5rem;flex-wrap:wrap}
.metric-card{flex:1;min-width:130px;background:#0d1117;border:1px solid #1e2936;border-radius:10px;padding:1rem 1.3rem;position:relative;overflow:hidden}
.metric-card::after{content:"";position:absolute;bottom:0;left:0;right:0;height:2px}
.mc-blue::after{background:#58a6ff}.mc-green::after{background:#3fb950}.mc-red::after{background:#f85149}.mc-yellow::after{background:#d29922}.mc-purple::after{background:#bc8cff}
.mc-label{font-size:.68rem;color:#6e7f8d;text-transform:uppercase;letter-spacing:1.5px;font-weight:500}
.mc-value{font-family:'Share Tech Mono',monospace;font-size:1.75rem;font-weight:700;margin-top:.2rem;line-height:1}
.mc-blue .mc-value{color:#58a6ff}.mc-green .mc-value{color:#3fb950}.mc-red .mc-value{color:#f85149}.mc-yellow .mc-value{color:#d29922}.mc-purple .mc-value{color:#bc8cff}
.mc-hint{font-size:.72rem;color:#6e7f8d;margin-top:.3rem}
.section-label{font-family:'Share Tech Mono',monospace;font-size:.68rem;color:#3fb950;letter-spacing:3px;text-transform:uppercase;margin-bottom:.5rem;margin-top:1.3rem}
div[data-testid="stButton"]>button{background:linear-gradient(135deg,#1a4f72,#0e3352);color:#58a6ff;border:1px solid #2d6fa8;border-radius:8px;font-family:'Share Tech Mono',monospace;letter-spacing:2px;font-size:.85rem;padding:.65rem 2rem;width:100%;transition:all .2s}
div[data-testid="stButton"]>button:hover{background:linear-gradient(135deg,#205f8a,#123f66);border-color:#58a6ff;transform:translateY(-1px);box-shadow:0 4px 20px #58a6ff22}
div[data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid #1e2936;gap:0}
button[data-baseweb="tab"]{font-family:'Share Tech Mono',monospace;font-size:.76rem;letter-spacing:1px;color:#6e7f8d !important;background:transparent !important;border:none !important;padding:.6rem 1.4rem !important}
button[data-baseweb="tab"][aria-selected="true"]{color:#58a6ff !important;border-bottom:2px solid #58a6ff !important}
div[data-baseweb="select"]>div{background:#0d1117 !important;border-color:#1e2936 !important;color:#c9d1d9 !important}
div[data-testid="stExpander"]{background:#0d1117;border:1px solid #1e2936;border-radius:8px}
div[data-testid="stInfo"]{background:#0e2a3e;border-left-color:#58a6ff}
div[data-testid="stSuccess"]{background:#0e2e1a;border-left-color:#3fb950}
div[data-testid="stWarning"]{background:#2a1e0e;border-left-color:#d29922}
hr{border-color:#1e2936}
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#080c10}
::-webkit-scrollbar-thumb{background:#1e2936;border-radius:3px}
</style>
""", unsafe_allow_html=True)

C = dict(bg="#080c10", panel="#0d1117", grid="#1e2936", text="#c9d1d9",
         blue="#58a6ff", green="#3fb950", red="#f85149", yellow="#d29922")

def style_ax(ax, title=""):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"], labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor(C["grid"])
    ax.grid(color=C["grid"], linestyle="--", linewidth=0.5, alpha=0.6)
    if title: ax.set_title(title, color=C["text"], fontsize=10, pad=6)

def make_price_chart(df, result, ticker, future_days):
    fig, ax = plt.subplots(figsize=(14, 4.5), facecolor=C["bg"])
    style_ax(ax, f"{ticker} — Actual vs Predicted Close Price  (1-step walk-forward)")
    dates = result["dates"]; split = result["split_idx"]
    ax.plot(dates[:split], df["Close"].values[:split],
            color=C["blue"], lw=1.0, alpha=0.7, label="Train")
    td = dates[split: split + len(result["y_real"])]
    ax.plot(td, result["y_real"], color=C["green"], lw=1.5, label="Actual (Test)")
    ax.plot(td, result["y_pred"], color=C["red"],   lw=1.5, linestyle="--", label="Predicted (1-step)")
    fd = pd.bdate_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_days)
    ax.plot(fd, result["future"], color=C["yellow"], lw=2,
            linestyle=":", marker="o", markersize=2.5, label=f"{future_days}-Day Forecast (speculative)")
    ax.axvline(dates[split], color=C["grid"], lw=1.2, linestyle="--")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], fontsize=8)
    ax.set_ylabel("Price (USD)", color=C["text"], fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout(pad=0.8)
    return fig

def make_loss_chart(tl, vl):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=C["bg"])
    style_ax(ax, "Training vs Validation Loss")
    ep = range(1, len(tl) + 1)
    ax.plot(ep, tl, color=C["blue"], lw=1.5, label="Train")
    ax.plot(ep, vl, color=C["red"],  lw=1.5, label="Val")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], fontsize=8)
    ax.set_xlabel("Epoch", color=C["text"], fontsize=9)
    ax.set_ylabel("Loss",  color=C["text"], fontsize=9)
    fig.tight_layout(pad=0.8)
    return fig

def make_scatter_chart(yr, yp):
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=C["bg"])
    style_ax(ax, "Actual vs Predicted (Scatter)")
    ax.scatter(yr, yp, color=C["blue"], alpha=0.3, s=8)
    lo, hi = min(yr.min(), yp.min()), max(yr.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], color=C["green"], lw=1.5, linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual",    color=C["text"], fontsize=9)
    ax.set_ylabel("Predicted", color=C["text"], fontsize=9)
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], fontsize=8)
    fig.tight_layout(pad=0.8)
    return fig

def make_residuals_chart(yr, yp):
    fig, ax = plt.subplots(figsize=(14, 3.2), facecolor=C["bg"])
    style_ax(ax, "Prediction Residuals")
    res = yr - yp
    ax.bar(range(len(res)), res,
           color=[C["green"] if r >= 0 else C["red"] for r in res], width=1, alpha=0.8)
    ax.axhline(0, color=C["text"], lw=0.8, linestyle="--")
    ax.set_xlabel("Sample",         color=C["text"], fontsize=9)
    ax.set_ylabel("Residual (USD)", color=C["text"], fontsize=9)
    fig.tight_layout(pad=0.8)
    return fig

def make_indicator_chart(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 7), facecolor=C["bg"])
    fig.subplots_adjust(hspace=0.45)

    ax = axes[0]; style_ax(ax, "Price + Bollinger Bands + SMA50")
    ax.plot(df.index, df["Close"],    color=C["blue"],   lw=1.2, label="Close")
    ax.plot(df.index, df["BB_Upper"], color=C["yellow"], lw=0.8, linestyle="--", label="BB Upper")
    ax.plot(df.index, df["BB_Lower"], color=C["yellow"], lw=0.8, linestyle="--", label="BB Lower")
    ax.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], color=C["yellow"], alpha=0.05)
    ax.plot(df.index, df["SMA_50"],   color=C["green"],  lw=0.9, linestyle=":", label="SMA 50")
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], fontsize=7, ncol=4)
    ax.set_ylabel("USD", color=C["text"], fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]; style_ax(ax, "RSI (14)")
    rsi_pct = df["RSI"] * 100
    ax.plot(df.index, rsi_pct, color=C["blue"], lw=1.2)
    ax.axhline(70, color=C["red"],   lw=0.8, linestyle="--", alpha=0.8, label="Overbought 70")
    ax.axhline(30, color=C["green"], lw=0.8, linestyle="--", alpha=0.8, label="Oversold 30")
    ax.fill_between(df.index, 70, rsi_pct, where=(rsi_pct > 70), color=C["red"],   alpha=0.1)
    ax.fill_between(df.index, 30, rsi_pct, where=(rsi_pct < 30), color=C["green"], alpha=0.1)
    ax.set_ylim(0, 100)
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], fontsize=7)
    ax.set_ylabel("RSI", color=C["text"], fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[2]; style_ax(ax, "MACD")
    hist = df["MACD"] - df["Signal"]
    ax.bar(df.index, hist, color=[C["green"] if v >= 0 else C["red"] for v in hist],
           width=1, alpha=0.6)
    ax.plot(df.index, df["MACD"],   color=C["blue"],   lw=1.0, label="MACD")
    ax.plot(df.index, df["Signal"], color=C["yellow"], lw=1.0, label="Signal")
    ax.axhline(0, color=C["grid"], lw=0.8)
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], fontsize=7)
    ax.set_ylabel("MACD", color=C["text"], fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.patch.set_facecolor(C["bg"])
    return fig


with st.sidebar:
    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;color:#58a6ff;
        font-size:1.1rem;letter-spacing:2px;padding:.8rem 0 1.2rem;">◈ STOCKSENSE AI</div>""",
        unsafe_allow_html=True)

    st.markdown('<div class="section-label">// DATA SOURCE</div>', unsafe_allow_html=True)
    ticker   = st.text_input("Ticker Symbol", value="AAPL").upper().strip()
    c1, c2   = st.columns(2)
    with c1: start_date = st.date_input("From", value=pd.Timestamp("2019-01-01"))
    with c2: end_date   = st.date_input("To",   value=pd.Timestamp("2026-03-01"))
    use_real = st.checkbox("Use Yahoo Finance (requires yfinance)", value=True)

    st.markdown('<div class="section-label">// MODEL CONFIG</div>', unsafe_allow_html=True)
    seq_len     = st.slider("Look-back Window (days)",  30, 120,  60)
    future_days = st.slider("Forecast Horizon (days)",   7,  90,  30)
    test_split  = st.slider("Test Split %",             10,  40,  20) / 100

    with st.expander("Advanced Hyperparameters"):
        epochs     = st.number_input("Epochs",       10, 300, 100, step=10)
        batch_size = st.number_input("Batch Size",    8, 128,  32, step=8)
        lstm_l1    = st.number_input("LSTM Layer 1", 64, 512, 128, step=64)
        lstm_l2    = st.number_input("LSTM Layer 2", 32, 256,  64, step=32)
        dropout    = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("⟶  RUN PREDICTION")

    st.markdown("""<div style="margin-top:1.5rem;font-size:.67rem;color:#3a4a57;
        font-family:'Share Tech Mono',monospace;line-height:2;">
        MODEL : BiLSTM + Attention<br>LOSS  : Huber<br>
        OPT   : Adam + LR Decay<br>TARGET: Close / SMA20 ratio<br>
        INVERT: ratio × SMA20 (exact)<br>FEATS : 19
    </div>""", unsafe_allow_html=True)


st.markdown(f"""
<div class="hero-banner">
  <div class="hero-title">STOCKSENSE AI</div>
  <div class="hero-sub">1-Step Walk-Forward Prediction &nbsp;·&nbsp; BiLSTM + Multi-Head Attention</div>
  <div class="hero-tags">
    <span class="hero-tag">TICKER: {ticker}</span>
    <span class="hero-tag">{start_date} → {end_date}</span>
    <span class="hero-tag">LOOK-BACK: {seq_len}d</span>
    <span class="hero-tag">FORECAST: {future_days}d</span>
    <span class="hero-tag">LSTM [{int(lstm_l1)}·{int(lstm_l2)}]</span>
  </div>
</div>
""", unsafe_allow_html=True)


if "result" not in st.session_state:
    st.markdown('<div class="section-label">// HOW TO USE</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, num, title, body in [
        (c1, "01", "Configure",  "Set ticker, date range and model params in the sidebar."),
        (c2, "02", "Run Model",  "Click RUN PREDICTION. Uses window-normalised price (Close/SMA20) for stable, drift-free predictions."),
        (c3, "03", "Analyze",    "Explore Price, Indicators, and Diagnostics tabs. Check Direction Accuracy — >52% is a real edge."),
    ]:
        with col:
            st.markdown(f"""<div style="background:#0d1117;border:1px solid #1e2936;
                border-radius:10px;padding:1.4rem;min-height:120px;">
              <div style="font-family:'Share Tech Mono',monospace;color:#58a6ff;font-size:1.4rem;">{num}</div>
              <div style="color:#c9d1d9;font-weight:600;margin:.3rem 0 .5rem;">{title}</div>
              <div style="color:#6e7f8d;font-size:.82rem;line-height:1.5;">{body}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # st.info("💡 **Architecture:** Target = **Close / SMA20** (always ≈0.85–1.15, works at any price level). "
    #         "Evaluation is **1-step walk-forward** — each prediction uses the true prior 60-day window, "
    #         "not compounded guesses. This gives honest R² and MAPE metrics.")


if run_btn:
    ui_config = {
        "ticker"      : ticker,
        "start_date"  : str(start_date),
        "end_date"    : str(end_date),
        "sequence_len": int(seq_len),
        "test_split"  : float(test_split),
        "epochs"      : int(epochs),
        "batch_size"  : int(batch_size),
        "lstm_units"  : [int(lstm_l1), int(lstm_l2)],
        "dropout_rate": float(dropout),
        "future_days" : int(future_days),
        "use_real"    : use_real,
        "output_dir"  : os.path.dirname(os.path.abspath(__file__)),
    }

    with st.spinner(""):
        progress = st.progress(0)
        status   = st.empty()
        try:
            status.text("📥  Loading data …")
            progress.progress(5)

            _orig_train = stock_model.train_model
            def _patched_train(model, X_train, y_train, epochs, batch_size, ckpt_path):
                from tensorflow.keras.callbacks import (
                    EarlyStopping, ReduceLROnPlateau,
                    ModelCheckpoint, LambdaCallback)
                status.text("🧠  Training BiLSTM + Attention …")
                def _cb(epoch, logs=None):
                    progress.progress(min(20 + int(60 * (epoch + 1) / epochs), 79))
                return model.fit(
                    X_train, y_train,
                    epochs=epochs, batch_size=batch_size,
                    validation_split=0.1, shuffle=False, verbose=1,
                    callbacks=[
                        EarlyStopping(patience=15, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7),
                        ModelCheckpoint(ckpt_path, save_best_only=True, verbose=0),
                        LambdaCallback(on_epoch_end=_cb),
                    ])
            stock_model.train_model = _patched_train

            raw = stock_model.run_pipeline(ui_config)
            stock_model.train_model = _orig_train

            progress.progress(95)
            status.text("📊  Rendering charts …")

            st.session_state["result"] = {
                "metrics"   : raw["metrics"],
                "future"    : raw["future_prices"],
                "train_loss": raw["history"].history["loss"],
                "val_loss"  : raw["history"].history["val_loss"],
                "y_real"    : raw["y_real"],
                "y_pred"    : raw["y_pred"],
                "split_idx" : raw["split_idx"],
                "dates"     : raw["df"].index,
                "is_real"   : raw["is_real_data"],
            }
            st.session_state["df"]          = raw["df"]
            st.session_state["ticker"]      = ticker
            st.session_state["future_days"] = future_days
            st.session_state["seq_len"]     = seq_len
            st.session_state["lstm_l1"]     = int(lstm_l1)
            st.session_state["lstm_l2"]     = int(lstm_l2)
            st.session_state["dropout"]     = dropout

            progress.progress(100)
            status.text("✅  Done!")

            if raw["is_real_data"]:
                st.success(f"✅ Real data loaded — {len(raw['df'])} trading days.")
            else:
                st.warning("⚠️ Using simulated data.")

        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.exception(e)
        finally:
            progress.empty(); status.empty()

if "result" in st.session_state:
    result = st.session_state["result"]
    df     = st.session_state["df"]
    m      = result["metrics"]
    t_s    = st.session_state.get("ticker",      ticker)
    fd_s   = st.session_state.get("future_days", future_days)
    sl_s   = st.session_state.get("seq_len",     seq_len)
    l1_s   = st.session_state.get("lstm_l1",     int(lstm_l1))
    l2_s   = st.session_state.get("lstm_l2",     int(lstm_l2))
    dr_s   = st.session_state.get("dropout",     dropout)

    # Direction accuracy colour
    dir_acc   = m.get("DIR_ACC", 50.0)
    dir_color = "#3fb950" if dir_acc >= 52 else "#d29922" if dir_acc >= 50 else "#f85149"
    dir_class = "mc-green" if dir_acc >= 52 else "mc-yellow" if dir_acc >= 50 else "mc-red"

    st.markdown('<div class="section-label">// PERFORMANCE METRICS</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card mc-red">
        <div class="mc-label">RMSE</div><div class="mc-value">${m['RMSE']:.2f}</div>
        <div class="mc-hint">Root Mean Squared Error</div>
      </div>
      <div class="metric-card mc-yellow">
        <div class="mc-label">MAE</div><div class="mc-value">${m['MAE']:.2f}</div>
        <div class="mc-hint">Mean Absolute Error</div>
      </div>
      <div class="metric-card mc-green">
        <div class="mc-label">R² Score</div><div class="mc-value">{m['R2']:.4f}</div>
        <div class="mc-hint">1.0 = perfect fit</div>
      </div>
      <div class="metric-card mc-blue">
        <div class="mc-label">MAPE</div><div class="mc-value">{m['MAPE']:.2f}%</div>
        <div class="mc-hint">Mean Abs. % Error</div>
      </div>
      <div class="metric-card {dir_class}">
        <div class="mc-label">Direction Acc.</div>
        <div class="mc-value">{dir_acc:.1f}%</div>
        <div class="mc-hint">Random baseline = 50%</div>
      </div>
      <div class="metric-card mc-blue">
        <div class="mc-label">Data Points</div><div class="mc-value">{len(df)}</div>
        <div class="mc-hint">Trading days loaded</div>
      </div>
      <div class="metric-card mc-green">
        <div class="mc-label">Forecast</div><div class="mc-value">{fd_s}d</div>
        <div class="mc-hint">Days ahead predicted</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "  PRICE PREDICTION  ",
        "  TECHNICAL INDICATORS  ",
        "  MODEL DIAGNOSTICS  ",
    ])

    with tab1:
        st.markdown('<div class="section-label">// PRICE CHART</div>',
                    unsafe_allow_html=True)
        st.pyplot(make_price_chart(df, result, t_s, fd_s))
        st.markdown('<div class="section-label">// FORECAST TABLE</div>',
                    unsafe_allow_html=True)
        fd = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=fd_s)
        st.dataframe(pd.DataFrame({
            "Date"           : fd.strftime("%Y-%m-%d"),
            "Predicted Price": [f"${p:.2f}" for p in result["future"]],
            "Day"            : [f"+{i+1}" for i in range(fd_s)],
        }), width="stretch", hide_index=True)

    with tab2:
        st.markdown('<div class="section-label">// TECHNICAL ANALYSIS</div>',
                    unsafe_allow_html=True)
        st.pyplot(make_indicator_chart(df))
        st.markdown('<div class="section-label">// RAW DATA (last 60 rows)</div>',
                    unsafe_allow_html=True)
        # Only include columns that exist in the new model's df
        disp_candidates = ["Close", "Open", "High", "Low", "Volume",
                           "RSI", "MACD", "BB_Upper", "BB_Lower",
                           "Momentum", "Volatility_norm", "ATR"]
        disp = [c for c in disp_candidates if c in df.columns]
        st.dataframe(df[disp].tail(60).round(4), width="stretch", height=260)

    with tab3:
        st.markdown('<div class="section-label">// MODEL DIAGNOSTICS</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.pyplot(make_loss_chart(result["train_loss"], result["val_loss"]))
        with c2: st.pyplot(make_scatter_chart(result["y_real"],  result["y_pred"]))
        st.markdown('<div class="section-label" style="margin-top:1rem;">// RESIDUALS</div>',
                    unsafe_allow_html=True)
        st.pyplot(make_residuals_chart(result["y_real"], result["y_pred"]))

        with st.expander("Model Architecture"):
            st.code(f"""
Input  →  (batch, {sl_s}, 19 features)
────────────────────────────────────────────────────────
Conv1D (64, k=3, causal, relu)   local pattern extractor
Conv1D (64, k=5, causal, relu)

Bidirectional LSTM ({l1_s})        return_sequences=True
BatchNormalization
Dropout({dr_s})

Bidirectional LSTM ({l2_s})        return_sequences=True
BatchNormalization
Dropout({dr_s})

MultiHeadAttention (heads=4)     attend key timesteps
LayerNorm + Residual
GlobalAveragePooling1D

Dense(128, relu) → Dropout({dr_s}) → Dense(64, relu) → Dense(1)
                                 ↑ Predicted scaled close_ratio
────────────────────────────────────────────────────────
TARGET   : close_ratio = Close / SMA20  (≈ 0.85–1.15)
INVERSION: predicted_ratio × SMA20_at_that_step  (no drift)
EVAL     : 1-step walk-forward on test set (teacher-forced)
LOSS     : Huber  |  Optimizer: Adam(lr=5e-4, clip=1.0)
────────────────────────────────────────────────────────
Features (all stationary / ratio-based):
  [0]  close_ratio     ← TARGET (Close / SMA20)
  [1]  open_ratio      (Open  / SMA20)
  [2]  high_ratio      (High  / SMA20)
  [3]  low_ratio       (Low   / SMA20)
  [4]  log_ret         daily log return
  [5]  log_ret_5       5-day log return
  [6]  log_ret_20      20-day log return
  [7]  Volume_scaled   MinMax on train
  [8]  RSI             0–1
  [9]  MACD_norm       MACD / SMA20
  [10] Signal_norm     Signal / SMA20
  [11] BB_Width        (upper-lower) / SMA20
  [12] BB_Position     (close-lower) / (upper-lower)
  [13] SMA_10_ratio    SMA10/SMA20 − 1
  [14] SMA_50_ratio    SMA50/SMA20 − 1
  [15] EMA_20_ratio    EMA20/SMA20 − 1
  [16] Volatility_norm rolling10std / SMA20
  [17] ATR_norm        ATR14 / SMA20
  [18] Volume_change   pct change
            """, language="text")

    st.markdown("""<div style="margin-top:2.5rem;padding:1rem 0;border-top:1px solid #1e2936;
        text-align:center;font-family:'Share Tech Mono',monospace;
        font-size:.67rem;color:#3a4a57;letter-spacing:1.5px;">
        STOCKSENSE AI &nbsp;·&nbsp; EDUCATIONAL USE ONLY &nbsp;·&nbsp; NOT FINANCIAL ADVICE
    </div>""", unsafe_allow_html=True)
