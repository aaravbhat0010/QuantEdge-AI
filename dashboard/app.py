from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/quant_edge.duckdb"

st.set_page_config(
    page_title="QuantEdge AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    max-width: 1600px;
    padding-top: 2rem;
}
[data-testid="stMetric"] {
    background: #131D2E;
    border: 1px solid #253449;
    padding: 20px;
    border-radius: 12px;
}
h1, h2, h3 {
    letter-spacing: -0.5px;
}
</style>
""", unsafe_allow_html=True)

MODELS = {
    "LoRA V7": {
        "table": "chronos2_lora_v7_test",
        "path": "models/checkpoints/chronos2_merged_v7",
        "variables": 29,
        "rank": 32,
        "precision": "FP32",
        "train_memory": 1.94,
        "train_time": 116.68,
        "infer_memory": 0.31,
        "infer_time": 122.06
    },
    "LoRA V8": {
        "table": "chronos2_lora_v8_test",
        "path": "models/checkpoints/chronos2_merged_v8",
        "variables": 1,
        "rank": 8,
        "precision": "FP32",
        "train_memory": 1.78,
        "train_time": 114.60,
        "infer_memory": 0.26,
        "infer_time": 4.71
    },
    "QLoRA V9": {
        "table": "chronos2_qlora_v9_test",
        "path": "models/adapters/chronos2_qlora_v9/final",
        "variables": 1,
        "rank": 8,
        "precision": "NF4",
        "train_memory": 0.776,
        "train_time": 141.01,
        "infer_memory": 0.10,
        "infer_time": 9.40
    }
}

FEATURES = [
    "return_1d", "return_20d",
    "sma_20_ratio", "sma_50_ratio",
    "ema_12_ratio", "ema_26_ratio",
    "macd_ratio", "rsi_14",
    "volatility_20", "relative_volume",
    "news_count", "has_news",
    "sentiment_mean", "positive_prob_mean",
    "negative_prob_mean",
    "sentiment_confidence_mean",
    "net_profit_margin", "operating_margin",
    "liabilities_to_assets", "equity_to_assets",
    "revenue_age_days", "federal_funds_rate",
    "consumer_price_index", "unemployment_rate",
    "treasury_yield_spread", "volatility_index",
    "crude_oil_price", "consumer_sentiment"
]

@st.cache_data
def query(sql, params=()):
    with duckdb.connect(str(DB), read_only=True) as con:
        return con.execute(sql, params).df()

def chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#080E1A",
        plot_bgcolor="#131D2E",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

def metrics(df):
    actual = df["actual_return"].to_numpy()
    predicted = df["q50"].to_numpy()
    error = predicted - actual

    return {
        "MAE (pp)": np.mean(np.abs(error)) * 100,
        "RMSE (pp)": np.sqrt(np.mean(error**2)) * 100,
        "Direction (%)": np.mean(
            np.sign(actual) == np.sign(predicted)
        ) * 100,
        "Coverage (%)": np.mean(
            (actual >= df["q10"]) &
            (actual <= df["q90"])
        ) * 100
    }

@st.cache_resource(max_entries=1)
def load_model(name):
    import torch
    from chronos import Chronos2Pipeline

    if name == "QLoRA V9":
        from transformers import BitsAndBytesConfig
        from peft import PeftModel

        pipeline = Chronos2Pipeline.from_pretrained(
            str(ROOT / "models/pretrained/chronos-2"),
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            dtype=torch.bfloat16,
            local_files_only=True
        )

        pipeline.model = PeftModel.from_pretrained(
            pipeline.model,
            str(ROOT / MODELS[name]["path"]),
            is_trainable=False,
            local_files_only=True
        )
    else:
        pipeline = Chronos2Pipeline.from_pretrained(
            str(ROOT / MODELS[name]["path"]),
            device_map="cuda",
            dtype=torch.bfloat16,
            local_files_only=True
        )

    pipeline.model.eval()
    return pipeline

st.sidebar.title("QuantEdge AI")
st.sidebar.caption("NVIDIA JETSON THOR")

page = st.sidebar.radio(
    "Workspace",
    [
        "Overview",
        "Market Intelligence",
        "AI Forecasting",
        "Sentiment Intelligence",
        "Model Arena",
        "Engineering Lab"
    ]
)

tickers = query("""
    SELECT DISTINCT ticker
    FROM master_dataset
    ORDER BY ticker
""")["ticker"].tolist()

ticker = st.sidebar.selectbox(
    "Company",
    tickers,
    index=tickers.index("NVDA")
)

model_name = st.sidebar.selectbox(
    "AI Model",
    list(MODELS)
)

st.sidebar.divider()
st.sidebar.caption("Historical research platform")
st.sidebar.caption("Data: 2016–2023")
st.sidebar.caption("No live market feed")

st.title("QuantEdge AI")
st.caption(
    "FINANCIAL INTELLIGENCE  /  LOCAL AI  /  NVIDIA THOR"
)

stock = query("""
    SELECT *
    FROM master_dataset
    WHERE ticker = ?
    ORDER BY date
""", (ticker,))

stock["date"] = pd.to_datetime(stock["date"])

if page == "Overview":
    st.header("Research overview")

    cols = st.columns(4)

    total = query("""
        SELECT COUNT(*) AS n
        FROM master_dataset
    """).iloc[0]["n"]

    news_count = query("""
        SELECT COUNT(*) AS n
        FROM news_sentiment
    """).iloc[0]["n"]

    cols[0].metric("Companies", len(tickers))
    cols[1].metric("Stock-day records", f"{total:,}")
    cols[2].metric("Financial headlines", f"{news_count:,}")
    cols[3].metric("Financial features", len(stock.columns)-2)

    st.subheader(f"{ticker} — Historical adjusted price")

    chart(px.line(
        stock,
        x="date",
        y="adjusted_close"
    ))

    st.subheader("Final test performance")

    rows = []
    for name, config in MODELS.items():
        df = query(f"SELECT * FROM {config['table']}")
        rows.append({"Model": name, **metrics(df)})

    st.dataframe(
        pd.DataFrame(rows).round(4),
        hide_index=True,
        use_container_width=True
    )

    st.info(
        "Fine-tuning reduced forecasting error, "
        "but the AI models did not outperform "
        "the strongest simple baselines."
    )

elif page == "Market Intelligence":
    st.header(f"{ticker} — Market Intelligence")

    start, end = st.date_input(
        "Historical period",
        value=(
            stock["date"].min().date(),
            stock["date"].max().date()
        ),
        min_value=stock["date"].min().date(),
        max_value=stock["date"].max().date()
    )

    df = stock[
        (stock["date"].dt.date >= start) &
        (stock["date"].dt.date <= end)
    ]

    if df.empty:
        st.warning("No trading records in this period.")
        st.stop()

    latest = df.iloc[-1]

    cols = st.columns(4)
    cols[0].metric(
        "Adjusted price",
        f"${latest['adjusted_close']:,.2f}"
    )
    cols[1].metric(
        "5-day return",
        f"{latest['return_5d']*100:.2f}%"
    )
    cols[2].metric("RSI", f"{latest['rsi_14']:.2f}")
    cols[3].metric(
        "Annualized volatility",
        f"{latest['volatility_20']*100:.2f}%"
    )

    fig = go.Figure(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    ))

    fig.update_layout(
        title="Historical OHLC",
        xaxis_rangeslider_visible=False
    )
    chart(fig)

    st.subheader("Explore financial indicators")

    numeric = df.select_dtypes(
        include="number"
    ).columns.tolist()

    feature = st.selectbox(
        "Financial indicator",
        numeric,
        index=numeric.index("rsi_14")
    )

    chart(px.line(df, x="date", y=feature))

    with st.expander("Underlying financial data"):
        st.dataframe(df, use_container_width=True)

elif page == "AI Forecasting":
    st.header(f"{ticker} — AI Forecasting")

    config = MODELS[model_name]

    cols = st.columns(4)
    cols[0].metric("Model", model_name)
    cols[1].metric("Variables", config["variables"])
    cols[2].metric("Base precision", config["precision"])
    cols[3].metric("Horizon", "5 trading days")

    st.caption(
        "Select a historical date to generate "
        "a new forecast using only prior information."
    )

    dates = stock["date"].iloc[64:].tolist()

    selected = st.selectbox(
        "Forecast origin",
        dates,
        index=len(dates)-1,
        format_func=lambda x: x.strftime("%Y-%m-%d")
    )

    context = stock[
        stock["date"] <= selected
    ].tail(60)

    chart(px.line(
        context,
        x="date",
        y="return_5d",
        title="60-day historical input"
    ))

    if st.button(
        "Generate AI forecast",
        type="primary"
    ):
        import time
        import torch

        history = context["return_5d"].to_numpy(
            dtype=np.float32
        )

        assert len(history) == 60
        assert np.isfinite(history).all()

        if model_name == "LoRA V7":
            inputs = [{
                "target": history,
                "past_covariates": {
                    f: context[f].to_numpy(
                        dtype=np.float32
                    )
                    for f in FEATURES
                }
            }]
        else:
            inputs = torch.from_numpy(
                history
            ).reshape(1, 1, 60)

        with st.spinner("Running local GPU inference..."):
            pipeline = load_model(model_name)

            torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.inference_mode():
                quantiles, _ = pipeline.predict_quantiles(
                    inputs,
                    prediction_length=5,
                    quantile_levels=[0.1, 0.5, 0.9],
                    context_length=60,
                    cross_learning=False
                )

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            result = quantiles[0].float().cpu().numpy()
            q10, q50, q90 = result[0, -1] * 100

        cols = st.columns(4)
        cols[0].metric("Lower quantile", f"{q10:.2f}%")
        cols[1].metric("Median forecast", f"{q50:.2f}%")
        cols[2].metric("Upper quantile", f"{q90:.2f}%")
        cols[3].metric("Inference", f"{elapsed*1000:.1f} ms")

        st.success(
            f"Five-day median return forecast: {q50:.2f}%"
        )

        st.caption(
            "Quantiles express model uncertainty, "
            "not guaranteed future outcomes."
        )

elif page == "Sentiment Intelligence":
    st.header(f"{ticker} — Sentiment Intelligence")

    news = query("""
        SELECT *
        FROM news_sentiment
        WHERE ticker = ?
        ORDER BY published_at DESC
    """, (ticker,))

    if news.empty:
        st.warning("No news records available.")
        st.stop()

    counts = news["predicted_sentiment"].value_counts()

    cols = st.columns(4)
    cols[0].metric("Headlines", f"{len(news):,}")

    for col, label in zip(
        cols[1:], ["positive", "neutral", "negative"]
    ):
        col.metric(
            label.title(),
            f"{counts.get(label, 0)/len(news)*100:.1f}%"
        )

    chart(px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.65,
        title="FinBERT sentiment distribution"
    ))

    chart(px.line(
        stock,
        x="date",
        y="sentiment_mean",
        title="Daily historical sentiment"
    ))

    st.subheader("Financial news archive")

    sentiment = st.selectbox(
        "Filter sentiment",
        ["All", "positive", "neutral", "negative"]
    )

    if sentiment != "All":
        news = news[
            news["predicted_sentiment"] == sentiment
        ]

    st.dataframe(
        news[[
            "published_at", "headline",
            "predicted_sentiment",
            "confidence", "sentiment_score"
        ]],
        hide_index=True,
        use_container_width=True
    )

elif page == "Model Arena":
    st.header("Model Arena")
    st.caption("Held-out test period: 2022–2023")

    rows = []
    fig = go.Figure()

    for name, config in MODELS.items():
        df = query(
            f"""
            SELECT *
            FROM {config['table']}
            WHERE ticker = ?
            ORDER BY date
            """,
            (ticker,)
        )

        rows.append({"Model": name, **metrics(df)})

        fig.add_trace(go.Scatter(
            x=df["target_date"],
            y=df["q50"]*100,
            name=name
        ))

    fig.add_trace(go.Scatter(
        x=df["target_date"],
        y=df["actual_return"]*100,
        name="Actual return",
        line=dict(color="white", width=2)
    ))

    fig.update_layout(
        title=f"{ticker} — Forecast vs. actual return",
        yaxis_title="Five-day return (%)"
    )

    chart(fig)

    st.dataframe(
        pd.DataFrame(rows).round(4),
        hide_index=True,
        use_container_width=True
    )

    selected_df = query(
        f"""
        SELECT *
        FROM {MODELS[model_name]['table']}
        WHERE ticker = ?
        ORDER BY date
        """,
        (ticker,)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=selected_df["target_date"],
        y=selected_df["q90"]*100,
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=selected_df["target_date"],
        y=selected_df["q10"]*100,
        fill="tonexty",
        fillcolor="rgba(56,189,248,0.18)",
        line=dict(width=0),
        name="80% prediction interval"
    ))

    fig.add_trace(go.Scatter(
        x=selected_df["target_date"],
        y=selected_df["q50"]*100,
        name="Median forecast"
    ))

    fig.add_trace(go.Scatter(
        x=selected_df["target_date"],
        y=selected_df["actual_return"]*100,
        name="Actual return"
    ))

    fig.update_layout(
        title=f"{model_name} — Forecast uncertainty"
    )

    chart(fig)

    st.download_button(
        "Download predictions",
        selected_df.to_csv(index=False),
        file_name=f"{ticker}_{model_name}_predictions.csv",
        mime="text/csv"
    )

elif page == "Engineering Lab":
    st.header("AI Engineering Lab")

    cols = st.columns(3)
    cols[0].metric("Base parameters", "119.48M")
    cols[1].metric("QLoRA memory reduction", "56.4%")
    cols[2].metric("Final AI forecasts", "99,000")

    performance = pd.DataFrame([
        {
            "Model": name,
            "Variables": c["variables"],
            "Rank": c["rank"],
            "Base precision": c["precision"],
            "Training memory (GiB)": c["train_memory"],
            "Training time (s)": c["train_time"],
            "Inference memory (GiB)": c["infer_memory"],
            "Inference time (s)": c["infer_time"]
        }
        for name, c in MODELS.items()
    ])

    st.dataframe(
        performance,
        hide_index=True,
        use_container_width=True
    )

    chart(px.bar(
        performance,
        x="Model",
        y="Training memory (GiB)",
        color="Model",
        title="Peak allocated GPU memory — Training"
    ))

    chart(px.bar(
        performance,
        x="Model",
        y="Inference time (s)",
        color="Model",
        title="Inference time — 24,750 forecasts"
    ))

    st.subheader("Scientific benchmark")

    rows = []

    for name, table in [
        ("Original Chronos-2", "chronos2_zero_shot_test"),
        *[(n, c["table"]) for n, c in MODELS.items()]
    ]:
        df = query(f"SELECT * FROM {table}")
        rows.append({"Model": name, **metrics(df)})

    baseline = query("""
        SELECT *
        FROM chronos2_zero_shot_test
    """)

    for name, prediction in [
        ("Zero return", np.zeros(len(baseline))),
        ("Momentum", baseline["momentum_return"])
    ]:
        actual = baseline["actual_return"]
        error = prediction - actual

        rows.append({
            "Model": name,
            "MAE (pp)": np.mean(abs(error))*100,
            "RMSE (pp)": np.sqrt(np.mean(error**2))*100,
            "Direction (%)": np.mean(
                np.sign(prediction) == np.sign(actual)
            )*100
        })

    rows.append({
        "Model": "Always positive",
        "Direction (%)": (
            baseline["actual_return"] > 0
        ).mean()*100
    })

    st.dataframe(
        pd.DataFrame(rows).round(4),
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "Historical research only. "
        "Pretraining overlap has not been ruled out. "
        "Forecasting accuracy does not establish "
        "trading profitability."
    )

st.divider()
st.caption(
    "QuantEdge AI V1 | PyTorch · Chronos-2 · "
    "FinBERT · DuckDB · LoRA · QLoRA · NVIDIA Thor"
)
