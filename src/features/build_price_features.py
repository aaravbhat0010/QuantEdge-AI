from pathlib import Path
import pandas as pd
import numpy as np
import duckdb

tickers = Path("configs/tickers.txt").read_text().split()
frames = []

for ticker in tickers:
    path = Path(f"data/raw/prices/{ticker}.csv")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"])

    df = df[df["date"].between(
        "2016-01-01", "2023-12-28"
    )].copy()

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError(f"Duplicate dates: {ticker}")

    cols = ["open", "high", "low", "close",
            "adj close", "volume"]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if not np.isfinite(df[cols].to_numpy()).all():
        raise ValueError(f"Invalid values: {ticker}")

    if (df[cols[:-1]] <= 0).any().any():
        raise ValueError(f"Invalid prices: {ticker}")

    if (df["volume"] < 0).any():
        raise ValueError(f"Invalid volume: {ticker}")

    df["ticker"] = ticker

    price = df["adj close"]

    df["return_1d"] = price.pct_change()
    df["return_5d"] = price.pct_change(5)
    df["return_20d"] = price.pct_change(20)

    sma20 = price.rolling(20).mean()
    sma50 = price.rolling(50).mean()

    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()

    df["sma_20_ratio"] = price / sma20 - 1
    df["sma_50_ratio"] = price / sma50 - 1
    df["ema_12_ratio"] = price / ema12 - 1
    df["ema_26_ratio"] = price / ema26 - 1
    df["macd_ratio"] = (ema12 - ema26) / price

    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1/14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_14"] = 100 - 100 / (1 + rs)

    df.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "rsi_14"
    ] = 100

    df.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "rsi_14"
    ] = 50

    df["volatility_20"] = (
        df["return_1d"].rolling(20).std()
        * np.sqrt(252)
    )

    df["relative_volume"] = (
        df["volume"] /
        df["volume"].rolling(20).mean().replace(0, np.nan)
    )

    df = df[
        (df["date"] >= "2016-01-01") &
        (df["date"] <= "2023-12-28")
    ].copy()

    frames.append(df)

    print(f"SUCCESS: {ticker:6} | {len(df):,} rows")

result = pd.concat(frames, ignore_index=True)
result = result.sort_values(["ticker", "date"])

result = result.rename(
    columns={"adj close": "adjusted_close"}
)

con = duckdb.connect("data/quant_edge.duckdb")

con.register("price_df", result)

con.execute("""
CREATE OR REPLACE TABLE price_features AS
SELECT * FROM price_df
""")

con.execute("""
COPY price_features
TO 'data/processed/price_features.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== PRICE FEATURE AUDIT ===")

print(con.execute("""
SELECT
    COUNT(DISTINCT ticker) AS companies,
    COUNT(*) AS records,
    MIN(date)::DATE AS first_date,
    MAX(date)::DATE AS last_date
FROM price_features
""").df().to_string(index=False))

print("\n=== MISSING FEATURE VALUES ===")

print(result.isna().sum().to_string())

print("\n=== SAMPLE FEATURES ===")

print(con.execute("""
SELECT ticker, date, close, return_1d,
       rsi_14, volatility_20
FROM price_features
WHERE ticker = 'NVDA'
ORDER BY date DESC
LIMIT 5
""").df().to_string(index=False))

con.close()
