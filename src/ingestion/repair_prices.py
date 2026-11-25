from pathlib import Path
import pandas as pd
import numpy as np
import shutil

tickers = ["AAPL", "AMZN", "GE", "NEE", "TSLA"]

source = Path("data/raw/price_verification")
destination = Path("data/raw/prices")
backup = Path("data/raw/prices_before_repair")

backup.mkdir(parents=True, exist_ok=True)

prepared = {}

for ticker in tickers:

    df = pd.read_csv(source / f"{ticker}.csv")

    df.columns = df.columns.str.strip().str.lower()

    df["date"] = pd.to_datetime(df["date"]).dt.date

    columns = [
        "date", "open", "high", "low",
        "close", "adj close", "volume"
    ]

    df = df[columns].sort_values("date")

    assert len(df) == 2012, ticker
    assert not df["date"].duplicated().any(), ticker
    assert np.isfinite(df[columns[1:]]).all().all()
    assert (df[columns[1:6]] > 0).all().all()
    assert (df["volume"] >= 0).all()

    prepared[ticker] = df

for ticker, df in prepared.items():

    original = destination / f"{ticker}.csv"
    backup_file = backup / f"{ticker}.csv"

    if not backup_file.exists():
        shutil.copy2(original, backup_file)

    df.to_csv(original, index=False)

    print(f"{ticker}: replaced {len(df)} records")

print("\n=== PRICE REPAIR: PASSED ===")
