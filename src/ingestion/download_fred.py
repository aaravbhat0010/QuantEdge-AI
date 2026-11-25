import os
import json
import time
import requests
import pandas as pd
import duckdb
from pathlib import Path

api_key = os.environ["FRED_API_KEY"].strip()

series = {
    "FEDFUNDS": "federal_funds_rate",
    "CPIAUCSL": "consumer_price_index",
    "UNRATE": "unemployment_rate",
    "GDPC1": "real_gdp",
    "DGS10": "treasury_10_year",
    "DGS2": "treasury_2_year",
    "VIXCLS": "volatility_index",
    "INDPRO": "industrial_production",
    "UMCSENT": "consumer_sentiment",
    "DCOILWTICO": "crude_oil_price",
    "M2SL": "money_supply",
    "RSAFS": "retail_sales"
}

session = requests.Session()

url = (
    "https://api.stlouisfed.org/"
    "fred/series/observations"
)

rows = []
failed = []

print("=== DOWNLOADING HISTORICAL ECONOMIC DATA ===")

for series_id, name in series.items():

    try:
        observations = []
        offset = 0

        while True:
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": "2014-01-01",
                "observation_end": "2023-12-31",
                "realtime_start": "2015-01-01",
                "realtime_end": "2023-12-31",
                "output_type": 1,
                "limit": 100000,
                "offset": offset
            }

            response = session.get(
                url, params=params, timeout=120
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}"
                )

            payload = response.json()
            batch = payload["observations"]
            observations.extend(batch)

            offset += len(batch)

            time.sleep(0.6)

            if offset >= payload["count"]:
                break

            if not batch:
                raise RuntimeError(
                    "Incomplete pagination"
                )

        path = Path(
            f"data/raw/fred/{series_id}.json"
        )

        path.write_text(json.dumps(observations))

        valid = 0

        for item in observations:
            if item["value"] == ".":
                continue

            rows.append({
                "series_id": series_id,
                "indicator": name,
                "observation_date": item["date"],
                "available_from": item["realtime_start"],
                "available_until": item["realtime_end"],
                "value": float(item["value"])
            })

            valid += 1

        print(
            f"SUCCESS: {series_id:12} | "
            f"{valid:,} observations"
        )

    except Exception as e:
        failed.append(series_id)
        print(f"FAILED: {series_id} | {type(e).__name__}")

if not rows:
    raise RuntimeError("No economic data downloaded")

df = pd.DataFrame(rows).drop_duplicates()

for col in [
    "observation_date",
    "available_from",
    "available_until"
]:
    df[col] = pd.to_datetime(df[col])

con = duckdb.connect("data/quant_edge.duckdb")

con.register("macro_df", df)

con.execute("""
CREATE OR REPLACE TABLE macroeconomic_facts AS
SELECT * FROM macro_df
""")

con.execute("""
COPY macroeconomic_facts
TO 'data/processed/macroeconomic_facts.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== MACROECONOMIC COVERAGE ===")

print(con.execute("""
SELECT
    series_id,
    COUNT(*) AS observations,
    COUNT(DISTINCT observation_date) AS unique_dates,
    MIN(observation_date)::DATE AS first_date,
    MAX(observation_date)::DATE AS last_date
FROM macroeconomic_facts
GROUP BY series_id
ORDER BY series_id
""").df().to_string(index=False))

print("\n=== FINAL DOWNLOAD AUDIT ===")
print("Indicators collected:", df["series_id"].nunique())
print("Total observations:", len(df))
print("Failed indicators:", failed)

con.close()
