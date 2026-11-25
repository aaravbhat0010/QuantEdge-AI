import os
import json
import time
import requests
import pandas as pd
import duckdb
from pathlib import Path

key = os.environ["FRED_API_KEY"].strip()

series = {
    "DGS10": "treasury_10_year",
    "DGS2": "treasury_2_year",
    "VIXCLS": "volatility_index"
}

url = "https://api.stlouisfed.org/fred/series/observations"

session = requests.Session()
con = duckdb.connect("data/quant_edge.duckdb")

for sid, name in series.items():
    observations = []
    failed = False

    print(f"\n=== DOWNLOADING {sid} ===")

    for start_year, end_year in [
        (2015, 2017),
        (2018, 2020),
        (2021, 2023)
    ]:
        params = {
            "series_id": sid,
            "api_key": key,
            "file_type": "json",
            "observation_start": "2014-01-01",
            "observation_end": "2023-12-31",
            "realtime_start": f"{start_year}-01-01",
            "realtime_end": f"{end_year}-12-31",
            "output_type": 1,
            "limit": 100000
        }

        try:
            response = session.get(
                url, params=params, timeout=120
            )

            if response.status_code != 200:
                print(
                    "API ERROR:",
                    response.status_code,
                    response.text[:500].replace(
                        key, "[REDACTED]"
                    )
                )
                failed = True
                break

            payload = response.json()
            batch = payload["observations"]

            if len(batch) < payload["count"]:
                print("Incomplete response:", start_year)
                failed = True
                break

            observations.extend(batch)

            print(f"{start_year}-{end_year}: {len(batch)} records")
            time.sleep(0.7)

        except Exception as e:
            print("FAILED:", type(e).__name__)
            failed = True
            break

    if failed:
        print(f"Skipping incomplete series: {sid}")
        continue

    rows = [
        {
            "series_id": sid,
            "indicator": name,
            "observation_date": item["date"],
            "available_from": item["realtime_start"],
            "available_until": item["realtime_end"],
            "value": float(item["value"])
        }
        for item in observations
        if item["value"] != "."
    ]

    if not rows:
        print("No valid observations:", sid)
        continue

    Path(f"data/raw/fred/{sid}.json").write_text(
        json.dumps(observations)
    )

    df = pd.DataFrame(rows).drop_duplicates()

    for col in [
        "observation_date",
        "available_from",
        "available_until"
    ]:
        df[col] = pd.to_datetime(df[col])

    con.register("recovered_df", df)

    con.execute("BEGIN TRANSACTION")

    try:
        con.execute(
            "DELETE FROM macroeconomic_facts WHERE series_id = ?",
            [sid]
        )

        con.execute("""
        INSERT INTO macroeconomic_facts
        SELECT * FROM recovered_df
        """)

        con.execute("COMMIT")

    except Exception:
        con.execute("ROLLBACK")
        raise

    con.unregister("recovered_df")

    print(f"SUCCESS: {sid} | {len(df)} observations")

con.execute("""
COPY macroeconomic_facts
TO 'data/processed/macroeconomic_facts.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== FINAL MACROECONOMIC AUDIT ===")

print(con.execute("""
SELECT
    series_id,
    COUNT(*) AS observations,
    COUNT(DISTINCT observation_date) AS unique_dates
FROM macroeconomic_facts
GROUP BY series_id
ORDER BY series_id
""").df().to_string(index=False))

print(con.execute("""
SELECT
    COUNT(DISTINCT series_id) AS indicators,
    COUNT(*) AS total_observations
FROM macroeconomic_facts
""").df().to_string(index=False))

con.close()
