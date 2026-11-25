import json
import pandas as pd
import duckdb
from pathlib import Path

metrics = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet"
    ],
    "net_income": ["NetIncomeLoss"],
    "eps": ["EarningsPerShareDiluted"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity"],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities"
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"]
}

rows = []

paths = sorted(
    Path("data/raw/sec/companyfacts").glob("*.json")
)

paths += [
    Path("data/raw/sec/historical/XOM_0000034088.json"),
    Path("data/raw/sec/historical/DIS_0001001039.json")
]

for path in paths:
    ticker = path.stem.split("_")[0]
    data = json.loads(path.read_text())
    facts = data["facts"].get("us-gaap", {})

    for metric, tags in metrics.items():
        unit = "USD/shares" if metric == "eps" else "USD"

        for priority, tag in enumerate(tags):
            observations = (
                facts.get(tag, {})
                .get("units", {})
                .get(unit, [])
            )

            for item in observations:
                filed = item.get("filed", "")
                end = item.get("end", "")

                if item.get("form") not in [
                    "10-K", "10-Q", "10-K/A", "10-Q/A"
                ]:
                    continue

                if not (
                    "2015-01-01" <= filed < "2024-01-01"
                ):
                    continue

                if not end or end > filed:
                    continue

                rows.append({
                    "ticker": ticker,
                    "cik": str(data["cik"]).zfill(10),
                    "metric": metric,
                    "tag": tag,
                    "tag_priority": priority,
                    "value": item["val"],
                    "unit": unit,
                    "period_start": item.get("start"),
                    "period_end": end,
                    "filed_date": filed,
                    "form": item.get("form"),
                    "accession": item.get("accn"),
                    "fiscal_year": item.get("fy"),
                    "fiscal_period": item.get("fp")
                })

df = pd.DataFrame(rows).drop_duplicates()

if df.empty:
    raise RuntimeError("No historical fundamentals found")

for col in [
    "period_start", "period_end", "filed_date"
]:
    df[col] = pd.to_datetime(df[col])

df = df.sort_values([
    "ticker", "filed_date", "metric", "period_end"
])

con = duckdb.connect("data/quant_edge.duckdb")

con.register("fundamental_df", df)

con.execute("""
CREATE OR REPLACE TABLE fundamental_facts AS
SELECT * FROM fundamental_df
""")

con.execute("""
COPY fundamental_facts
TO 'data/processed/fundamental_facts.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== FUNDAMENTAL COVERAGE ===")

print(con.execute("""
SELECT
    c.ticker,
    COUNT(f.ticker) AS observations,
    COUNT(DISTINCT f.metric) AS metrics,
    MIN(f.filed_date)::DATE AS first_filing,
    MAX(f.filed_date)::DATE AS last_filing
FROM companies c
LEFT JOIN fundamental_facts f
ON c.ticker = f.ticker
GROUP BY c.ticker
ORDER BY observations ASC
""").df().to_string(index=False))

print("\n=== METRIC COVERAGE ===")

print(con.execute("""
SELECT
    metric,
    COUNT(*) AS observations,
    COUNT(DISTINCT ticker) AS companies
FROM fundamental_facts
GROUP BY metric
ORDER BY metric
""").df().to_string(index=False))

print("\nTotal fundamental observations:", len(df))

con.close()
