from pathlib import Path
import duckdb

con = duckdb.connect("data/quant_edge.duckdb")

con.execute("SET threads = 8")
con.execute("SET memory_limit = '16GB'")

print("\n=== SOURCE DATASETS ===")

tables = [
    "price_features",
    "daily_sentiment",
    "daily_fundamentals",
    "daily_macro"
]

# Verify row counts and unique join keys
for table in tables:

    keys = "date" if table == "daily_macro" else "ticker, date"

    count = con.execute(f"""
        SELECT COUNT(*) FROM {table}
    """).fetchone()[0]

    duplicates = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT {keys}
            FROM {table}
            GROUP BY {keys}
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    assert duplicates == 0, f"Duplicates in {table}"

    print(f"{table:25s} {count:>10,} rows")

print("\n=== COMBINING ALL DATASETS ===")

con.execute("""
    CREATE OR REPLACE TABLE master_dataset AS

    SELECT
        p.*,
        s.* EXCLUDE (ticker, date),
        f.* EXCLUDE (ticker, date),
        m.* EXCLUDE (date)

    FROM price_features p

    LEFT JOIN daily_sentiment s
        ON p.ticker = s.ticker
        AND CAST(p.date AS DATE) = s.date

    LEFT JOIN daily_fundamentals f
        ON p.ticker = f.ticker
        AND CAST(p.date AS DATE) = f.date

    LEFT JOIN daily_macro m
        ON CAST(p.date AS DATE) = m.date

    ORDER BY p.ticker, p.date
""")

print("\n=== MASTER DATASET AUDIT ===")

summary = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT ticker) AS companies,
        COUNT(DISTINCT date) AS trading_dates,
        MIN(date) AS first_date,
        MAX(date) AS last_date
    FROM master_dataset
""").df()

print(summary.to_string(index=False))

rows = int(summary["total_rows"].iloc[0])

expected = con.execute("""
    SELECT COUNT(*) FROM price_features
""").fetchone()[0]

assert rows == expected
assert int(summary["companies"].iloc[0]) == 50

schema = con.execute("""
    DESCRIBE master_dataset
""").df()

features = [
    name for name in schema["column_name"]
    if name not in ["ticker", "date"]
]

print("\nTotal columns:", len(schema))
print("Feature columns:", len(features))

print("\n=== FEATURE COVERAGE ===")

for feature in features:

    available = con.execute(f"""
        SELECT COUNT("{feature}")
        FROM master_dataset
    """).fetchone()[0]

    coverage = 100 * available / rows

    print(
        f"{feature:32s} "
        f"{coverage:6.2f}% available"
    )

print("\n=== RECORDS PER YEAR ===")

print(con.execute("""
    SELECT
        YEAR(date) AS year,
        COUNT(*) AS records,
        COUNT(DISTINCT ticker) AS companies
    FROM master_dataset
    GROUP BY YEAR(date)
    ORDER BY year
""").df().to_string(index=False))

print("\n=== NVIDIA SAMPLE ===")

print(con.execute("""
    SELECT
        ticker,
        date,
        open,
        close,
        ROUND(return_1d, 4) AS daily_return,
        ROUND(rsi_14, 2) AS rsi,
        news_count,
        ROUND(sentiment_mean, 4) AS sentiment,
        ROUND(revenue / 1e9, 3) AS revenue_billions,
        federal_funds_rate,
        volatility_index

    FROM master_dataset

    WHERE ticker = 'NVDA'

    ORDER BY date DESC

    LIMIT 10
""").df().to_string(index=False))

print("\n=== SAVING MASTER DATASET ===")

output = "data/processed/master_dataset.parquet"

con.execute(f"""
    COPY master_dataset
    TO '{output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

saved = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{output}')
""").fetchone()[0]

assert saved == rows

print("Saved rows:", f"{saved:,}")
print("Features:", len(features))
print("File size:", round(
    Path(output).stat().st_size / 1024**2,
    2
), "MiB")

con.close()

print("\n=== MASTER DATASET: PASSED ===")
