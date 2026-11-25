from pathlib import Path
import duckdb

con = duckdb.connect("data/quant_edge.duckdb")

con.execute("SET threads = 8")
con.execute("SET memory_limit = '16GB'")

print("\n=== AUDITING MASTER DATASET ===")

print(con.execute("""
    SELECT
        COUNT(*) AS records,
        COUNT(DISTINCT ticker) AS companies,
        COUNT(DISTINCT date) AS trading_days,
        COUNT(*) FILTER (
            WHERE adjusted_close IS NULL
            OR adjusted_close <= 0
            OR NOT isfinite(adjusted_close)
        ) AS invalid_prices
    FROM master_dataset
""").df().to_string(index=False))

# Verify that all companies share the same calendar.

calendar_errors = con.execute("""
    SELECT COUNT(*) FROM (
        SELECT date
        FROM master_dataset
        GROUP BY date
        HAVING COUNT(*) != 50
        OR COUNT(DISTINCT ticker) != 50
    )
""").fetchone()[0]

assert calendar_errors == 0

invalid_prices = con.execute("""
    SELECT COUNT(*)
    FROM master_dataset
    WHERE adjusted_close IS NULL
        OR adjusted_close <= 0
        OR NOT isfinite(adjusted_close)
""").fetchone()[0]

assert invalid_prices == 0

print("\n=== CREATING FUTURE TARGETS ===")

con.execute("""
    CREATE OR REPLACE TABLE forecast_dataset AS

    WITH targets AS (
        SELECT *,

            LEAD(adjusted_close, 5) OVER (
                PARTITION BY ticker
                ORDER BY date
            ) AS future_price_5d,

            LEAD(date, 5) OVER (
                PARTITION BY ticker
                ORDER BY date
            ) AS target_date,

            ROW_NUMBER() OVER (
                PARTITION BY ticker
                ORDER BY date
            ) AS history_days

        FROM master_dataset
    )

    SELECT *,

        future_price_5d / adjusted_close - 1
            AS target_return_5d,

        CASE
            WHEN future_price_5d > adjusted_close THEN 1
            WHEN future_price_5d < adjusted_close THEN -1
            WHEN future_price_5d = adjusted_close THEN 0
        END AS target_direction,

        CASE
            WHEN history_days < 60
                THEN 'insufficient_history'

            WHEN target_date IS NULL
                THEN 'unavailable'

            WHEN date < DATE '2021-01-01'
                AND target_date < DATE '2021-01-01'
                THEN 'train'

            WHEN date >= DATE '2021-01-01'
                AND date < DATE '2022-01-01'
                AND target_date < DATE '2022-01-01'
                THEN 'validation'

            WHEN date >= DATE '2022-01-01'
                AND target_date < DATE '2024-01-01'
                THEN 'test'

            ELSE 'boundary_excluded'

        END AS dataset_split

    FROM targets

    ORDER BY ticker, date
""")

print("\n=== CHRONOLOGICAL SPLITS ===")

print(con.execute("""
    SELECT
        dataset_split,
        COUNT(*) AS records,
        COUNT(DISTINCT ticker) AS companies,
        MIN(date) AS first_date,
        MAX(date) AS last_date,
        MAX(target_date) AS last_target_date

    FROM forecast_dataset

    GROUP BY dataset_split
    ORDER BY first_date
""").df().to_string(index=False))

print("\n=== LEAKAGE VALIDATION ===")

violations = con.execute("""
    SELECT COUNT(*)
    FROM forecast_dataset
    WHERE
        (dataset_split = 'train'
            AND target_date >= DATE '2021-01-01')

        OR

        (dataset_split = 'validation'
            AND target_date >= DATE '2022-01-01')

        OR

        (dataset_split IN (
            'train', 'validation', 'test'
        ) AND (
            target_date <= date
            OR target_date IS NULL
            OR target_return_5d IS NULL
            OR NOT isfinite(target_return_5d)
            OR history_days < 60
        ))
""").fetchone()[0]

assert violations == 0

print("Target boundary violations:", violations)

print("\n=== NVIDIA TARGET EXAMPLE ===")

print(con.execute("""
    SELECT
        ticker,
        date,
        ROUND(adjusted_close, 2) AS current_price,
        ROUND(future_price_5d, 2) AS future_price,
        target_date,
        ROUND(target_return_5d * 100, 3)
            AS future_return_pct,
        target_direction,
        dataset_split

    FROM forecast_dataset

    WHERE ticker = 'NVDA'
        AND dataset_split = 'test'

    ORDER BY date DESC
    LIMIT 10
""").df().to_string(index=False))

print("\n=== FEATURE COVERAGE BY SPLIT ===")

features = [
    row[0] for row in con.execute(
        "DESCRIBE master_dataset"
    ).fetchall()
    if row[0] not in ("ticker", "date")
]

for feature in features:

    coverage = con.execute(f"""
        SELECT
            dataset_split,
            ROUND(
                100.0 * COUNT("{feature}") / COUNT(*),
                2
            ) AS coverage

        FROM forecast_dataset

        WHERE dataset_split IN (
            'train', 'validation', 'test'
        )

        GROUP BY dataset_split
    """).fetchall()

    values = dict(coverage)

    print(
        f"{feature:32s} "
        f"train={values.get('train', 0):6.2f}% "
        f"val={values.get('validation', 0):6.2f}% "
        f"test={values.get('test', 0):6.2f}%"
    )

output = "data/processed/forecast_dataset.parquet"

con.execute(f"""
    COPY forecast_dataset
    TO '{output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== SAVED FORECAST DATASET ===")

print("File size:", round(
    Path(output).stat().st_size / 1024**2, 2
), "MiB")

con.close()

print("\n=== FORECAST TARGETS: PASSED ===")
