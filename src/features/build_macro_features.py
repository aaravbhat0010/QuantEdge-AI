from pathlib import Path
import duckdb

con = duckdb.connect("data/quant_edge.duckdb")

con.execute("SET threads = 8")
con.execute("SET memory_limit = '16GB'")

print("\n=== MACROECONOMIC SOURCE DATA ===")

print(con.execute("""
    SELECT
        series_id,
        indicator,
        COUNT(*) AS observations
    FROM macroeconomic_facts
    GROUP BY series_id, indicator
    ORDER BY series_id
""").df().to_string(index=False))

# Create a common trading calendar
con.execute("""
    CREATE OR REPLACE TABLE trading_calendar AS
    SELECT DISTINCT CAST(date AS DATE) AS date
    FROM price_features
    ORDER BY date
""")

print("\n=== POINT-IN-TIME ALIGNMENT ===")

# Use only observations and historical vintages
# available before the prediction date.

con.execute("""
    CREATE OR REPLACE TABLE macro_daily_long AS

    SELECT
        t.date,
        m.series_id,
        m.indicator,
        m.value,
        m.observation_date,
        m.available_from,
        m.available_until

    FROM trading_calendar t

    INNER JOIN macroeconomic_facts m

        ON CAST(m.available_from AS DATE) < t.date

        AND CAST(m.observation_date AS DATE) < t.date

        AND CAST(m.available_until AS DATE)
            >= t.date - INTERVAL 1 DAY

    WHERE m.value IS NOT NULL
        AND isfinite(m.value)

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY t.date, m.series_id
        ORDER BY
            m.observation_date DESC,
            m.available_from DESC
    ) = 1
""")

# Check for future information
violations = con.execute("""
    SELECT COUNT(*)
    FROM macro_daily_long
    WHERE
        CAST(available_from AS DATE) >= date
        OR CAST(observation_date AS DATE) >= date
        OR CAST(available_until AS DATE)
            < date - INTERVAL 1 DAY
""").fetchone()[0]

assert violations == 0

print("Chronological violations:", violations)

indicators = [
    "federal_funds_rate",
    "consumer_price_index",
    "unemployment_rate",
    "real_gdp",
    "treasury_10_year",
    "treasury_2_year",
    "volatility_index",
    "industrial_production",
    "consumer_sentiment",
    "crude_oil_price",
    "money_supply",
    "retail_sales"
]

columns = ",\n".join([
    f"MAX(CASE WHEN indicator = '{name}' "
    f"THEN value END) AS {name}"
    for name in indicators
])

print("\n=== BUILDING DAILY MACRO FEATURES ===")

con.execute(f"""
    CREATE OR REPLACE TABLE macro_daily_pivot AS

    SELECT
        date,
        {columns}

    FROM macro_daily_long

    GROUP BY date
""")

con.execute("""
    CREATE OR REPLACE TABLE daily_macro AS

    SELECT
        t.date,
        m.* EXCLUDE (date),

        m.treasury_10_year - m.treasury_2_year
            AS treasury_yield_spread

    FROM trading_calendar t

    LEFT JOIN macro_daily_pivot m
        ON t.date = m.date

    ORDER BY t.date
""")

print("\n=== VALIDATING DATASET ===")

expected = con.execute("""
    SELECT COUNT(*) FROM trading_calendar
""").fetchone()[0]

actual = con.execute("""
    SELECT COUNT(*) FROM daily_macro
""").fetchone()[0]

assert actual == expected

print("Trading dates:", actual)

print("\n=== FEATURE COVERAGE ===")

for feature in indicators + ["treasury_yield_spread"]:

    available = con.execute(f"""
        SELECT COUNT({feature})
        FROM daily_macro
    """).fetchone()[0]

    print(
        f"{feature:25s} "
        f"{available:>5,} / {actual:,} "
        f"({100 * available / actual:.2f}%)"
    )

print("\n=== SAMPLE MACRO DATA ===")

print(con.execute("""
    SELECT
        date,
        federal_funds_rate,
        unemployment_rate,
        treasury_10_year,
        treasury_2_year,
        treasury_yield_spread,
        volatility_index

    FROM daily_macro

    ORDER BY date DESC

    LIMIT 10
""").df().to_string(index=False))

output = "data/processed/daily_macro.parquet"

con.execute(f"""
    COPY daily_macro
    TO '{output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

saved = con.execute(f"""
    SELECT COUNT(*) FROM read_parquet('{output}')
""").fetchone()[0]

assert saved == expected

print("\n=== SAVED DATASET ===")

print("Rows:", f"{saved:,}")

print("File size:", round(
    Path(output).stat().st_size / 1024**2, 2
), "MiB")

con.close()

print("\n=== MACRO FEATURES: PASSED ===")
