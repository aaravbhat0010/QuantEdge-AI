from pathlib import Path
import duckdb

con = duckdb.connect("data/quant_edge.duckdb")

con.execute("SET threads = 8")
con.execute("SET memory_limit = '16GB'")

print("\n=== NORMALIZING FINANCIAL RECORDS ===")

con.execute("""
    CREATE OR REPLACE TABLE fundamental_normalized AS

    SELECT *,
        CASE
            WHEN period_start IS NULL
                THEN 'snapshot'
            WHEN date_diff(
                'day', period_start, period_end
            ) BETWEEN 70 AND 110
                THEN 'quarterly'
            ELSE 'other'
        END AS period_type

    FROM fundamental_facts

    WHERE period_end <= filed_date
        AND value IS NOT NULL
        AND isfinite(value)
""")

# Retain comparable reporting periods
con.execute("""
    CREATE OR REPLACE TABLE fundamental_eligible AS

    SELECT *
    FROM fundamental_normalized

    WHERE
        (
            metric IN (
                'assets', 'liabilities', 'equity'
            )
            AND period_type = 'snapshot'
        )
        OR
        (
            metric IN (
                'revenue', 'net_income', 'eps',
                'gross_profit', 'operating_income',
                'operating_cash_flow'
            )
            AND period_type = 'quarterly'
        )
""")

print("\n=== ELIGIBLE FINANCIAL RECORDS ===")

print(con.execute("""
    SELECT
        metric,
        COUNT(*) AS observations,
        COUNT(DISTINCT ticker) AS companies
    FROM fundamental_eligible
    GROUP BY metric
    ORDER BY metric
""").df().to_string(index=False))

print("\n=== ALIGNING FILINGS TO TRADING DAYS ===")

# For each trading date and financial metric,
# select the most recent reporting period
# that was publicly filed before that date.
#
# Older periods repeated in newer filings
# must not replace newer financial periods.

con.execute("""
    CREATE OR REPLACE TABLE fundamental_daily_long AS

    SELECT
        p.ticker,
        CAST(p.date AS DATE) AS date,
        f.metric,
        f.value,
        f.period_start,
        f.period_end,
        f.filed_date,
        f.accession,
        f.cik

    FROM price_features p

    INNER JOIN fundamental_eligible f
        ON p.ticker = f.ticker
        AND CAST(f.filed_date AS DATE)
            < CAST(p.date AS DATE)

    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY p.ticker, p.date, f.metric
        ORDER BY
            f.period_end DESC,
            f.filed_date DESC,
            f.tag_priority ASC,
            f.accession DESC,
            f.tag ASC,
            f.period_start DESC,
            f.value DESC
    ) = 1
""")

violations = con.execute("""
    SELECT COUNT(*)
    FROM fundamental_daily_long
    WHERE CAST(filed_date AS DATE) >= date
""").fetchone()[0]

assert violations == 0, "Future filing detected"

print("Filing date violations:", violations)

print("\n=== BUILDING DAILY FEATURES ===")

metrics = [
    "revenue",
    "net_income",
    "eps",
    "assets",
    "liabilities",
    "equity",
    "operating_cash_flow",
    "gross_profit",
    "operating_income"
]

columns = []

for metric in metrics:
    columns.append(
        f"MAX(CASE WHEN metric = '{metric}' "
        f"THEN value END) AS {metric}"
    )

    columns.append(
        f"MAX(CASE WHEN metric = '{metric}' "
        f"THEN period_end END) AS {metric}_end"
    )

    columns.append(
        f"MAX(CASE WHEN metric = '{metric}' "
        f"THEN period_start END) AS {metric}_start"
    )

    columns.append(
        f"MAX(CASE WHEN metric = '{metric}' "
        f"THEN cik END) AS {metric}_cik"
    )

pivot_columns = ",\n".join(columns)

con.execute(f"""
    CREATE OR REPLACE TABLE fundamental_daily_pivot AS

    SELECT
        ticker,
        date,
        {pivot_columns}

    FROM fundamental_daily_long

    GROUP BY ticker, date
""")

# Calculate financial ratios only when
# their underlying reporting periods match.

con.execute("""
    CREATE OR REPLACE TABLE daily_fundamentals AS

    SELECT
        p.ticker,
        CAST(p.date AS DATE) AS date,

        f.revenue,
        f.net_income,
        f.eps,
        f.assets,
        f.liabilities,
        f.equity,
        f.operating_cash_flow,
        f.gross_profit,
        f.operating_income,

        CASE
            WHEN f.net_income_end = f.revenue_end
            AND f.net_income_start = f.revenue_start
            AND f.net_income_cik = f.revenue_cik
            AND f.revenue > 0
            THEN f.net_income / f.revenue
        END AS net_profit_margin,

        CASE
            WHEN f.gross_profit_end = f.revenue_end
            AND f.gross_profit_start = f.revenue_start
            AND f.gross_profit_cik = f.revenue_cik
            AND f.revenue > 0
            THEN f.gross_profit / f.revenue
        END AS gross_profit_margin,

        CASE
            WHEN f.operating_income_end = f.revenue_end
            AND f.operating_income_start = f.revenue_start
            AND f.operating_income_cik = f.revenue_cik
            AND f.revenue > 0
            THEN f.operating_income / f.revenue
        END AS operating_margin,

        CASE
            WHEN f.liabilities_end = f.assets_end
            AND f.liabilities_cik = f.assets_cik
            AND f.assets > 0
            THEN f.liabilities / f.assets
        END AS liabilities_to_assets,

        CASE
            WHEN f.equity_end = f.assets_end
            AND f.equity_cik = f.assets_cik
            AND f.assets > 0
            THEN f.equity / f.assets
        END AS equity_to_assets,

        date_diff(
            'day', f.revenue_end, p.date
        ) AS revenue_age_days,

        CASE
            WHEN f.ticker IS NULL THEN 0
            ELSE 1
        END AS has_fundamentals

    FROM price_features p

    LEFT JOIN fundamental_daily_pivot f
        ON p.ticker = f.ticker
        AND CAST(p.date AS DATE) = f.date

    ORDER BY p.ticker, p.date
""")

print("\n=== VALIDATING DATASET ===")

expected = con.execute("""
    SELECT COUNT(*) FROM price_features
""").fetchone()[0]

actual = con.execute("""
    SELECT COUNT(*) FROM daily_fundamentals
""").fetchone()[0]

assert actual == expected

print("Expected rows:", f"{expected:,}")
print("Actual rows:", f"{actual:,}")

print("\n=== FEATURE COVERAGE ===")

features = metrics + [
    "net_profit_margin",
    "gross_profit_margin",
    "operating_margin",
    "liabilities_to_assets",
    "equity_to_assets",
    "revenue_age_days"
]

for feature in features:
    available = con.execute(f"""
        SELECT COUNT({feature})
        FROM daily_fundamentals
    """).fetchone()[0]

    print(
        f"{feature:25s} "
        f"{available:>8,} / {actual:,} "
        f"({100 * available / actual:.2f}%)"
    )

print("\n=== NVIDIA SAMPLE ===")

print(con.execute("""
    SELECT
        ticker,
        date,
        ROUND(revenue / 1e9, 3)
            AS revenue_billions,
        ROUND(net_income / 1e9, 3)
            AS profit_billions,
        ROUND(net_profit_margin * 100, 2)
            AS profit_margin_pct,
        eps,
        revenue_age_days

    FROM daily_fundamentals

    WHERE ticker = 'NVDA'

    ORDER BY date DESC

    LIMIT 10
""").df().to_string(index=False))

output = "data/processed/daily_fundamentals.parquet"

con.execute(f"""
    COPY daily_fundamentals
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

print("\n=== FUNDAMENTAL FEATURES: PASSED ===")
