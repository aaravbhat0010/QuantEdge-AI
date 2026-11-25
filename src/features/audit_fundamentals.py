import duckdb

con = duckdb.connect(
    "data/quant_edge.duckdb",
    read_only=True
)

print("\n=== FUNDAMENTAL DATA OVERVIEW ===")

print(con.execute("""
    SELECT
        COUNT(*) AS observations,
        COUNT(DISTINCT ticker) AS companies,
        COUNT(DISTINCT metric) AS metrics,
        MIN(filed_date) AS first_filing,
        MAX(filed_date) AS last_filing
    FROM fundamental_facts
""").df().to_string(index=False))

print("\n=== METRIC COVERAGE ===")

print(con.execute("""
    SELECT
        metric,
        COUNT(*) AS observations,
        COUNT(DISTINCT ticker) AS companies,
        COUNT(*) FILTER (
            WHERE period_start IS NULL
        ) AS point_in_time_records
    FROM fundamental_facts
    GROUP BY metric
    ORDER BY metric
""").df().to_string(index=False))

print("\n=== REPORTING PERIOD TYPES ===")

print(con.execute("""
    WITH classified AS (
        SELECT *,
            CASE
                WHEN period_start IS NULL
                    THEN 'Balance sheet snapshot'
                WHEN date_diff(
                    'day', period_start, period_end
                ) BETWEEN 70 AND 110
                    THEN 'Quarterly'
                WHEN date_diff(
                    'day', period_start, period_end
                ) BETWEEN 150 AND 210
                    THEN 'Six months'
                WHEN date_diff(
                    'day', period_start, period_end
                ) BETWEEN 240 AND 300
                    THEN 'Nine months'
                WHEN date_diff(
                    'day', period_start, period_end
                ) BETWEEN 330 AND 390
                    THEN 'Annual'
                ELSE 'Other'
            END AS period_type
        FROM fundamental_facts
    )
    SELECT
        metric,
        period_type,
        COUNT(*) AS observations
    FROM classified
    GROUP BY metric, period_type
    ORDER BY metric, period_type
""").df().to_string(index=False))

print("\n=== NVIDIA REVENUE EXAMPLE ===")

print(con.execute("""
    SELECT
        ticker,
        metric,
        ROUND(value / 1e9, 3) AS revenue_billions,
        CAST(period_start AS DATE) AS period_start,
        CAST(period_end AS DATE) AS period_end,
        CAST(filed_date AS DATE) AS filed_date,
        form,
        fiscal_period
    FROM fundamental_facts
    WHERE ticker = 'NVDA'
        AND metric = 'revenue'
    ORDER BY filed_date DESC, period_end DESC
    LIMIT 20
""").df().to_string(index=False))

print("\n=== DATA QUALITY CHECK ===")

print(con.execute("""
    SELECT
        COUNT(*) FILTER (
            WHERE period_end > filed_date
        ) AS invalid_filing_dates,

        COUNT(*) FILTER (
            WHERE value IS NULL
        ) AS missing_values,

        COUNT(*) FILTER (
            WHERE period_start > period_end
        ) AS invalid_periods

    FROM fundamental_facts
""").df().to_string(index=False))

con.close()

print("\n=== FUNDAMENTAL AUDIT COMPLETE ===")
