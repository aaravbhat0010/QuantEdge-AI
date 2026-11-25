import duckdb
import time

start = time.time()

con = duckdb.connect("data/quant_edge.duckdb")

con.execute("SET memory_limit='24GB'")
con.execute("SET threads=8")
con.execute("SET TimeZone='UTC'")

print("=== FILTERING EXTENDED NEWS DATASET ===")

con.execute("""
CREATE OR REPLACE TABLE nasdaq_news AS

WITH parsed AS (
    SELECT
        TRY_CAST(Date AS TIMESTAMPTZ) AS published_at,
        CASE
            WHEN UPPER(TRIM(Stock_symbol)) = 'FB'
            THEN 'META'
            ELSE UPPER(TRIM(Stock_symbol))
        END AS ticker,
        TRIM(Article_title) AS headline,
        TRIM(Url) AS url,
        Publisher AS publisher
    FROM read_csv(
        'data/raw/fnspid/nasdaq_exteral_data.csv',
        header=true,
        all_varchar=true,
        max_line_size=16777216,
        strict_mode=false,
        ignore_errors=true,
        store_rejects=true
    )
)

SELECT DISTINCT *
FROM parsed
WHERE ticker IN (SELECT ticker FROM companies)
AND published_at >= TIMESTAMPTZ '2016-01-01 00:00:00+00'
AND published_at < TIMESTAMPTZ '2024-01-01 00:00:00+00'
AND headline IS NOT NULL
AND headline != ''
""")

print("NASDAQ filtered records:",
      con.execute(
          "SELECT COUNT(*) FROM nasdaq_news"
      ).fetchone()[0])

print("\n=== COMBINING DATASETS ===")

con.execute("""
CREATE OR REPLACE TABLE financial_news_combined AS

WITH combined AS (
    SELECT *, 'external' AS source
    FROM financial_news

    UNION ALL

    SELECT *, 'nasdaq' AS source
    FROM nasdaq_news
)

SELECT *
FROM combined

QUALIFY ROW_NUMBER() OVER (
    PARTITION BY
        ticker,
        CAST(published_at AS DATE),
        LOWER(TRIM(headline))
    ORDER BY published_at, source, url
) = 1
""")

print("\n=== EXPORTING COMBINED PARQUET ===")

con.execute("""
COPY (
    SELECT *
    FROM financial_news_combined
    ORDER BY ticker, published_at
)
TO 'data/processed/financial_news_combined.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== COVERAGE BY COMPANY ===")

print(con.execute("""
SELECT
    c.ticker,
    COUNT(n.ticker) AS articles,
    MIN(n.published_at)::DATE AS first_date,
    MAX(n.published_at)::DATE AS last_date,
    COUNT(DISTINCT YEAR(n.published_at)) AS years
FROM companies c
LEFT JOIN financial_news_combined n
ON c.ticker = n.ticker
GROUP BY c.ticker
ORDER BY articles DESC
""").df().to_string(index=False))

print("\n=== COVERAGE BY YEAR ===")

print(con.execute("""
SELECT
    YEAR(published_at) AS year,
    COUNT(*) AS articles,
    COUNT(DISTINCT ticker) AS companies
FROM financial_news_combined
GROUP BY year
ORDER BY year
""").df().to_string(index=False))

print("\n=== FINAL STATISTICS ===")

print(con.execute("""
SELECT
    COUNT(*) AS total_news_records,
    COUNT(DISTINCT ticker) AS companies,
    MIN(published_at)::DATE AS first_date,
    MAX(published_at)::DATE AS last_date
FROM financial_news_combined
""").df().to_string(index=False))

print(f"\nProcessing time: {time.time()-start:.2f} seconds")

print("\n=== CSV REJECTION AUDIT ===")
con.execute("""
CREATE OR REPLACE TABLE news_csv_rejections AS
SELECT * FROM reject_errors
""")
print(con.execute("""
SELECT error_type, COUNT(*) AS errors
FROM news_csv_rejections
GROUP BY error_type
""").fetchall())

con.close()
