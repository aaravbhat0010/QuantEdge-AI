import duckdb
from pathlib import Path

tickers = Path("configs/tickers.txt").read_text().split()

con = duckdb.connect("data/quant_edge.duckdb")

con.execute("SET memory_limit='16GB'")
con.execute("SET threads=8")
con.execute("SET TimeZone='UTC'")

con.execute("""
CREATE OR REPLACE TABLE companies (
    ticker VARCHAR PRIMARY KEY
)
""")

con.executemany(
    "INSERT INTO companies VALUES (?)",
    [(ticker,) for ticker in tickers]
)

print("=== FILTERING FINANCIAL NEWS ===")

con.execute("""
CREATE OR REPLACE TABLE financial_news AS

WITH parsed AS (
    SELECT
        TRY_CAST(Date AS TIMESTAMPTZ) AS published_at,
        CASE
            WHEN UPPER(TRIM(Stock_symbol)) = 'FB'
            THEN 'META'
            ELSE UPPER(TRIM(Stock_symbol))
        END AS ticker,
        TRIM(Article_title) AS headline,
        Url AS url,
        Publisher AS publisher
    FROM read_csv(
        'data/raw/fnspid/All_external.csv',
        header=true,
        all_varchar=true,
        max_line_size=16777216
    )
)

SELECT DISTINCT
    published_at,
    ticker,
    headline,
    url,
    publisher
FROM parsed
WHERE ticker IN (SELECT ticker FROM companies)
AND published_at >= TIMESTAMPTZ '2016-01-01 00:00:00+00'
AND published_at < TIMESTAMPTZ '2024-01-01 00:00:00+00'
AND headline IS NOT NULL
AND headline != ''
""")

print("\n=== EXPORT TO PARQUET ===")

con.execute("""
COPY financial_news
TO 'data/processed/financial_news.parquet'
(FORMAT PARQUET, COMPRESSION ZSTD)
""")

print("\n=== NEWS COVERAGE BY COMPANY ===")

print(con.execute("""
SELECT
    c.ticker,
    COUNT(n.ticker) AS articles,
    MIN(n.published_at)::DATE AS first_date,
    MAX(n.published_at)::DATE AS last_date,
    COUNT(DISTINCT YEAR(n.published_at)) AS years
FROM companies c
LEFT JOIN financial_news n
ON c.ticker = n.ticker
GROUP BY c.ticker
ORDER BY articles DESC
""").df().to_string(index=False))

print("\n=== NEWS COVERAGE BY YEAR ===")

print(con.execute("""
SELECT
    YEAR(published_at) AS year,
    COUNT(*) AS articles,
    COUNT(DISTINCT ticker) AS companies
FROM financial_news
GROUP BY year
ORDER BY year
""").df().to_string(index=False))

print("\n=== TOTAL NEWS RECORDS ===")

print(con.execute("""
SELECT COUNT(*) FROM financial_news
""").fetchone()[0])

con.close()
