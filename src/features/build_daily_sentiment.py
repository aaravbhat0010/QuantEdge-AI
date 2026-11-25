from pathlib import Path
import duckdb

DB_PATH = "data/quant_edge.duckdb"
OUTPUT_PATH = "data/processed/daily_sentiment.parquet"

con = duckdb.connect(DB_PATH)

con.execute("SET TimeZone = 'UTC'")
con.execute("SET threads = 8")
con.execute("SET memory_limit = '16GB'")

print("\n=== SOURCE DATA ===")

total_news = con.execute("""
    SELECT COUNT(*) FROM news_sentiment
""").fetchone()[0]

total_prices = con.execute("""
    SELECT COUNT(*) FROM price_features
""").fetchone()[0]

print("News headlines:", f"{total_news:,}")
print("Stock trading rows:", f"{total_prices:,}")

# Ensure unique company-date price records
duplicates = con.execute("""
    SELECT COUNT(*)
    FROM (
        SELECT ticker, date
        FROM price_features
        GROUP BY ticker, date
        HAVING COUNT(*) > 1
    )
""").fetchone()[0]

assert duplicates == 0, "Duplicate trading dates"

print("\n=== ALIGNING NEWS TO TRADING DAYS ===")

# Assign each headline to the first trading day
# strictly after its recorded UTC publication date.

con.execute("""
    CREATE OR REPLACE TABLE news_trading_aligned AS

    SELECT
        n.*,
        CAST(n.published_at AS DATE) AS news_date,
        CAST(p.date AS DATE) AS trading_date

    FROM news_sentiment n

    ASOF LEFT JOIN price_features p

        ON n.ticker = p.ticker

        AND CAST(n.published_at AS DATE)
            < CAST(p.date AS DATE)
""")

# Verify chronological alignment
violations = con.execute("""
    SELECT COUNT(*)
    FROM news_trading_aligned
    WHERE trading_date <= news_date
""").fetchone()[0]

assert violations == 0, "Data leakage detected"

aligned = con.execute("""
    SELECT COUNT(*)
    FROM news_trading_aligned
    WHERE trading_date IS NOT NULL
""").fetchone()[0]

unaligned = con.execute("""
    SELECT COUNT(*)
    FROM news_trading_aligned
    WHERE trading_date IS NULL
""").fetchone()[0]

assert aligned + unaligned == total_news

print("Aligned headlines:", f"{aligned:,}")
print("Outside trading range:", f"{unaligned:,}")
print("Chronological violations:", violations)

print("\n=== AGGREGATING DAILY SENTIMENT ===")

con.execute("""
    CREATE OR REPLACE TABLE daily_sentiment_observed AS

    SELECT
        ticker,
        trading_date AS date,

        COUNT(*) AS news_count,

        AVG(sentiment_score) AS sentiment_mean,

        STDDEV_POP(sentiment_score)
            AS sentiment_std,

        MIN(sentiment_score) AS sentiment_min,

        MAX(sentiment_score) AS sentiment_max,

        AVG(positive_probability)
            AS positive_prob_mean,

        AVG(negative_probability)
            AS negative_prob_mean,

        AVG(neutral_probability)
            AS neutral_prob_mean,

        AVG(confidence)
            AS sentiment_confidence_mean,

        AVG(
            CASE WHEN predicted_sentiment = 'positive'
            THEN 1.0 ELSE 0.0 END
        ) AS positive_news_ratio,

        AVG(
            CASE WHEN predicted_sentiment = 'negative'
            THEN 1.0 ELSE 0.0 END
        ) AS negative_news_ratio,

        AVG(
            CASE WHEN predicted_sentiment = 'neutral'
            THEN 1.0 ELSE 0.0 END
        ) AS neutral_news_ratio

    FROM news_trading_aligned

    WHERE trading_date IS NOT NULL

    GROUP BY ticker, trading_date
""")

# Include every company trading day.
# Missing sentiment remains NULL, not neutral.

con.execute("""
    CREATE OR REPLACE TABLE daily_sentiment AS

    SELECT
        p.ticker,
        CAST(p.date AS DATE) AS date,

        COALESCE(s.news_count, 0) AS news_count,

        CASE WHEN s.news_count IS NULL
            THEN 0 ELSE 1
        END AS has_news,

        s.sentiment_mean,
        s.sentiment_std,
        s.sentiment_min,
        s.sentiment_max,

        s.positive_prob_mean,
        s.negative_prob_mean,
        s.neutral_prob_mean,

        s.sentiment_confidence_mean,

        s.positive_news_ratio,
        s.negative_news_ratio,
        s.neutral_news_ratio

    FROM price_features p

    LEFT JOIN daily_sentiment_observed s

        ON p.ticker = s.ticker

        AND CAST(p.date AS DATE) = s.date

    ORDER BY p.ticker, p.date
""")

print("\n=== VALIDATING DATASET ===")

summary = con.execute("""
    SELECT
        COUNT(*) AS trading_rows,
        COUNT(DISTINCT ticker) AS companies,
        SUM(has_news) AS days_with_news,
        SUM(1 - has_news) AS days_without_news,
        SUM(news_count) AS total_headlines,
        MIN(date) AS first_date,
        MAX(date) AS last_date
    FROM daily_sentiment
""").df()

print(summary.to_string(index=False))

assert int(summary["trading_rows"].iloc[0]) == total_prices

assert int(summary["total_headlines"].iloc[0]) == aligned

invalid_scores = con.execute("""
    SELECT COUNT(*)
    FROM daily_sentiment
    WHERE sentiment_mean NOT BETWEEN -1 AND 1
""").fetchone()[0]

assert invalid_scores == 0

invalid_missing = con.execute("""
    SELECT COUNT(*)
    FROM daily_sentiment
    WHERE
        (has_news = 0 AND sentiment_mean IS NOT NULL)
        OR
        (has_news = 1 AND sentiment_mean IS NULL)
""").fetchone()[0]

assert invalid_missing == 0

print("\n=== COVERAGE BY YEAR ===")

print(con.execute("""
    SELECT
        YEAR(date) AS year,
        COUNT(*) AS trading_rows,
        SUM(has_news) AS rows_with_news,
        ROUND(100.0 * AVG(has_news), 2)
            AS coverage_pct,
        SUM(news_count) AS headlines
    FROM daily_sentiment
    GROUP BY YEAR(date)
    ORDER BY year
""").df().to_string(index=False))

print("\n=== NVIDIA SAMPLE ===")

print(con.execute("""
    SELECT
        ticker,
        date,
        news_count,
        has_news,
        ROUND(sentiment_mean, 4) AS sentiment,
        ROUND(sentiment_confidence_mean, 4)
            AS confidence
    FROM daily_sentiment
    WHERE ticker = 'NVDA'
    ORDER BY date DESC
    LIMIT 15
""").df().to_string(index=False))

print("\n=== SAVING PARQUET ===")

con.execute(f"""
    COPY daily_sentiment
    TO '{OUTPUT_PATH}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

parquet_rows = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{OUTPUT_PATH}')
""").fetchone()[0]

assert parquet_rows == total_prices

print("Parquet rows:", f"{parquet_rows:,}")

print("File size:", round(
    Path(OUTPUT_PATH).stat().st_size / 1024**2,
    2
), "MiB")

con.close()

print("\n=== DAILY SENTIMENT: PASSED ===")
