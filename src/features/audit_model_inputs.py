import duckdb

con = duckdb.connect(
    "data/quant_edge.duckdb",
    read_only=True
)

print("\n=== DATASET SPLITS ===")

print(con.execute("""
    SELECT
        dataset_split,
        COUNT(*) AS samples,
        COUNT(DISTINCT ticker) AS companies,
        MIN(date) AS first_date,
        MAX(date) AS last_date
    FROM forecast_dataset
    GROUP BY dataset_split
    ORDER BY first_date
""").df().to_string(index=False))

features = [
    row[0]
    for row in con.execute(
        "DESCRIBE master_dataset"
    ).fetchall()
    if row[0] not in ("ticker", "date")
]

print("\n=== TRAINING FEATURE QUALITY ===")

for feature in features:

    total, available, minimum, maximum = con.execute(f"""
        SELECT
            COUNT(*),
            COUNT("{feature}") FILTER (
                WHERE isfinite("{feature}")
            ),
            MIN("{feature}"),
            MAX("{feature}")
        FROM forecast_dataset
        WHERE dataset_split = 'train'
    """).fetchone()

    coverage = 100 * available / total

    print(
        f"{feature:32s} "
        f"coverage={coverage:6.2f}% "
        f"min={minimum} max={maximum}"
    )

print("\n=== TRAINING TARGET DISTRIBUTION ===")

print(con.execute("""
    SELECT
        target_direction,
        COUNT(*) AS samples,
        ROUND(
            100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
            2
        ) AS percentage
    FROM forecast_dataset
    WHERE dataset_split = 'train'
    GROUP BY target_direction
    ORDER BY target_direction
""").df().to_string(index=False))

print("\n=== PROHIBITED MODEL INPUTS ===")

for column in [
    "future_price_5d",
    "target_date",
    "target_return_5d",
    "target_direction",
    "dataset_split",
    "history_days"
]:
    assert column not in features
    print(column, "- excluded")

print("\nCandidate features:", len(features))

con.close()

print("\n=== MODEL INPUT AUDIT: PASSED ===")
