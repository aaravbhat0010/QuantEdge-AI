import duckdb
import numpy as np

con = duckdb.connect(
    "data/quant_edge.duckdb",
    read_only=True
)

df = con.execute("""
    SELECT *
    FROM master_dataset
    WHERE date < DATE '2021-01-01'
""").df()

features = [
    c for c in df.columns
    if c not in ["ticker", "date"]
]

assert len(features) == 59
assert "return_5d" in features

for name in features:
    assert not any(
        word in name.lower()
        for word in ["future", "target", "prediction"]
    )

print("\n=== ALL 59 FINANCIAL FEATURES ===\n")

print(
    f"{'FEATURE':<30}"
    f"{'COVERAGE':>12}"
    f"{'MIN':>16}"
    f"{'MAX':>16}"
)

for feature in features:

    values = df[feature].to_numpy(
        dtype=np.float64,
        na_value=np.nan
    )

    valid = values[np.isfinite(values)]

    coverage = len(valid) / len(values) * 100

    minimum = valid.min() if len(valid) else np.nan
    maximum = valid.max() if len(valid) else np.nan

    print(
        f"{feature:<30}"
        f"{coverage:>11.2f}%"
        f"{minimum:>16.4g}"
        f"{maximum:>16.4g}"
    )

print("\nTotal variables:", len(features))
print("Target series: return_5d")
print("Additional covariates:", len(features) - 1)
print("Companies:", df["ticker"].nunique())
print("Historical rows:", len(df))
print("Start:", df["date"].min())
print("End:", df["date"].max())

con.close()

print("\n=== FEATURE AUDIT COMPLETE ===")
