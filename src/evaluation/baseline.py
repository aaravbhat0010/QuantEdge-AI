import duckdb
import numpy as np

con = duckdb.connect("data/quant_edge.duckdb")

print("\n=== CORRECTED PRICE AUDIT ===")

anomalies = con.execute("""
    SELECT ticker, date,
           ROUND(return_1d * 100, 2) AS return_pct
    FROM master_dataset
    WHERE ABS(return_1d) > 0.40
    ORDER BY ABS(return_1d) DESC
""").df()

print(anomalies.to_string(index=False))

print("\n=== VALIDATION BASELINES ===")

df = con.execute("""
    SELECT ticker, date, return_5d,
           target_return_5d
    FROM forecast_dataset
    WHERE dataset_split = 'validation'
    ORDER BY ticker, date
""").df()

actual = df["target_return_5d"].to_numpy()

assert np.isfinite(actual).all()
assert np.isfinite(df["return_5d"]).all()

baselines = {
    "Zero return": np.zeros(len(df)),
    "Momentum": df["return_5d"].to_numpy(),
    "Always positive": None
}

for name, predicted in baselines.items():

    print(f"\nModel: {name}")

    if predicted is None:
        accuracy = np.mean(actual > 0)
        print(f"Directional accuracy: {accuracy*100:.2f}%")
        continue

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))

    accuracy = np.mean(
        np.sign(actual) == np.sign(predicted)
    )

    print(f"MAE: {mae*100:.4f} percentage points")
    print(f"RMSE: {rmse*100:.4f} percentage points")
    print(f"Directional accuracy: {accuracy*100:.2f}%")

print("\nValidation samples:", len(df))

con.close()

print("\n=== BASELINE EVALUATION: PASSED ===")
