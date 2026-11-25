import duckdb
import torch
import numpy as np
from chronos import Chronos2Pipeline

con = duckdb.connect(
    "data/quant_edge.duckdb",
    read_only=True
)

ticker = "NVDA"
origin = "2021-06-30"

df = con.execute("""
    SELECT date, return_5d
    FROM forecast_dataset
    WHERE ticker = ?
      AND date <= CAST(? AS DATE)
    ORDER BY date DESC
    LIMIT 60
""", [ticker, origin]).df()

df = df.sort_values("date")

assert len(df) == 60

values = df["return_5d"].to_numpy(
    dtype=np.float32
)

assert np.isfinite(values).all()

print("\n=== MODEL INPUT ===")
print("Company:", ticker)
print("First date:", df["date"].iloc[0])
print("Last date:", df["date"].iloc[-1])
print("Trading days:", len(values))

pipeline = Chronos2Pipeline.from_pretrained(
    "models/pretrained/chronos-2",
    device_map="cuda",
    dtype=torch.bfloat16,
    local_files_only=True
)

context = torch.tensor(values).reshape(1, 1, 60)

print("\nInput tensor shape:", context.shape)

with torch.inference_mode():
    quantiles, mean = pipeline.predict_quantiles(
        context,
        prediction_length=5,
        quantile_levels=[0.1, 0.5, 0.9]
    )

forecast = quantiles[0].detach().float().cpu()

print("Forecast shape:", forecast.shape)

assert forecast.shape == (1, 5, 3)
assert torch.isfinite(forecast).all()

lower, median, upper = forecast[0, -1].tolist()

actual, target_date = con.execute("""
    SELECT target_return_5d, target_date
    FROM forecast_dataset
    WHERE ticker = ?
      AND date = CAST(? AS DATE)
      AND dataset_split = 'validation'
""", [ticker, origin]).fetchone()

print("\n=== FIRST CHRONOS-2 FORECAST ===")
print("Company:", ticker)
print("Prediction date:", origin)
print("Target date:", target_date)

print(f"10th percentile: {lower*100:.4f}%")
print(f"Median prediction: {median*100:.4f}%")
print(f"90th percentile: {upper*100:.4f}%")

print(f"Actual return: {actual*100:.4f}%")
print(f"Absolute error: {abs(median-actual)*100:.4f} pp")

print(
    "Correct direction:",
    np.sign(median) == np.sign(actual)
)

con.close()

print("\n=== FIRST FORECAST: PASSED ===")
