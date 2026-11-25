import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from chronos import Chronos2Pipeline

CONTEXT = 60
HORIZON = 5
BATCH_SIZE = 256
TASK_BATCH = 64

FEATURES = ['open',
 'high',
 'low',
 'close',
 'adjusted_close',
 'volume',
 'return_1d',
 'return_20d',
 'sma_20_ratio',
 'sma_50_ratio',
 'ema_12_ratio',
 'ema_26_ratio',
 'macd_ratio',
 'rsi_14',
 'volatility_20',
 'relative_volume',
 'news_count',
 'has_news',
 'sentiment_mean',
 'sentiment_std',
 'sentiment_min',
 'sentiment_max',
 'positive_prob_mean',
 'negative_prob_mean',
 'neutral_prob_mean',
 'sentiment_confidence_mean',
 'positive_news_ratio',
 'negative_news_ratio',
 'neutral_news_ratio',
 'revenue',
 'net_income',
 'eps',
 'assets',
 'liabilities',
 'equity',
 'operating_cash_flow',
 'gross_profit',
 'operating_income',
 'net_profit_margin',
 'gross_profit_margin',
 'operating_margin',
 'liabilities_to_assets',
 'equity_to_assets',
 'revenue_age_days',
 'has_fundamentals',
 'federal_funds_rate',
 'consumer_price_index',
 'unemployment_rate',
 'real_gdp',
 'treasury_10_year',
 'treasury_2_year',
 'volatility_index',
 'industrial_production',
 'consumer_sentiment',
 'crude_oil_price',
 'money_supply',
 'retail_sales',
 'treasury_yield_spread']

con = duckdb.connect("data/quant_edge.duckdb")

print("\n=== PREPARING FINANCIAL INPUTS ===")

# Verify database columns before inference
available = {
    row[0] for row in con.execute(
        "DESCRIBE forecast_dataset"
    ).fetchall()
}

missing = set(FEATURES) - available

if missing:
    raise ValueError(
        f"Missing database columns: {sorted(missing)}"
    )

columns = ", ".join(FEATURES)

df = con.execute(f"""
    SELECT ticker, date, return_5d,
           target_return_5d, target_date,
           dataset_split, {columns}
    FROM forecast_dataset
    WHERE date < DATE '2022-01-01'
    ORDER BY ticker, date
""").df()

inputs = []
records = []

for ticker, group in df.groupby("ticker"):

    group = group.reset_index(drop=True)

    target = group["return_5d"].to_numpy(
        dtype=np.float32
    )

    covariates = group[FEATURES].to_numpy(
        dtype=np.float32
    )

    covariates[~np.isfinite(covariates)] = np.nan

    for i in range(CONTEXT - 1, len(group)):

        row = group.iloc[i]

        if row["dataset_split"] != "validation":
            continue

        start = i - CONTEXT + 1
        end = i + 1

        history = target[start:end]

        assert len(history) == CONTEXT
        assert np.isfinite(history).all()
        assert row["target_date"] > row["date"]

        inputs.append({
            "target": history.copy(),
            "past_covariates": {
                feature: covariates[
                    start:end, j
                ].copy()
                for j, feature in enumerate(FEATURES)
            }
        })

        records.append({
            "ticker": ticker,
            "date": row["date"],
            "target_date": row["target_date"],
            "actual_return": row["target_return_5d"]
        })

assert len(inputs) == 12350

print("Prediction windows:", len(inputs))
print("Historical covariates:", len(FEATURES))
print("Total variables:", len(FEATURES) + 1)
print("Context length:", CONTEXT)
print("Future covariates: None")

print("\n=== LOADING CHRONOS-2 ===")

pipeline = Chronos2Pipeline.from_pretrained(
    "models/checkpoints/chronos2_merged_v5",
    device_map="cuda",
    dtype=torch.bfloat16,
    local_files_only=True
)

pipeline.model.eval()

def predict(batch):

    with torch.inference_mode():

        quantiles, _ = pipeline.predict_quantiles(
            batch,
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
            context_length=CONTEXT,
            batch_size=BATCH_SIZE,
            cross_learning=False
        )

    assert len(quantiles) == len(batch)

    predictions = []

    for forecast in quantiles:

        assert forecast.shape == (1, 5, 3)

        predictions.append(
            forecast[0, -1]
            .detach().float().cpu().numpy()
        )

    result = np.stack(predictions)

    assert np.isfinite(result).all()

    return result

print("\n=== MULTIVARIATE SMOKE TEST ===")

sample = predict(inputs[:4])

print("Output shape:", sample.shape)
print("Smoke test passed")

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()
predictions = []

print("\n=== MULTIVARIATE INFERENCE ===")

for i in tqdm(
    range(0, len(inputs), TASK_BATCH)
):

    batch = inputs[i:i+TASK_BATCH]

    predictions.append(predict(batch))

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

predictions = np.concatenate(predictions)

results = pd.DataFrame(records)

assert len(predictions) == len(results)

results["q10"] = predictions[:, 0]
results["q50"] = predictions[:, 1]
results["q90"] = predictions[:, 2]

assert (results["q10"] <= results["q50"]).all()
assert (results["q50"] <= results["q90"]).all()

actual = results["actual_return"].to_numpy()
predicted = results["q50"].to_numpy()

mae = np.mean(np.abs(actual - predicted))

rmse = np.sqrt(
    np.mean((actual - predicted)**2)
)

accuracy = np.mean(
    np.sign(actual) == np.sign(predicted)
)

coverage = np.mean(
    (actual >= results["q10"]) &
    (actual <= results["q90"])
)

print("\n=== MULTIVARIATE VALIDATION RESULTS ===")

print(f"MAE: {mae*100:.4f} pp")
print(f"RMSE: {rmse*100:.4f} pp")
print(f"Directional accuracy: {accuracy*100:.2f}%")
print(f"80% interval coverage: {coverage*100:.2f}%")

print("\n=== COMPARISON WITH PREVIOUS MODEL ===")

print("Univariate accuracy: 49.76%")
print(f"Multivariate accuracy: {accuracy*100:.2f}%")
print(f"Accuracy change: {accuracy*100-49.76:+.2f} pp")
print("Always-positive baseline: 56.63%")

print("\n=== GPU PERFORMANCE ===")

print(f"Inference time: {elapsed:.2f} seconds")
print(f"Predictions/sec: {len(results)/elapsed:.2f}")

print(
    "Peak allocated GPU memory:",
    f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB"
)

output = Path("experiments/predictions")
output.mkdir(parents=True, exist_ok=True)

results.to_parquet(
    output / "chronos2_lora_v5_validation.parquet",
    index=False
)

con.register("prediction_results", results)

con.execute("""
    CREATE OR REPLACE TABLE
    chronos2_lora_v5_validation AS
    SELECT * FROM prediction_results
""")

con.close()

print("\n=== MULTIVARIATE EVALUATION: PASSED ===")
