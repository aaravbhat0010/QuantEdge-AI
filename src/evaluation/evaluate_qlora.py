import bitsandbytes as bnb
from peft import PeftModel
from transformers import BitsAndBytesConfig
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
BATCH_SIZE = 128

con = duckdb.connect("data/quant_edge.duckdb")

print("\n=== PREPARING VALIDATION WINDOWS ===")

df = con.execute("""
    SELECT ticker, date, return_5d,
           target_return_5d, target_date,
           dataset_split
    FROM forecast_dataset
    WHERE date < DATE '2022-01-01'
    ORDER BY ticker, date
""").df()

windows = []
records = []

for ticker, group in df.groupby("ticker"):

    group = group.reset_index(drop=True)

    values = group["return_5d"].to_numpy(
        dtype=np.float32
    )

    for i in range(CONTEXT - 1, len(group)):

        row = group.iloc[i]

        if row["dataset_split"] != "validation":
            continue

        window = values[i-CONTEXT+1:i+1]

        assert len(window) == CONTEXT
        assert np.isfinite(window).all()

        windows.append(window)

        records.append({
            "ticker": ticker,
            "date": row["date"],
            "target_date": row["target_date"],
            "actual_return": row["target_return_5d"],
            "momentum_return": row["return_5d"]
        })

inputs = np.stack(windows)

assert len(inputs) == 12350

print("Total windows:", len(inputs))
print("Input shape:", inputs.shape)

print("\n=== LOADING CHRONOS-2 ===")

pipeline = Chronos2Pipeline.from_pretrained(
    "models/pretrained/chronos-2",
    device_map={"": 0},
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    dtype=torch.bfloat16,
    local_files_only=True
)
pipeline.model = PeftModel.from_pretrained(
    pipeline.model,
    "models/adapters/chronos2_qlora_v9/final",
    is_trainable=False,
    local_files_only=True
)
pipeline.model.eval()

assert sum(
    isinstance(m, bnb.nn.Linear4bit)
    for m in pipeline.model.modules()
) == 125

print("Quantized model and trained adapters loaded")


pipeline.model.eval()

torch.cuda.reset_peak_memory_stats()

predictions = []
start = time.perf_counter()

print("\n=== BATCH INFERENCE ===")

with torch.inference_mode():

    for i in tqdm(
        range(0, len(inputs), BATCH_SIZE)
    ):

        batch = torch.from_numpy(
            inputs[i:i+BATCH_SIZE]
        ).unsqueeze(1)

        quantiles, _ = pipeline.predict_quantiles(
            batch,
            prediction_length=HORIZON,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=BATCH_SIZE
        )

        assert len(quantiles) == len(batch)

        for forecast in quantiles:

            assert forecast.shape == (1, 5, 3)

            predictions.append(
                forecast[0, -1]
                .detach().float().cpu().numpy()
            )

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

predictions = np.stack(predictions)

assert len(predictions) == len(records)
assert np.isfinite(predictions).all()

results = pd.DataFrame(records)

results["q10"] = predictions[:, 0]
results["q50"] = predictions[:, 1]
results["q90"] = predictions[:, 2]

assert (results["q10"] <= results["q50"]).all()
assert (results["q50"] <= results["q90"]).all()

actual = results["actual_return"].to_numpy()
predicted = results["q50"].to_numpy()

print("\n=== VALIDATION RESULTS ===")

for name, pred in {
    "Zero return": np.zeros(len(actual)),
    "Momentum": results["momentum_return"].to_numpy(),
    "QLoRA V9": predicted
}.items():

    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))

    accuracy = np.mean(
        np.sign(actual) == np.sign(pred)
    )

    print(f"\n{name}")
    print(f"MAE: {mae*100:.4f} pp")
    print(f"RMSE: {rmse*100:.4f} pp")
    print(f"Directional accuracy: {accuracy*100:.2f}%")

coverage = np.mean(
    (actual >= results["q10"]) &
    (actual <= results["q90"])
)

print("\nAlways-positive accuracy:",
      f"{np.mean(actual > 0)*100:.2f}%")

print("80% prediction interval coverage:",
      f"{coverage*100:.2f}%")

print("\n=== GPU PERFORMANCE ===")

print(f"Inference time: {elapsed:.2f} seconds")

print(f"Predictions per second: "
      f"{len(results)/elapsed:.2f}")

print(f"Peak allocated GPU memory: "
      f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")

print("\n=== SAVING PREDICTIONS ===")

output = Path("experiments/predictions")
output.mkdir(parents=True, exist_ok=True)

results.to_parquet(
    output / "chronos2_qlora_v9_validation.parquet",
    index=False
)

con.register("prediction_results", results)

con.execute("""
    CREATE OR REPLACE TABLE
    chronos2_qlora_v9_validation AS
    SELECT * FROM prediction_results
""")

con.close()

print("\n=== LORA V8 EVALUATION: PASSED ===")
