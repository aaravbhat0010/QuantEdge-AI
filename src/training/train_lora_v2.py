import time
from pathlib import Path

import duckdb
import numpy as np
import torch

from chronos import Chronos2Pipeline
from peft import LoraConfig
from transformers import set_seed

set_seed(42)

CONTEXT = 60
HORIZON = 5
STEPS = 1000

FEATURES = [
    # Technical indicators
    "return_1d",
    "return_20d",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "macd_ratio",
    "rsi_14",
    "volatility_20",
    "relative_volume",

    # Financial sentiment
    "news_count",
    "has_news",
    "sentiment_mean",
    "positive_prob_mean",
    "negative_prob_mean",
    "sentiment_confidence_mean",

    # Company fundamentals
    "net_profit_margin",
    "operating_margin",
    "liabilities_to_assets",
    "equity_to_assets",
    "revenue_age_days",

    # Macroeconomics
    "federal_funds_rate",
    "consumer_price_index",
    "unemployment_rate",
    "treasury_yield_spread",
    "volatility_index",
    "crude_oil_price",
    "consumer_sentiment"
]

print("\n=== PREPARING TRAINING DATA ===")

con = duckdb.connect(
    "data/quant_edge.duckdb",
    read_only=True
)

columns = ", ".join(FEATURES)

df = con.execute(f"""
    SELECT ticker, date, return_5d, {columns}
    FROM forecast_dataset
    WHERE date < DATE '2021-01-01'
    ORDER BY ticker, date
""").df()

con.close()

assert df["date"].max().year == 2020
assert df["ticker"].nunique() == 50

inputs = []

for ticker, group in df.groupby("ticker"):

    group = group.sort_values("date")

    group = group[
        np.isfinite(group["return_5d"])
    ]

    target = group["return_5d"].to_numpy(
        dtype=np.float32
    )

    covariates = group[FEATURES].to_numpy(
        dtype=np.float32
    )

    covariates[~np.isfinite(covariates)] = np.nan

    assert len(target) >= CONTEXT + HORIZON
    assert np.isfinite(target).all()

    inputs.append({
        "target": target,
        "past_covariates": {
            name: covariates[:, j].copy()
            for j, name in enumerate(FEATURES)
        }
    })

print("Companies:", len(inputs))
print("Historical database rows:", len(df))
print("Usable target observations:",
      sum(len(item["target"]) for item in inputs))
print("Training start:", df["date"].min())
print("Training end:", df["date"].max())
print("Covariates:", len(FEATURES))
print("Optimization steps:", STEPS)

print("\n=== LOADING PRETRAINED MODEL ===")

assert torch.cuda.is_available()
assert torch.cuda.is_bf16_supported()

pipeline = Chronos2Pipeline.from_pretrained(
    "models/pretrained/chronos-2",
    device_map="cuda",
    dtype=torch.float32,
    local_files_only=True
)

total_params = sum(
    p.numel()
    for p in pipeline.model.parameters()
)

print("Original parameters:", total_params)

# Identify attention layers for LoRA

linear_names = [
    name
    for name, module in pipeline.model.named_modules()
    if isinstance(module, torch.nn.Linear)
]

candidates = [
    "q", "k", "v", "o",
    "q_proj", "k_proj", "v_proj", "o_proj"
]

target_modules = sorted({
    name.split(".")[-1]
    for name in linear_names
    if name.split(".")[-1] in candidates
})

print("LoRA target modules:", target_modules)

if not target_modules:
    raise RuntimeError(
        f"No matching attention layers: {linear_names}"
    )

# Configure LoRA adapters

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=target_modules,
    bias="none"
)

output = Path("models/adapters/chronos2_lora_v2")

if output.exists():
    raise FileExistsError(
        f"{output} already exists. "
        "Preserve it before rerunning."
    )

print("\n=== STARTING LORA TRAINING ===")

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

start = time.perf_counter()

trained_pipeline = pipeline.fit(
    inputs=inputs,
    prediction_length=HORIZON,
    finetune_mode="lora",
    lora_config=lora_config,
    context_length=CONTEXT,
    min_past=CONTEXT,
    learning_rate=1e-4,
    num_steps=STEPS,
    batch_size=128,
    output_dir=output,
    finetuned_ckpt_name="final",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch",
    bf16=True,
    report_to=[],
    seed=42
)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print("\n=== TRAINING PERFORMANCE ===")

print(f"Training time: {elapsed:.2f} seconds")

print(
    "Peak allocated GPU memory:",
    f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB"
)

print("\n=== SAVED MODEL FILES ===")

for path in output.rglob("*"):
    if path.is_file():
        print(path)

print("\n=== LORA TRAINING: PASSED ===")
