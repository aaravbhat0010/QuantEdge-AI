import time
import argparse
from pathlib import Path

import duckdb
import numpy as np
import torch
import bitsandbytes as bnb

from chronos import Chronos2Pipeline
from chronos.chronos2.trainer import Chronos2Trainer
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=10)
args = parser.parse_args()

set_seed(42)

CONTEXT = 60
HORIZON = 5
BATCH_SIZE = 128

output = Path(
    "models/adapters/chronos2_qlora_v9"
    if args.steps == 1000
    else "experiments/qlora_smoke_test"
)

assert not (output / "final").exists()

print("\n=== LOADING TRAINING DATA ===")

con = duckdb.connect(
    "data/quant_edge.duckdb",
    read_only=True
)

df = con.execute("""
    SELECT ticker, date, return_5d
    FROM forecast_dataset
    WHERE date < DATE '2021-01-01'
    ORDER BY ticker, date
""").df()

con.close()

inputs = []

for ticker, group in df.groupby("ticker"):
    target = group["return_5d"].to_numpy(
        dtype=np.float32
    )
    target = target[np.isfinite(target)]

    assert len(target) >= CONTEXT + HORIZON

    inputs.append({"target": target})

print("Companies:", len(inputs))
print("Training observations:",
      sum(len(x["target"]) for x in inputs))

print("\n=== LOADING 4-BIT MODEL ===")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipeline = Chronos2Pipeline.from_pretrained(
    "models/pretrained/chronos-2",
    device_map={"": 0},
    quantization_config=quant_config,
    dtype=torch.bfloat16,
    local_files_only=True
)

model = prepare_model_for_kbit_training(
    pipeline.model,
    use_gradient_checkpointing=False
)

model = get_peft_model(
    model,
    LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "k", "v", "o"],
        bias="none"
    )
)

model.print_trainable_parameters()

trainable = [
    (n, p) for n, p in model.named_parameters()
    if p.requires_grad
]

assert trainable
assert all("lora_" in n for n, p in trainable)

print("Adapter dtypes:",
      {str(p.dtype) for n, p in trainable})

# Reuse the dataset classes from Chronos's fit function.
fit_globals = Chronos2Pipeline.fit.__globals__

Chronos2Dataset = fit_globals["Chronos2Dataset"]
DatasetMode = fit_globals["DatasetMode"]

dataset = Chronos2Dataset(
    inputs=inputs,
    context_length=CONTEXT,
    prediction_length=HORIZON,
    batch_size=BATCH_SIZE,
    output_patch_size=pipeline.model_output_patch_size,
    min_past=CONTEXT,
    mode=DatasetMode.TRAIN
)

training_args = TrainingArguments(
    output_dir=str(output),
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=1e-4,
    max_steps=args.steps,
    lr_scheduler_type="linear",
    warmup_ratio=0.0,
    optim="adamw_torch",
    bf16=True,
    tf32=True,
    logging_steps=1 if args.steps == 10 else 10,
    save_strategy="no",
    eval_strategy="no",
    report_to=[],
    dataloader_num_workers=0,
    gradient_accumulation_steps=1,
    prediction_loss_only=True,
    seed=42
)

training_args._n_gpu = 1

trainer = Chronos2Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Keep a copy of adapter weights for verification.
before = {
    n: p.detach().cpu().clone()
    for n, p in trainable
}

torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

print("\n=== STARTING QLORA TRAINING ===")

result = trainer.train()

torch.cuda.synchronize()

assert np.isfinite(result.training_loss)

changed = sum(
    not torch.equal(before[n], p.detach().cpu())
    for n, p in trainable
)

assert changed > 0

quantized_layers = [
    m for m in model.modules()
    if isinstance(m, bnb.nn.Linear4bit)
]

assert len(quantized_layers) == 125

assert all(
    m.weight.dtype == torch.uint8
    and not m.weight.requires_grad
    and m.weight.quant_state is not None
    for m in quantized_layers
)

model.save_pretrained(
    output / "final",
    safe_serialization=True
)

print("\n=== QLORA RESULTS ===")
print("Training steps:", trainer.state.global_step)
print("Training loss:", result.training_loss)
print("Updated adapter tensors:", changed)
print("Frozen 4-bit layers:", len(quantized_layers))
print("Training time:", round(
    time.perf_counter() - start, 2), "seconds")
print("Peak allocated GPU memory:", round(
    torch.cuda.max_memory_allocated() / 1024**3, 3
), "GiB")
print("Saved adapters:", output / "final")

print("\n=== QLORA TRAINING PASSED ===")
