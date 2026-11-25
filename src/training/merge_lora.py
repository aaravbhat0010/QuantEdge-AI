import time
from pathlib import Path

import torch
from chronos import Chronos2Pipeline
from peft import PeftModel

BASE = Path("models/pretrained/chronos-2")
ADAPTER = Path("models/adapters/chronos2_lora_v1/final")
OUTPUT = Path("models/checkpoints/chronos2_merged_v1")

assert BASE.exists()
assert ADAPTER.exists()
assert torch.cuda.is_available()

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists."
    )

start = time.perf_counter()

print("\n=== LOADING ORIGINAL CHRONOS-2 ===")

pipeline = Chronos2Pipeline.from_pretrained(
    str(BASE),
    device_map="cuda",
    dtype=torch.float32,
    local_files_only=True
)

base_params = sum(
    p.numel() for p in pipeline.model.parameters()
)

print("Original parameters:", base_params)

print("\n=== ATTACHING TRAINED LORA ADAPTERS ===")

adapted_model = PeftModel.from_pretrained(
    pipeline.model,
    str(ADAPTER),
    is_trainable=False,
    local_files_only=True
)

adapter_params = sum(
    p.numel()
    for name, p in adapted_model.named_parameters()
    if "lora_" in name
)

print("LoRA parameters:", adapter_params)

print(
    "Adapter-to-base parameter ratio:",
    f"{100 * adapter_params / base_params:.4f}%"
)

print("\n=== MERGING WEIGHTS ===")

merged_model = adapted_model.merge_and_unload(
    safe_merge=True
)

merged_params = sum(
    p.numel() for p in merged_model.parameters()
)

assert merged_params == base_params

assert not any(
    "lora_" in name
    for name, _ in merged_model.named_parameters()
)

assert all(
    torch.isfinite(p).all().item()
    for p in merged_model.parameters()
)

print("Merged parameters:", merged_params)
print("Model precision:", next(
    merged_model.parameters()
).dtype)

print("\n=== SAVING MERGED MODEL ===")

merged_model.save_pretrained(
    str(OUTPUT),
    safe_serialization=True
)

print("\n=== VERIFYING SAVED MODEL ===")

reloaded = Chronos2Pipeline.from_pretrained(
    str(OUTPUT),
    device_map="cuda",
    dtype=torch.float32,
    local_files_only=True
)

assert sum(
    p.numel() for p in reloaded.model.parameters()
) == base_params

for path in OUTPUT.rglob("*"):
    if path.is_file():
        print(
            path,
            f"{path.stat().st_size / 1024**2:.2f} MB"
        )

print("\nTotal time:",
      f"{time.perf_counter() - start:.2f} seconds")

print("Original model preserved:", BASE.exists())
print("LoRA adapters preserved:", ADAPTER.exists())

print("\n=== MODEL MERGING: PASSED ===")
