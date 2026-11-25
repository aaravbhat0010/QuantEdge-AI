import torch
from chronos import Chronos2Pipeline
from pathlib import Path
from huggingface_hub import snapshot_download

print("\n=== GPU CHECK ===")

assert torch.cuda.is_available()

print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

model_path = Path("models/pretrained/chronos-2")

print("\n=== DOWNLOADING CHRONOS-2 ===")

snapshot_download(
    repo_id="amazon/chronos-2",
    local_dir=str(model_path)
)

print("\n=== LOADING LOCAL MODEL ===")

pipeline = Chronos2Pipeline.from_pretrained(
    str(model_path),
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    local_files_only=True
)

model = pipeline.model

parameters = sum(
    p.numel() for p in model.parameters()
)

print("Parameters:", f"{parameters:,}")
print("Model device:", next(model.parameters()).device)
print("Model dtype:", next(model.parameters()).dtype)
print("Model location:", model_path)

assert next(model.parameters()).is_cuda

print("\n=== CHRONOS-2 SETUP: PASSED ===")
