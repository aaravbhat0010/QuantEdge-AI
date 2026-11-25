# QuantEdge AI - Benchmark Results

## Held-Out Test Performance

24,750 forecast windows per model, 2022-2023.

| Model | MAE (pp) | RMSE (pp) | Direction | Coverage |
|---|---:|---:|---:|---:|
| Zero Return | 3.3572 | 4.6474 | 0.07% | N/A |
| Momentum | 4.7912 | 6.5633 | 51.53% | N/A |
| Chronos-2 | 4.2114 | 5.7964 | 50.74% | 72.20% |
| LoRA V7 | 3.5460 | 4.8751 | 48.69% | 77.33% |
| LoRA V8 | 3.5193 | 4.8408 | 48.94% | 76.57% |
| QLoRA V9 | 3.5207 | 4.8433 | 48.91% | 76.97% |

## Hardware Efficiency

| Metric | LoRA V8 | QLoRA V9 |
|---|---:|---:|
| Trainable parameters | 1179648 | 1179648 |
| Peak training allocation (GiB) | 1.78 | 0.776 |
| Peak inference allocation (GiB) | 0.26 | 0.10 |
| Training time (s) | 114.60 | 141.01 |
| Test inference time (s) | 4.71 | 9.40 |

Memory represents PyTorch peak allocated GPU memory.
LoRA V8 uses merged BF16 inference while QLoRA V9
uses an NF4 base with attached adapters.

## Findings

- LoRA V8 reduced pretrained model RMSE by 16.49%.
- QLoRA reduced peak training allocation by 56.4%.
- QLoRA reduced peak inference allocation by 61.5%.
- Quantization reduced memory but increased execution time.
- The strongest naive baselines remained superior.
