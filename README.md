# QuantEdge AI

Foundation Model Fine-Tuning, Quantization and Edge AI Engineering

**Author:** Aarav Bhat

**Hardware:** NVIDIA Jetson AGX Thor

## Overview

An end-to-end financial AI research framework exploring
parameter-efficient fine-tuning, quantization, GPU optimization
and local deployment of pretrained foundation models.

## Technical Highlights

- Amazon Chronos-2: 119.48M parameters
- FinBERT: 110M pretrained parameters
- LoRA ranks: 8, 16 and 32
- QLoRA with 4-bit NF4 double quantization
- BF16 mixed-precision training
- PyTorch, CUDA, PEFT and BitsAndBytes
- DuckDB, Parquet, Docker and Streamlit

## Dataset

- Eight years of real financial data (2016-2023)
- 50 publicly traded companies
- 100,550 stock-day observations
- 255,369 financial headlines
- 43,420 fundamental observations
- 24,293 macroeconomic observations
- 59 engineered financial features

## Experimental Results

| Model | Test MAE | Test RMSE | Directional Accuracy |
|---|---:|---:|---:|
| Zero-return baseline | 3.3572 | 4.6474 | 0.07% |
| Momentum baseline | 4.7912 | 6.5633 | 51.53% |
| Pretrained Chronos-2 | 4.2114 | 5.7964 | 50.74% |
| LoRA V7 | 3.5460 | 4.8751 | 48.69% |
| LoRA V8 | 3.5193 | 4.8408 | 48.94% |
| QLoRA V9 | 3.5207 | 4.8433 | 48.91% |

MAE and RMSE are measured in percentage points.

## Engineering Achievements

- 16.49% RMSE improvement over pretrained Chronos-2.
- 56.4% reduction in peak training GPU allocation.
- 61.5% reduction in peak inference GPU allocation.
- 3,185 financial headlines processed per second.
- Nine fine-tuning configurations.
- 24,750 held-out forecast windows per final model.

## Evaluation Methodology

Training: 2016-2020

Validation: 2021

Testing: 2022-2023

Context: 60 trading observations

Forecast horizon: 5 trading days

## Limitations

The adapted models did not outperform the strongest naive
forecasting baselines. Pretraining overlap remains unverified.
The project demonstrates AI engineering feasibility rather
than proven investment profitability.

## Acknowledgments

OpenAI ChatGPT assisted with technical guidance, code
development, debugging and manuscript drafting.

This is independent technical research, not a peer-reviewed
publication or financial advice.

Datasets, model weights and credentials are excluded.

## Technical Documentation

- [Experimental benchmarks](docs/benchmarks.md)
- [Hardware and software environment](docs/environment.md)

## System Architecture

    Historical Financial Data
              |
    Data Ingestion and FinBERT
              |
    DuckDB + Parquet
              |
    59 Engineered Features
              |
    Chronos-2 Foundation Model
         /           \
       LoRA         QLoRA
         \           /
    Chronological Backtesting
              |
    Streamlit Dashboard

## Technology Stack

| Layer | Technology |
|---|---|
| Hardware | NVIDIA Jetson AGX Thor |
| Deep Learning | PyTorch, CUDA |
| Foundation Model | Amazon Chronos-2 |
| NLP | FinBERT |
| Fine-Tuning | Hugging Face PEFT |
| Quantization | BitsAndBytes NF4 |
| Storage | DuckDB, Parquet |
| Deployment | NVIDIA Docker |
| Frontend | Streamlit, Plotly |

## Docker Setup

    sudo docker build -t quant-edge-ai:v1 .
    sudo docker build -f Dockerfile.ai -t quant-edge-ai:ai .
    sudo docker build -f Dockerfile.chronos -t quant-edge-ai:chronos .
    sudo docker build -f Dockerfile.qlora -t quant-edge-ai:qlora .

The dashboard requires locally prepared datasets and
model checkpoints.

## References

- [Chronos-2](https://huggingface.co/amazon/chronos-2)
- [FinBERT](https://huggingface.co/ProsusAI/finbert)
- [LoRA](https://arxiv.org/abs/2106.09685)
- [QLoRA](https://arxiv.org/abs/2305.14314)
- [FNSPID](https://arxiv.org/abs/2402.06698)
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing)
