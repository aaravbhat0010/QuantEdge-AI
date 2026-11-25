FROM nvcr.io/nvidia/pytorch:25.08-py3

WORKDIR /workspace/quant-edge-ai

RUN python -m pip install --no-cache-dir \
    yfinance \
    duckdb \
    pyarrow \
    pandas \
    scikit-learn \
    streamlit \
    plotly \
    requests \
    tqdm \
    huggingface_hub

ENV HF_HOME=/workspace/quant-edge-ai/models/huggingface
ENV PYTHONUNBUFFERED=1
