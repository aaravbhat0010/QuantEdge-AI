import time
from pathlib import Path

import torch
import duckdb
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

BATCH_SIZE = 256
MODEL_PATH = "models/sentiment/finbert"
OUTPUT_PATH = "data/processed/news_sentiment.parquet"

start_time = time.perf_counter()

assert torch.cuda.is_available(), "GPU unavailable"

print("\n=== LOADING LOCAL FINBERT ===")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = model.to(
    device="cuda",
    dtype=torch.bfloat16
)

model.eval()

print("GPU:", torch.cuda.get_device_name(0))
print("Precision:", next(model.parameters()).dtype)
print("Batch size:", BATCH_SIZE)

# Retrieve all financial headlines
con = duckdb.connect("data/quant_edge.duckdb")

news = con.execute("""
    SELECT *
    FROM financial_news_combined
    ORDER BY ticker, published_at, headline
""").df()

total = len(news)

assert total > 0, "No headlines found"
assert news["headline"].notna().all()

print("\n=== DATASET ===")
print("Headlines:", f"{total:,}")
print("Companies:", news["ticker"].nunique())

# Identify sentiment label positions
label_map = {
    label.lower(): int(index)
    for index, label in model.config.id2label.items()
}

assert set(label_map) == {
    "positive", "negative", "neutral"
}

all_probabilities = []

torch.cuda.synchronize()
inference_start = time.perf_counter()

print("\n=== STARTING BATCH INFERENCE ===")

with torch.inference_mode():

    for start in tqdm(
        range(0, total, BATCH_SIZE),
        desc="Processing batches",
        unit="batch"
    ):

        headlines = news["headline"].iloc[
            start:start + BATCH_SIZE
        ].tolist()

        inputs = tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to("cuda")

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits.float(),
            dim=-1
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

torch.cuda.synchronize()

inference_time = time.perf_counter() - inference_start

# Combine results from all batches
probabilities = np.concatenate(
    all_probabilities,
    axis=0
)

# Validate outputs
assert len(probabilities) == total
assert probabilities.shape == (total, 3)
assert np.isfinite(probabilities).all()

assert np.allclose(
    probabilities.sum(axis=1),
    1.0,
    atol=1e-6
)

# Create numerical sentiment features
news["positive_probability"] = probabilities[
    :, label_map["positive"]
]

news["negative_probability"] = probabilities[
    :, label_map["negative"]
]

news["neutral_probability"] = probabilities[
    :, label_map["neutral"]
]

news["sentiment_score"] = (
    news["positive_probability"]
    - news["negative_probability"]
)

news["confidence"] = probabilities.max(axis=1)

news["predicted_sentiment"] = [
    model.config.id2label[int(i)].lower()
    for i in probabilities.argmax(axis=1)
]

assert news["sentiment_score"].between(-1, 1).all()

# Save results into DuckDB
print("\n=== SAVING RESULTS ===")

con.register("sentiment_df", news)

con.execute("""
    CREATE OR REPLACE TABLE news_sentiment AS
    SELECT * FROM sentiment_df
""")

# Export to Parquet
Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

con.execute(f"""
    COPY news_sentiment
    TO '{OUTPUT_PATH}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# Verify saved records
saved_count = con.execute("""
    SELECT COUNT(*) FROM news_sentiment
""").fetchone()[0]

parquet_count = con.execute(f"""
    SELECT COUNT(*)
    FROM read_parquet('{OUTPUT_PATH}')
""").fetchone()[0]

assert saved_count == total
assert parquet_count == total

print("\n=== SENTIMENT DISTRIBUTION ===")

print(con.execute("""
    SELECT
        predicted_sentiment,
        COUNT(*) AS headlines,
        ROUND(
            100.0 * COUNT(*) /
            SUM(COUNT(*)) OVER (),
            2
        ) AS percentage,
        ROUND(AVG(confidence), 4)
            AS average_confidence
    FROM news_sentiment
    GROUP BY predicted_sentiment
    ORDER BY headlines DESC
""").df().to_string(index=False))

print("\n=== SAMPLE RESULTS ===")

print(con.execute("""
    SELECT
        ticker,
        published_at,
        predicted_sentiment,
        ROUND(sentiment_score, 4) AS score,
        ROUND(confidence, 4) AS confidence
    FROM news_sentiment
    ORDER BY published_at DESC
    LIMIT 10
""").df().to_string(index=False))

con.close()

print("\n=== PERFORMANCE REPORT ===")
print("Headlines processed:", f"{total:,}")
print("Inference seconds:", round(inference_time, 2))
print("Headlines per second:", round(total / inference_time, 2))
print("Total runtime:", round(time.perf_counter() - start_time, 2), "seconds")
print("Peak allocated GPU memory:", round(
    torch.cuda.max_memory_allocated() / 1024**3, 2
), "GiB")

print("DuckDB records:", f"{saved_count:,}")
print("Parquet records:", f"{parquet_count:,}")
print("Parquet file size:", round(
    Path(OUTPUT_PATH).stat().st_size / 1024**2, 2
), "MiB")

print("\n=== BATCH INFERENCE: PASSED ===")
