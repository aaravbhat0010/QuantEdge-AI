import torch
import duckdb
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

model_id = "ProsusAI/finbert"
local_path = Path("models/sentiment/finbert")

assert torch.cuda.is_available(), "GPU unavailable"

print("\n=== DOWNLOADING FINBERT ===")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSequenceClassification.from_pretrained(
    model_id
)

local_path.mkdir(parents=True, exist_ok=True)

tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path, safe_serialization=True)

print("Model saved to:", local_path)

# Move model to Thor GPU using BF16
model = model.to(device="cuda", dtype=torch.bfloat16)
model.eval()

print("\n=== MODEL AUDIT ===")
print("Architecture:", model.config.model_type)
print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")
print("Device:", next(model.parameters()).device)
print("Precision:", next(model.parameters()).dtype)
print("Labels:", model.config.id2label)

# Retrieve five real financial headlines
con = duckdb.connect(
    "data/quant_edge.duckdb",
    read_only=True
)

news = con.execute("""
SELECT ticker, published_at, headline
FROM financial_news_combined
WHERE ticker = 'NVDA'
ORDER BY published_at DESC, headline
LIMIT 5
""").df()

con.close()

assert len(news) == 5, "Expected five headlines"

# Convert text into numerical tokens
inputs = tokenizer(
    news["headline"].tolist(),
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
).to("cuda")

# Forward propagation and Softmax
with torch.inference_mode():
    outputs = model(**inputs)
    logits = outputs.logits.float().cpu()
    probabilities = torch.softmax(logits, dim=-1)

# Validate numerical outputs
assert torch.isfinite(probabilities).all().item()

assert torch.allclose(
    probabilities.sum(dim=-1),
    torch.ones(len(news)),
    atol=1e-6
)

print("\n=== REAL FINANCIAL NEWS INFERENCE ===")

for i, row in news.iterrows():

    scores = {
        model.config.id2label[j].lower():
        float(probabilities[i, j])
        for j in range(model.config.num_labels)
    }

    label = max(scores, key=scores.get)

    sentiment_score = (
        scores["positive"] - scores["negative"]
    )

    print("\n" + "=" * 60)
    print("Company:", row["ticker"])
    print("Date:", row["published_at"])
    print("Headline:", row["headline"])

    print("\nRaw logits:", logits[i].tolist())

    print("\nSoftmax probabilities:")

    for name, score in scores.items():
        print(f"  {name}: {score * 100:.2f}%")

    print("\nPredicted sentiment:", label)
    print("Confidence:", round(scores[label] * 100, 2), "%")
    print("Sentiment score:", round(sentiment_score, 4))

print("\n=== FINBERT INFERENCE: PASSED ===")
