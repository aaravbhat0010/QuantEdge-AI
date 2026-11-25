import requests
import json
import time
import os
from pathlib import Path

email = os.environ["SEC_EMAIL"]

session = requests.Session()
session.headers.update({
    "User-Agent": f"QuantEdge Research {email}",
    "Accept-Encoding": "gzip, deflate"
})

def fetch(url):
    for attempt in range(3):
        try:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))

tickers = Path("configs/tickers.txt").read_text().split()

print("=== DOWNLOADING SEC COMPANY DIRECTORY ===")

directory = fetch(
    "https://www.sec.gov/files/company_tickers.json"
)

Path("data/raw/sec/company_tickers.json").write_text(
    json.dumps(directory)
)

mapping = {
    item["ticker"]: item
    for item in directory.values()
}

failed = []
completed = []

print("\n=== DOWNLOADING COMPANY FUNDAMENTALS ===")

for ticker in tickers:
    try:
        company = mapping[ticker]
        cik = str(company["cik_str"]).zfill(10)

        path = Path(
            f"data/raw/sec/companyfacts/{ticker}.json"
        )

        if path.exists():
            data = json.loads(path.read_text())
            assert "facts" in data
        else:
            url = (
                "https://data.sec.gov/api/xbrl/"
                f"companyfacts/CIK{cik}.json"
            )

            data = fetch(url)
            assert "facts" in data

            path.write_text(json.dumps(data))

        metrics = len(
            data["facts"].get("us-gaap", {})
        )

        completed.append({
            "ticker": ticker,
            "cik": cik,
            "company": company["title"]
        })

        print(
            f"SUCCESS: {ticker:6} | "
            f"CIK: {cik} | "
            f"{metrics} accounting concepts"
        )

    except Exception as e:
        failed.append(ticker)
        print(f"FAILED: {ticker} | {e}")

    time.sleep(0.3)

Path("data/metadata/sec_companies.json").write_text(
    json.dumps(completed, indent=2)
)

print("\n=== FINAL SEC DOWNLOAD AUDIT ===")
print("Companies downloaded:", len(completed))
print("Failed companies:", failed)
