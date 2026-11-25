import requests
import json
import os
import time
from pathlib import Path

session = requests.Session()

session.headers.update({
    "User-Agent": f"QuantEdge Research {os.environ['SEC_EMAIL']}",
    "Accept-Encoding": "gzip, deflate"
})

historical = {
    "XOM": "0000034088",
    "DIS": "0001001039",
    "LIN": "0000884905"
}

for ticker, cik in historical.items():
    try:
        url = (
            "https://data.sec.gov/api/xbrl/"
            f"companyfacts/CIK{cik}.json"
        )

        response = session.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        path = Path(
            f"data/raw/sec/historical/{ticker}_{cik}.json"
        )

        path.write_text(json.dumps(data))

        facts = data["facts"].get("us-gaap", {})

        dates = [
            item["filed"]
            for concept in facts.values()
            for observations in concept["units"].values()
            for item in observations
            if item.get("filed")
            and "2015-01-01" <= item["filed"] < "2024-01-01"
        ]

        print(f"\nTicker: {ticker}")
        print("Company:", data.get("entityName"))
        print("CIK:", cik)
        print("Accounting concepts:", len(facts))
        print("Historical observations:", len(dates))
        print("First filing:", min(dates, default=None))
        print("Last filing:", max(dates, default=None))

    except Exception as e:
        print(f"FAILED: {ticker} | {e}")

    time.sleep(0.5)
