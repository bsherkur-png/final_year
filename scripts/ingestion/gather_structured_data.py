from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://www.searchapi.io/api/v1/search"
API_KEY = "mmsiig8LzqFpGqWH7UN6WRdu"
BASE_QUERY = '"shamima begum"'
SITES = [
    "bbc.co.uk",
    "theguardian.com",
    "independent.co.uk",
    "express.co.uk",
    "mirror.co.uk",
]
COUNTRY = "uk"
DATE_FROM = "10/03/2021"
DATE_TO = "01/04/2021"
PAGES = (1, 2, 3, 4, 5)
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
COLUMNS = ["title", "link", "source", "date", "snippet", "page", "site_filter"]


def fetch_page(session, api_key, query, page_number):
    params = {
        "engine": "google_news",
        "q": query,
        "ulle": COUNTRY,
        "time_period_min": DATE_FROM,
        "time_period_max": DATE_TO,
        "page": page_number,
        "num": "60",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    response = session.get(API_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def normalize_text(value):
    if value is None:
        return ""
    return str(value)


def extract_rows(results, page_number, site):
    rows = []
    for result in results:
        if not isinstance(result, dict):
            rows.append(["", "", "", "", "", page_number, site])
            continue
        rows.append(
            [
                normalize_text(result.get("title")),
                normalize_text(result.get("link")),
                normalize_text(result.get("source")),
                normalize_text(result.get("date")),
                normalize_text(result.get("snippet")),
                page_number,
                site,
            ]
        )
    return rows


def build_output_path():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return RAW_DATA_DIR / f"google_news_results_{timestamp}.csv"


def main():
    all_rows = []

    with requests.Session() as session:
        for site in SITES:
            query = f'{BASE_QUERY} site:{site}'
            print(f"Searching: {query}")

            for page_number in PAGES:
                try:
                    payload = fetch_page(session, API_KEY, query, page_number)
                except requests.RequestException as exc:
                    print(f"  Request failed for {site} page {page_number}: {exc}")
                    continue
                except ValueError as exc:
                    print(f"  Invalid JSON for {site} page {page_number}: {exc}")
                    continue

                organic_results = payload.get("organic_results", [])
                if not isinstance(organic_results, list):
                    print(f"  Unexpected format on {site} page {page_number}.")
                    continue

                all_rows.extend(extract_rows(organic_results, page_number, site))
                print(f"  {site} page {page_number}: {len(organic_results)} results")

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    output_path = build_output_path()
    df.to_csv(output_path, index=False)

    print(f"\nSaved {len(df)} rows to {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()
