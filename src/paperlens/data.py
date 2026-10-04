"""Download paper metadata from the public arXiv API."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
DEFAULT_CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "cs.AI"]
REQUEST_DELAY_S = 3.0  # arXiv asks API users to wait ~3 seconds between requests


def _clean(text: str) -> str:
    """Collapse all runs of whitespace into single spaces."""
    return " ".join(text.split())


def parse_entry(entry: ET.Element) -> dict:
    """Convert one Atom <entry> element into a flat dictionary."""

    def text(tag: str) -> str:
        node = entry.find(tag, NS)
        return _clean(node.text) if node is not None and node.text else ""

    raw_id = text("atom:id")
    primary = entry.find("arxiv:primary_category", NS)
    return {
        "arxiv_id": raw_id.rsplit("/abs/", 1)[-1],
        "title": text("atom:title"),
        "abstract": text("atom:summary"),
        "authors": [
            _clean(a.findtext("atom:name", default="", namespaces=NS))
            for a in entry.findall("atom:author", NS)
        ],
        "published": text("atom:published"),
        "primary_category": primary.get("term", "") if primary is not None else "",
        "url": raw_id,
    }


def parse_feed(xml_text: str) -> list[dict]:
    """Parse a full arXiv API response into a list of paper dictionaries."""
    root = ET.fromstring(xml_text)
    return [parse_entry(e) for e in root.findall("atom:entry", NS)]


def fetch_page(
    session: requests.Session,
    category: str,
    start: int,
    max_results: int,
    retries: int = 3,
) -> list[dict]:
    """Fetch one page of results for a category, retrying on transient errors."""
    params = {
        "search_query": f"cat:{category}",
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            return parse_feed(response.text)
        except (requests.RequestException, ET.ParseError) as error:
            last_error = error
            time.sleep(REQUEST_DELAY_S * attempt * 2)
    raise RuntimeError(f"arXiv request failed after {retries} attempts") from last_error


def build_dataset(
    categories: list[str],
    per_category: int,
    page_size: int = 100,
    out_path: Path = Path("data/papers.parquet"),
) -> pd.DataFrame:
    """Download papers for each category, de-duplicate, and save as Parquet."""
    session = requests.Session()
    session.headers["User-Agent"] = "paperlens/0.1 (student research project)"

    records: dict[str, dict] = {}
    for category in categories:
        for start in tqdm(range(0, per_category, page_size), desc=category):
            n = min(page_size, per_category - start)
            page = fetch_page(session, category, start, n)
            if not page:
                break
            for record in page:
                records.setdefault(record["arxiv_id"], record)
            time.sleep(REQUEST_DELAY_S)

    df = pd.DataFrame(list(records.values()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df
