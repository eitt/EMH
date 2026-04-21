#!/usr/bin/env python3
"""Search Scopus and return normalized paper metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests

SCOPUS_ENDPOINT = "https://api.elsevier.com/content/search/scopus"
MAX_PAGE_SIZE = 25
DEFAULT_TIMEOUT = 30


def parse_year(value: str | None) -> int | None:
    """Extract year from Scopus date strings."""
    if not value:
        return None
    prefix = value[:4]
    if prefix.isdigit():
        return int(prefix)
    return None


def normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Map Scopus entry to a stable schema."""
    cover_date = entry.get("prism:coverDate") or entry.get("prism:coverDisplayDate")
    cited_by_raw = entry.get("citedby-count")
    try:
        cited_by = int(cited_by_raw) if cited_by_raw is not None else 0
    except (TypeError, ValueError):
        cited_by = 0

    doi = (entry.get("prism:doi") or "").strip() or None
    title = (entry.get("dc:title") or "").strip() or None
    source = (entry.get("prism:publicationName") or "").strip() or None
    url = (entry.get("prism:url") or "").strip() or None
    eid = (entry.get("eid") or "").strip() or None

    return {
        "title": title,
        "doi": doi,
        "year": parse_year(cover_date),
        "cover_date": cover_date,
        "source": source,
        "cited_by": cited_by,
        "eid": eid,
        "url": url,
    }


def build_query(
    keywords: str | None,
    raw_query: str | None,
    from_year: int | None,
) -> str:
    """Build Scopus query from keywords or raw query and optional year filter."""
    has_keywords = bool(keywords and keywords.strip())
    has_raw_query = bool(raw_query and raw_query.strip())

    if has_keywords == has_raw_query:
        raise ValueError("Provide exactly one of --keywords or --query.")

    if has_raw_query:
        query = raw_query.strip()
    else:
        escaped = keywords.strip().replace('"', '\\"')
        query = f'TITLE-ABS-KEY("{escaped}")'

    if from_year is not None:
        query = f"({query}) AND PUBYEAR > {from_year - 1}"

    return query


def run_search(
    api_key: str,
    query: str,
    count: int,
    sort: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[int, list[dict[str, Any]]]:
    """Fetch up to `count` records from Scopus."""
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }

    records: list[dict[str, Any]] = []
    total_hits = 0
    start = 0

    while len(records) < count:
        page_size = min(MAX_PAGE_SIZE, count - len(records))
        params = {
            "query": query,
            "count": page_size,
            "start": start,
            "sort": sort,
        }
        response = requests.get(
            SCOPUS_ENDPOINT,
            headers=headers,
            params=params,
            timeout=timeout,
        )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text[:300]
            raise RuntimeError(
                f"Scopus request failed ({response.status_code}): {detail}"
            ) from exc

        payload = response.json()
        search_results = payload.get("search-results", {})
        try:
            total_hits = int(search_results.get("opensearch:totalResults", 0) or 0)
        except (TypeError, ValueError):
            total_hits = 0

        entries = search_results.get("entry") or []
        if not entries:
            break

        for entry in entries:
            records.append(normalize_entry(entry))
            if len(records) >= count:
                break

        start += len(entries)
        if start >= total_hits:
            break

    return total_hits, records[:count]


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Write normalized records as CSV."""
    fields = ["title", "doi", "year", "cover_date", "source", "cited_by", "eid", "url"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Scopus and return paper metadata."
    )
    parser.add_argument("--keywords", help='Keywords query, e.g. "pedestrian simulation".')
    parser.add_argument("--query", help="Raw Scopus query string.")
    parser.add_argument("--count", type=int, default=20, help="Number of results to fetch.")
    parser.add_argument("--sort", default="-citedby-count", help="Scopus sort expression.")
    parser.add_argument("--from-year", type=int, help="Lower publication year bound (inclusive).")
    parser.add_argument("--out", type=Path, help="Optional output path.")
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format when --out is used.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        print("--count must be > 0", file=sys.stderr)
        return 2

    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key:
        print("Missing ELSEVIER_API_KEY environment variable.", file=sys.stderr)
        return 2

    try:
        query = build_query(args.keywords, args.query, args.from_year)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        total_hits, records = run_search(
            api_key=api_key,
            query=query,
            count=args.count,
            sort=args.sort,
            timeout=args.timeout,
        )
    except Exception as exc:  # pragma: no cover - network/runtime paths.
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    report = {
        "query": query,
        "sort": args.sort,
        "from_year": args.from_year,
        "total_hits": total_hits,
        "returned": len(records),
        "records": records,
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "csv":
            write_csv(args.out, records)
        else:
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
