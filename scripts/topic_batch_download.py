#!/usr/bin/env python3
"""End-to-end topic search + DOI download pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Make sibling script imports stable when run as `python scripts/topic_batch_download.py`.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from download_open_access import download_dois  # noqa: E402
from search_scopus import build_query, run_search  # noqa: E402

QUANTITY_TARGETS = {
    "few": 5,
    "batch": 20,
    "max": 100,
}
DEFAULT_TIMEOUT = 45


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def resolve_from_year(latest: bool, years_back: int, explicit_from_year: int | None) -> int | None:
    if explicit_from_year is not None:
        return explicit_from_year
    if latest:
        current_year = datetime.now().year
        return current_year - years_back + 1
    return None


def resolve_sort(latest: bool, explicit_sort: str | None) -> str:
    if explicit_sort and explicit_sort.strip():
        return explicit_sort.strip()
    if latest:
        return "-coverDate"
    return "-citedby-count"


def resolve_target(quantity_mode: str, explicit_target: int | None) -> int:
    if explicit_target is not None:
        return explicit_target
    return QUANTITY_TARGETS[quantity_mode]


def resolve_scan_count(target: int, max_scan: int) -> int:
    suggested = max(20, target * 3)
    bounded = min(suggested, max_scan) if max_scan > 0 else suggested
    return max(target, bounded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Scopus by topic and download candidate PDFs by DOI."
    )
    query_source = parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--keywords", help='Keywords, e.g. "pedestrian simulation".')
    query_source.add_argument("--query", help="Raw Scopus query string.")

    parser.add_argument(
        "--quantity-mode",
        choices=("few", "batch", "max"),
        default="batch",
        help="Target amount policy when --target is not provided.",
    )
    parser.add_argument("--target", type=int, help="Explicit DOI target count.")
    parser.add_argument("--latest", action="store_true", help="Prioritize recent papers.")
    parser.add_argument(
        "--years-back",
        type=int,
        default=3,
        help="When --latest and no --from-year, keep this many recent years.",
    )
    parser.add_argument("--from-year", type=int, help="Explicit lower year bound (inclusive).")
    parser.add_argument("--sort", help="Explicit Scopus sort expression override.")
    parser.add_argument("--max-scan", type=int, default=200, help="Maximum Scopus rows to scan.")
    parser.add_argument("--outdir", type=Path, default=Path("./downloads"), help="Download directory.")
    parser.add_argument(
        "--scihub-fallback",
        choices=("auto", "never", "always"),
        default="auto",
        help="Fallback policy for DOI download.",
    )
    parser.add_argument("--scihub-cmd", help="Explicit scihub-cli command prefix.")
    parser.add_argument("--unpaywall-email", help="Override UNPAYWALL_EMAIL environment variable.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP/subprocess timeout.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PDF files.")
    parser.add_argument("--report", type=Path, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.target is not None and args.target <= 0:
        print("--target must be > 0", file=sys.stderr)
        return 2
    if args.years_back <= 0:
        print("--years-back must be > 0", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be > 0", file=sys.stderr)
        return 2

    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key:
        print("Missing ELSEVIER_API_KEY environment variable.", file=sys.stderr)
        return 2

    from_year = resolve_from_year(args.latest, args.years_back, args.from_year)
    target = resolve_target(args.quantity_mode, args.target)
    sort = resolve_sort(args.latest, args.sort)
    scan_count = resolve_scan_count(target, args.max_scan)

    try:
        query = build_query(args.keywords, args.query, from_year)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    try:
        total_hits, records = run_search(
            api_key=api_key,
            query=query,
            count=scan_count,
            sort=sort,
            timeout=args.timeout,
        )
    except Exception as exc:  # pragma: no cover - network/runtime paths.
        print(f"Scopus search failed: {exc}", file=sys.stderr)
        return 1

    scanned_entries = len(records)
    candidate_dois = dedupe([row["doi"] for row in records if row.get("doi")])
    selected_dois = candidate_dois[:target]

    email = args.unpaywall_email or os.getenv("UNPAYWALL_EMAIL")
    download_results = download_dois(
        dois=selected_dois,
        outdir=args.outdir,
        unpaywall_email=email,
        fallback_mode=args.scihub_fallback,
        scihub_cmd=args.scihub_cmd,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )

    downloaded_count = sum(1 for row in download_results if row["status"] == "downloaded")
    report = {
        "query": query,
        "sort": sort,
        "from_year": from_year,
        "total_hits": total_hits,
        "scanned_entries": scanned_entries,
        "candidate_doi_count": len(candidate_dois),
        "attempted_doi_count": len(selected_dois),
        "downloaded_count": downloaded_count,
        "results": download_results,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
