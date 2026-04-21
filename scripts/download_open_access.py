#!/usr/bin/env python3
"""Download papers by DOI using Unpaywall, with optional Sci-Hub fallback."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

UNPAYWALL_ENDPOINT = "https://api.unpaywall.org/v2"
DEFAULT_TIMEOUT = 45
UVX_SCIHUB_CMD = [
    "uvx",
    "--from",
    "git+https://github.com/Oxidane-bot/scihub-cli.git",
    "scihub-cli",
]


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned.strip("._") or "paper"


def doi_to_path(outdir: Path, doi: str) -> Path:
    return outdir / f"{sanitize_filename(doi)}.pdf"


def read_dois(single_doi: str | None, doi_file: Path | None) -> list[str]:
    if single_doi:
        candidate = single_doi.strip()
        return [candidate] if candidate else []

    if not doi_file:
        return []

    dois: list[str] = []
    seen: set[str] = set()
    for raw_line in doi_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        doi = line.split()[0].strip()
        if doi and doi not in seen:
            seen.add(doi)
            dois.append(doi)
    return dois


def resolve_scihub_base_cmd(scihub_cmd: str | None) -> list[str]:
    if scihub_cmd:
        parsed = shlex.split(scihub_cmd)
        if parsed:
            return parsed
    if shutil.which("scihub-cli"):
        return ["scihub-cli"]
    return UVX_SCIHUB_CMD.copy()


def get_unpaywall_pdf_url(doi: str, email: str, timeout: int) -> str | None:
    endpoint = f"{UNPAYWALL_ENDPOINT}/{quote(doi, safe='')}"
    response = requests.get(endpoint, params={"email": email}, timeout=timeout)
    if response.status_code == 404:
        return None
    response.raise_for_status()

    payload = response.json()
    candidates: list[str] = []

    best = payload.get("best_oa_location") or {}
    for key in ("url_for_pdf", "url"):
        value = best.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)

    for location in payload.get("oa_locations") or []:
        if not isinstance(location, dict):
            continue
        for key in ("url_for_pdf", "url"):
            value = location.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)

    for url in candidates:
        if url.lower().startswith("http"):
            return url
    return None


def download_pdf(url: str, destination: Path, timeout: int, overwrite: bool) -> Path:
    if destination.exists() and not overwrite:
        return destination

    temp_path = destination.with_suffix(".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if chunk:
                    handle.write(chunk)

    with temp_path.open("rb") as handle:
        signature = handle.read(4)

    if signature != b"%PDF":
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file is not a valid PDF.")

    temp_path.replace(destination)
    return destination


def snapshot_pdfs(outdir: Path) -> dict[Path, float]:
    snapshot: dict[Path, float] = {}
    for path in outdir.rglob("*.pdf"):
        try:
            snapshot[path.resolve()] = path.stat().st_mtime
        except OSError:
            continue
    return snapshot


def find_new_pdf(before: dict[Path, float], outdir: Path) -> Path | None:
    newest: Path | None = None
    newest_time = -1.0
    for path in outdir.rglob("*.pdf"):
        try:
            resolved = path.resolve()
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if resolved not in before or mtime > before[resolved]:
            if mtime > newest_time:
                newest = path
                newest_time = mtime
    return newest


def try_scihub(doi: str, outdir: Path, base_cmd: list[str], timeout: int) -> tuple[bool, Path | None, str | None]:
    attempts = [
        base_cmd + ["download", "--doi", doi, "--outdir", str(outdir)],
        base_cmd + ["download", doi, "--outdir", str(outdir)],
        base_cmd + ["--doi", doi, "--outdir", str(outdir)],
        base_cmd + [doi, "--outdir", str(outdir)],
        base_cmd + [doi, "-o", str(outdir)],
    ]

    errors: list[str] = []
    for command in attempts:
        before = snapshot_pdfs(outdir)
        try:
            completed = subprocess.run(
                command,
                cwd=str(outdir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError:
            return False, None, f"Command not found: {' '.join(base_cmd)}"
        except subprocess.TimeoutExpired:
            errors.append(f"Timeout: {' '.join(command)}")
            continue
        except Exception as exc:  # pragma: no cover - runtime path.
            errors.append(f"{' '.join(command)} -> {exc}")
            continue

        if completed.returncode == 0:
            created = find_new_pdf(before, outdir)
            if created:
                return True, created, None
            guessed = doi_to_path(outdir, doi)
            if guessed.exists():
                return True, guessed, None
            return True, None, None

        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        error_text = stderr or stdout or f"exit={completed.returncode}"
        errors.append(f"{' '.join(command)} -> {error_text}")

    return False, None, " | ".join(errors) if errors else "Sci-Hub fallback failed."


def download_dois(
    dois: list[str],
    outdir: Path,
    unpaywall_email: str | None,
    fallback_mode: str,
    scihub_cmd: str | None,
    timeout: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    outdir.mkdir(parents=True, exist_ok=True)
    scihub_base_cmd = resolve_scihub_base_cmd(scihub_cmd)
    results: list[dict[str, Any]] = []

    for doi in dois:
        target_path = doi_to_path(outdir, doi)
        if target_path.exists() and not overwrite:
            results.append(
                {
                    "doi": doi,
                    "status": "exists",
                    "method": "existing",
                    "path": str(target_path),
                    "error": None,
                }
            )
            continue

        if fallback_mode == "always":
            order = ("scihub", "unpaywall")
        elif fallback_mode == "never":
            order = ("unpaywall",)
        else:
            order = ("unpaywall", "scihub")

        errors: list[str] = []
        success = False
        method: str | None = None
        saved_path: str | None = None

        for mode in order:
            if mode == "unpaywall":
                if not unpaywall_email:
                    errors.append("UNPAYWALL_EMAIL missing.")
                    continue
                try:
                    pdf_url = get_unpaywall_pdf_url(doi, unpaywall_email, timeout)
                    if not pdf_url:
                        errors.append("Unpaywall returned no OA PDF URL.")
                        continue
                    written = download_pdf(
                        url=pdf_url,
                        destination=target_path,
                        timeout=timeout,
                        overwrite=overwrite,
                    )
                    success = True
                    method = "unpaywall"
                    saved_path = str(written)
                    break
                except Exception as exc:  # pragma: no cover - network/runtime paths.
                    errors.append(f"Unpaywall error: {exc}")
                    continue

            if mode == "scihub":
                ok, written, error_text = try_scihub(
                    doi=doi,
                    outdir=outdir,
                    base_cmd=scihub_base_cmd,
                    timeout=timeout,
                )
                if ok:
                    success = True
                    method = "scihub"
                    saved_path = str(written) if written else None
                    if saved_path is None and target_path.exists():
                        saved_path = str(target_path)
                    break
                errors.append(error_text or "Sci-Hub fallback failed.")

        results.append(
            {
                "doi": doi,
                "status": "downloaded" if success else "failed",
                "method": method,
                "path": saved_path,
                "error": None if success else " | ".join(errors),
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PDFs by DOI using Unpaywall with optional Sci-Hub fallback."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--doi", help='Single DOI, e.g. "10.2307/2392994".')
    source.add_argument("--doi-file", type=Path, help="Path to text file containing one DOI per line.")

    parser.add_argument("--outdir", type=Path, default=Path("./downloads"), help="Download folder.")
    parser.add_argument(
        "--scihub-fallback",
        choices=("auto", "never", "always"),
        default="auto",
        help="Fallback policy after/before Unpaywall attempts.",
    )
    parser.add_argument("--scihub-cmd", help="Explicit scihub-cli command prefix.")
    parser.add_argument("--unpaywall-email", help="Override UNPAYWALL_EMAIL environment variable.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PDF files.")
    parser.add_argument("--report", type=Path, help="Optional JSON report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print("--timeout must be > 0", file=sys.stderr)
        return 2

    if args.doi_file and not args.doi_file.exists():
        print(f"DOI file not found: {args.doi_file}", file=sys.stderr)
        return 2

    dois = read_dois(args.doi, args.doi_file)
    if not dois:
        print("No DOI values found.", file=sys.stderr)
        return 2

    email = args.unpaywall_email or os.getenv("UNPAYWALL_EMAIL")
    results = download_dois(
        dois=dois,
        outdir=args.outdir,
        unpaywall_email=email,
        fallback_mode=args.scihub_fallback,
        scihub_cmd=args.scihub_cmd,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )

    downloaded_count = sum(1 for row in results if row["status"] == "downloaded")
    exists_count = sum(1 for row in results if row["status"] == "exists")
    report = {
        "attempted_doi_count": len(dois),
        "downloaded_count": downloaded_count,
        "already_available_count": exists_count,
        "results": results,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
