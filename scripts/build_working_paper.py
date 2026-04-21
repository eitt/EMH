#!/usr/bin/env python3
"""Package Overleaf-ready working-paper bundle with article/prose checks."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd().resolve()
TEX_MAIN = ROOT / "tex" / "main.tex"
OUT = ROOT / "output" / "working_paper"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_article_style(main_tex: str) -> list[str]:
    checks: list[str] = []
    checks.append("ok" if "\\documentclass[11pt]{article}" in main_tex else "missing_article_class")
    checks.append("ok" if "\\usepackage[authoryear]{natbib}" in main_tex else "missing_authoryear_natbib")
    checks.append("ok" if "\\begin{abstract}" in main_tex else "missing_abstract")
    return checks


def check_theory_citations(intro_text: str, lit_text: str) -> list[str]:
    required = ["fama1970", "fama1991"]
    merged = intro_text + "\n" + lit_text
    found = []
    for key in required:
        found.append(f"ok:{key}" if key in merged else f"missing:{key}")
    return found


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    if not TEX_MAIN.exists():
        raise FileNotFoundError(f"Missing manuscript entrypoint: {TEX_MAIN}")

    OUT.mkdir(parents=True, exist_ok=True)
    bundle = OUT / "bundle"

    # Copy manuscript and assets for Overleaf upload.
    copy_tree(ROOT / "tex", bundle / "tex")
    copy_tree(ROOT / "tables" / "generated", bundle / "tables" / "generated")
    copy_tree(ROOT / "figures" / "generated", bundle / "figures" / "generated")
    (bundle / "paper").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "paper" / "references.bib", bundle / "paper" / "references.bib")

    intro = read_text(ROOT / "tex" / "sections" / "01_introduction.tex")
    lit = read_text(ROOT / "tex" / "sections" / "02_literature_gap.tex")
    main_tex = read_text(TEX_MAIN)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "article_style_checks": check_article_style(main_tex),
        "theory_citation_checks": check_theory_citations(intro, lit),
        "entrypoint": "tex/main.tex",
        "notes": [
            "Bundle is Overleaf-ready with relative paths.",
            "Run pdflatex/bibtex on tex/main.tex after upload.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Built working-paper bundle at: {bundle}")
    print(f"Wrote manifest: {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
