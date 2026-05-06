"""Scaffold a gold example from a raw source.

Runs the rules compiler to produce a starting-point EOM, which the human
then hand-corrects before committing.

Usage:
    uv run python scripts/scaffold_gold.py \\
        --input data/raw/freight_memo.md \\
        --doc-type memo \\
        --slug freight_memo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eom.compilers.rules import RulesCompiler
from eom.normalise import normalise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--doc-type", required=True,
                   choices=["memo", "report", "paper", "transcript",
                            "news", "policy", "other"])
    p.add_argument("--profile", default="executive_brief")
    p.add_argument("--slug", required=True)
    args = p.parse_args()

    src_text = Path(args.input).read_text(encoding="utf-8")
    norm = normalise(src_text)

    out_dir = Path("data/gold") / args.doc_type
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.slug}.md"
    eom_path = out_dir / f"{args.slug}.eom.json"
    md_path.write_text(norm, encoding="utf-8")

    compiler = RulesCompiler()
    eom = compiler.compile(norm, hints={
        "document_type": args.doc_type,
        "render_profile": args.profile,
    })
    eom_path.write_text(eom.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote scaffold:\n  source: {md_path}\n  eom:    {eom_path}")
    print("Hand-correct the EOM file, then run scripts/validate_gold.py.")


if __name__ == "__main__":
    main()
