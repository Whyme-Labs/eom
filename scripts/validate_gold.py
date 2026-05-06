"""Validate every (source, eom) pair under data/gold/ against the harness.

Updates data/gold/MANIFEST.json with the inventory.
Exits non-zero if any pair fails the harness.

Usage:
    uv run python scripts/validate_gold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from eom.harness import validate
from eom.normalise import normalise
from eom.schema import EOMDocument

GOLD_DIR = Path("data/gold")


def main() -> int:
    examples = []
    failures_total = 0
    for type_dir in sorted(p for p in GOLD_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            slug = md.stem
            eom_path = type_dir / f"{slug}.eom.json"
            if not eom_path.exists():
                print(f"  ! {md} has no matching .eom.json")
                failures_total += 1
                continue
            source = normalise(md.read_text(encoding="utf-8"))
            try:
                eom = EOMDocument.model_validate_json(eom_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  ! {eom_path} invalid schema: {e}")
                failures_total += 1
                continue
            report = validate(eom, source)
            if report.passed:
                print(f"  PASS  {type_dir.name}/{slug}")
                examples.append({
                    "doc_type": type_dir.name,
                    "slug": slug,
                    "source": str(md),
                    "eom": str(eom_path),
                    "n_blocks": int(report.metrics["n_blocks"]),
                })
            else:
                print(f"  FAIL  {type_dir.name}/{slug} ({len(report.failures)} failures)")
                for f in report.failures[:5]:
                    tag = f" [{f.block_id}]" if f.block_id else ""
                    print(f"      {f.rule}{tag}: {f.message}")
                failures_total += 1

    manifest = {
        "version": "0.1",
        "examples": examples,
    }
    Path(GOLD_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(examples)} pass, {failures_total} fail.")
    return 0 if failures_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
