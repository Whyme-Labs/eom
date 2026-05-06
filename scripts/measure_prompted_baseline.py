"""Measure compile_with_repair(PromptedCompiler(OpenRouter, Gemma-4-31B)) on the gold seed.

Usage:
    set -a && source .env && set +a
    uv run python scripts/measure_prompted_baseline.py [--workers 4] [--max-attempts 3] [--limit N]

Output: data/eval/prompted_baseline.csv with one row per gold example, columns:
    doc_type, slug, n_blocks_gold, attempts, final_passed,
    h1_failures, h2_failures, ..., h12_failures, total_failures,
    elapsed_seconds, error
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eom.compilers.llm_client import OpenRouterClient
from eom.compilers.prompted import PromptedCompiler
from eom.harness import validate
from eom.normalise import normalise
from eom.repair import compile_with_repair
from eom.schema import EOMDocument

GOLD_DIR = Path("data/gold")
OUT = Path("data/eval/prompted_baseline.csv")
RULES = ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12"]


def gold_pairs() -> list[tuple[str, Path, Path]]:
    out = []
    for type_dir in sorted(p for p in GOLD_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            ej = type_dir / f"{md.stem}.eom.json"
            if ej.exists():
                out.append((type_dir.name, md, ej))
    return out


def measure_one(doc_type: str, md_path: Path, eom_path: Path, max_attempts: int) -> dict:
    """Run compile_with_repair on one (source, gold) pair; return a row dict."""
    slug = md_path.stem
    source = normalise(md_path.read_text(encoding="utf-8"))
    gold_eom = EOMDocument.model_validate_json(eom_path.read_text(encoding="utf-8"))
    n_blocks_gold = len(gold_eom.blocks)
    row: dict = {
        "doc_type": doc_type,
        "slug": slug,
        "n_blocks_gold": n_blocks_gold,
        "attempts": 0,
        "final_passed": False,
        "total_failures": 0,
        "elapsed_seconds": 0.0,
        "error": "",
    }
    for r in RULES:
        row[f"{r.lower()}_failures"] = 0
    t0 = time.time()
    try:
        client = OpenRouterClient()
        compiler = PromptedCompiler(client=client, few_shots=[])
        eom, attempts = compile_with_repair(
            compiler,
            source,
            hints={"document_type": doc_type, "render_profile": "executive_brief"},
            max_attempts=max_attempts,
        )
        report = validate(eom, source)
        row["attempts"] = attempts
        row["final_passed"] = report.passed
        row["total_failures"] = len(report.failures)
        for f in report.failures:
            key = f"{f.rule.lower()}_failures"
            if key in row:
                row[key] += 1
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["error_trace"] = traceback.format_exc()[-500:]
    row["elapsed_seconds"] = round(time.time() - t0, 2)
    return row


def existing_slugs() -> set[str]:
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {row["slug"] for row in csv.DictReader(f)}


def write_header_if_needed() -> None:
    if OUT.exists():
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = (
        ["doc_type", "slug", "n_blocks_gold", "attempts", "final_passed", "total_failures"]
        + [f"{r.lower()}_failures" for r in RULES]
        + ["elapsed_seconds", "error"]
    )
    with OUT.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=cols).writeheader()


def append_row(row: dict) -> None:
    with OUT.open() as f:
        cols = next(csv.reader(f))
    with OUT.open("a", newline="") as f:
        # Trim row to known cols (drop error_trace which is just for logging)
        out_row = {k: row.get(k, "") for k in cols}
        csv.DictWriter(f, fieldnames=cols).writerow(out_row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only measure the first N examples (after dedupe by existing CSV)",
    )
    args = p.parse_args()

    write_header_if_needed()
    done = existing_slugs()
    pairs = [(t, md, ej) for (t, md, ej) in gold_pairs() if md.stem not in done]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print(f"All {len(gold_pairs())} examples already measured.")
        return

    print(f"Measuring {len(pairs)} examples with {args.workers} workers…")
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(measure_one, t, md, ej, args.max_attempts): (t, md.stem)
            for (t, md, ej) in pairs
        }
        for fut in as_completed(futs):
            t, slug = futs[fut]
            row = fut.result()
            append_row(row)
            err = row.get("error") or ""
            tag = "PASS" if row["final_passed"] else f"FAIL({row['total_failures']})"
            extra = f" err={err[:60]}" if err else ""
            print(
                f"  [{row['attempts']}a {row['elapsed_seconds']:>5.1f}s] "
                f"{tag:<10} {t:>10}/{slug}{extra}"
            )
    print(f"\nWall: {time.time() - t_start:.1f}s. CSV at {OUT}")


if __name__ == "__main__":
    main()
