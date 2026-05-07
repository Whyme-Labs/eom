"""Run PromptedCompiler over data/raw/, save harness-passing pairs to data/synthetic/.

For each raw .md, runs compile_with_repair(PromptedCompiler(OpenRouter, few_shots=3 random gold),
source, hints, max_attempts=3). Saves passing pairs as data/synthetic/<doc_type>/<slug>.eom.json
plus a copy of the source as data/synthetic/<doc_type>/<slug>.md (post-normalise).

Logs per-source outcome (slug, attempts, passed/failed, elapsed) to data/synthetic/log.jsonl.

Usage:
    set -a && source .env && set +a
    uv run python scripts/generate_synthetic.py [--workers 6] [--max-attempts 3] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eom.compilers.llm_client import OpenRouterClient
from eom.compilers.prompted import PromptedCompiler
from eom.harness import validate
from eom.normalise import normalise
from eom.repair import compile_with_repair
from eom.schema import EOMDocument

RAW_DIR = Path("data/raw")
SYNTH_DIR = Path("data/synthetic")
LOG = SYNTH_DIR / "log.jsonl"
GOLD_DIR = Path("data/gold")


def load_gold_few_shots() -> list[tuple[str, EOMDocument]]:
    """Load all gold (source, eom) pairs as candidate few-shots."""
    pairs = []
    for type_dir in sorted(p for p in GOLD_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            ej = type_dir / f"{md.stem}.eom.json"
            if ej.exists():
                src = normalise(md.read_text(encoding="utf-8"))
                eom = EOMDocument.model_validate_json(ej.read_text(encoding="utf-8"))
                pairs.append((src, eom))
    return pairs


def raw_pairs() -> list[tuple[str, Path]]:
    """List all raw markdown files. Returns (doc_type, path)."""
    out = []
    for type_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            out.append((type_dir.name, md))
    return out


def existing_synth_slugs() -> set[str]:
    if not SYNTH_DIR.exists():
        return set()
    return {
        p.stem
        for type_dir in SYNTH_DIR.iterdir()
        if type_dir.is_dir()
        for p in type_dir.glob("*.eom.json")
    }


def generate_one(
    doc_type: str,
    md_path: Path,
    few_shot_pool: list[tuple[str, EOMDocument]],
    max_attempts: int,
) -> dict:
    slug = md_path.stem
    source = normalise(md_path.read_text(encoding="utf-8"))
    t0 = time.time()
    rec: dict = {
        "slug": slug,
        "doc_type": doc_type,
        "passed": False,
        "attempts": 0,
        "elapsed_s": 0.0,
        "error": "",
    }
    try:
        # Random 3 few-shots, prefer same doc_type
        same_type = [(s, e) for s, e in few_shot_pool if e.document_type == doc_type]
        others = [(s, e) for s, e in few_shot_pool if e.document_type != doc_type]
        random.shuffle(same_type)
        random.shuffle(others)
        few_shots = (same_type + others)[:3]

        client = OpenRouterClient()
        compiler = PromptedCompiler(client=client, few_shots=few_shots)
        eom, attempts = compile_with_repair(
            compiler,
            source,
            hints={"document_type": doc_type, "render_profile": "executive_brief"},
            max_attempts=max_attempts,
        )
        rec["attempts"] = attempts
        report = validate(eom, source)
        rec["passed"] = report.passed
        if report.passed:
            out_dir = SYNTH_DIR / doc_type
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{slug}.md").write_text(source, encoding="utf-8")
            (out_dir / f"{slug}.eom.json").write_text(
                eom.model_dump_json(indent=2), encoding="utf-8"
            )
        else:
            rec["error"] = f"harness failures: {[f.rule for f in report.failures[:5]]}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["elapsed_s"] = round(time.time() - t0, 2)
    return rec


def build_quality_report(log_path: Path) -> dict:
    """Analyse log.jsonl and return a quality report dict."""
    records = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"error": "no log records found"}

    passing = [r for r in records if r["passed"]]
    failing = [r for r in records if not r["passed"]]

    # Per doc_type pass rate
    by_type: dict[str, dict] = {}
    for r in records:
        dt = r["doc_type"]
        if dt not in by_type:
            by_type[dt] = {"pass": 0, "fail": 0}
        if r["passed"]:
            by_type[dt]["pass"] += 1
        else:
            by_type[dt]["fail"] += 1

    per_type_rate = {}
    for dt, counts in sorted(by_type.items()):
        total = counts["pass"] + counts["fail"]
        per_type_rate[dt] = {
            "pass": counts["pass"],
            "fail": counts["fail"],
            "total": total,
            "yield_pct": round(100.0 * counts["pass"] / total, 1) if total else 0.0,
        }

    # Failure mode analysis
    failure_modes: dict[str, int] = {}
    for r in failing:
        err = r.get("error", "")
        # Extract harness rule names
        rules = re.findall(r"H\d+", err)
        if rules:
            for rule in rules:
                failure_modes[rule] = failure_modes.get(rule, 0) + 1
        elif err:
            key = err.split(":")[0][:40]
            failure_modes[key] = failure_modes.get(key, 0) + 1

    # Sample passing and failing
    sample_pass = [
        {"slug": r["slug"], "doc_type": r["doc_type"], "attempts": r["attempts"],
         "elapsed_s": r["elapsed_s"]}
        for r in random.sample(passing, min(3, len(passing)))
    ] if passing else []
    sample_fail = [
        {"slug": r["slug"], "doc_type": r["doc_type"], "error": r.get("error", ""),
         "attempts": r["attempts"], "elapsed_s": r["elapsed_s"]}
        for r in random.sample(failing, min(3, len(failing)))
    ] if failing else []

    n_total = len(records)
    n_pass = len(passing)
    n_fail = len(failing)
    yield_pct = round(100.0 * n_pass / n_total, 1) if n_total else 0.0

    mean_attempts_pass = (
        round(sum(r["attempts"] for r in passing) / len(passing), 2) if passing else None
    )
    mean_attempts_fail = (
        round(sum(r["attempts"] for r in failing) / len(failing), 2) if failing else None
    )

    # Error rate (exceptions, not harness failures)
    n_errors = sum(
        1 for r in records
        if r.get("error", "") and not r.get("error", "").startswith("harness failures")
    )

    return {
        "total_attempted": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "yield_pct": yield_pct,
        "error_rate_pct": round(100.0 * n_errors / n_total, 1) if n_total else 0.0,
        "mean_attempts_passing": mean_attempts_pass,
        "mean_attempts_failing": mean_attempts_fail,
        "per_doc_type": per_type_rate,
        "top_failure_modes": dict(
            sorted(failure_modes.items(), key=lambda x: -x[1])[:10]
        ),
        "sample_passing": sample_pass,
        "sample_failing": sample_fail,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    done = existing_synth_slugs()
    pairs = [(t, path) for (t, path) in raw_pairs() if path.stem not in done]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("Nothing to do.")
        return

    few_shot_pool = load_gold_few_shots()
    print(f"Few-shot pool: {len(few_shot_pool)} gold examples")
    print(f"Generating from {len(pairs)} raw docs ({args.workers} workers)…")

    t_start = time.time()
    n_pass = 0
    n_fail = 0
    with LOG.open("a") as logf, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(generate_one, t, path, few_shot_pool, args.max_attempts): (t, path.stem)
            for (t, path) in pairs
        }
        for fut in as_completed(futs):
            t, slug = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {
                    "slug": slug,
                    "doc_type": t,
                    "passed": False,
                    "attempts": 0,
                    "elapsed_s": 0.0,
                    "error": f"FutureError: {e}",
                }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if rec["passed"]:
                n_pass += 1
            else:
                n_fail += 1
            tag = "PASS" if rec["passed"] else "FAIL"
            err = f" err={rec['error'][:60]}" if rec["error"] else ""
            print(
                f"  [{rec['attempts']}a {rec['elapsed_s']:>5.1f}s] "
                f"{tag:<5} {t:>10}/{slug}{err}"
            )

    wall = time.time() - t_start
    total = n_pass + n_fail
    yield_pct = n_pass / total if total else 0.0
    print(
        f"\nWall: {wall:.1f}s. Pass: {n_pass}, Fail: {n_fail}, "
        f"Yield: {yield_pct:.0%}"
    )

    # Build and save quality report
    report = build_quality_report(LOG)
    report_path = SYNTH_DIR / "quality_report_first_batch.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Quality report saved to {report_path}")

    # Guardrail checks
    if total > 0 and yield_pct < 0.30:
        print(
            f"\nBLOCKED: yield {yield_pct:.0%} < 30% stop-and-audit threshold. "
            "Surface to controller before proceeding."
        )
    elif total > 0:
        error_rate = report.get("error_rate_pct", 0.0)
        if error_rate > 10.0:
            print(
                f"\nBLOCKED: error rate {error_rate:.1f}% > 10%. "
                "Likely API/network problems. Surface to controller."
            )
        else:
            print(f"\nPipeline complete. Yield: {yield_pct:.0%}")


if __name__ == "__main__":
    main()
