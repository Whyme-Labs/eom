"""Evaluate val-generations.jsonl against the EOM harness.

Reads {prompt, generated} pairs, finds the matching source markdown by inspecting
the prompt, parses the generated JSON, runs validate(eom, source). Reports pass
rate + per-rule failure breakdown.

Usage:
    uv run python scripts/eval_generations.py runs/sft/modal-distill/val-generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from eom.harness import validate
from eom.normalise import normalise
from eom.schema import EOMDocument


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)
# When the input was the distill template the source text is bracketed by <<<>>>.
_SRC_RE = re.compile(r"<<<\n(.*?)\n>>>", re.DOTALL)


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = _FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def evaluate(path: Path) -> dict:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    n = len(rows)
    n_pass = 0
    n_invalid_json = 0
    n_invalid_schema = 0
    rule_fails: Counter[str] = Counter()
    per_doc: list[dict] = []

    for i, row in enumerate(rows):
        m = _SRC_RE.search(row["prompt"])
        if not m:
            print(f"  ! [{i}] couldn't extract source from prompt")
            continue
        source = normalise(m.group(1))

        body = _strip_fences(row["generated"])
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            n_invalid_json += 1
            per_doc.append({"i": i, "result": "INVALID_JSON", "detail": str(e)[:120]})
            continue

        try:
            eom = EOMDocument.model_validate(payload)
        except Exception as e:
            n_invalid_schema += 1
            per_doc.append({"i": i, "result": "INVALID_SCHEMA", "detail": str(e)[:120]})
            continue

        report = validate(eom, source)
        if report.passed:
            n_pass += 1
            per_doc.append({"i": i, "result": "PASS", "n_blocks": len(eom.blocks)})
        else:
            for f in report.failures:
                rule_fails[f.rule] += 1
            per_doc.append({
                "i": i,
                "result": "HARNESS_FAIL",
                "n_failures": len(report.failures),
                "rules": sorted({f.rule for f in report.failures}),
            })

    return {
        "total": n,
        "pass": n_pass,
        "invalid_json": n_invalid_json,
        "invalid_schema": n_invalid_schema,
        "harness_fail": n - n_pass - n_invalid_json - n_invalid_schema,
        "rule_failure_counts": dict(rule_fails.most_common()),
        "per_doc": per_doc,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path)
    args = p.parse_args()

    if not args.path.exists():
        print(f"Not found: {args.path}", file=sys.stderr)
        sys.exit(1)

    r = evaluate(args.path)
    print(f"\n=== {args.path} ===")
    print(f"Total:           {r['total']}")
    print(f"PASS harness:    {r['pass']} ({100*r['pass']/r['total']:.0f}%)")
    print(f"INVALID_JSON:    {r['invalid_json']}")
    print(f"INVALID_SCHEMA:  {r['invalid_schema']}")
    print(f"HARNESS_FAIL:    {r['harness_fail']}")
    print()
    if r["rule_failure_counts"]:
        print("Rule failure counts (across harness-fail docs):")
        for rule, n in r["rule_failure_counts"].items():
            print(f"  {rule}: {n}")
        print()
    print("Per-doc:")
    for d in r["per_doc"]:
        if d["result"] == "PASS":
            print(f"  [{d['i']}] PASS ({d['n_blocks']} blocks)")
        elif d["result"] == "HARNESS_FAIL":
            print(f"  [{d['i']}] FAIL rules={d['rules']} (n={d['n_failures']})")
        else:
            print(f"  [{d['i']}] {d['result']}: {d['detail']}")


if __name__ == "__main__":
    main()
