"""Build the SFT training dataset from gold + synthetic (source, EOM) pairs.

Each example is a (input, target) text pair:
- input = SYSTEM_PROMPT + user prompt with reference-spans menu (no few-shots —
  the fine-tuned model should not need them at inference)
- target = compact JSON of the EOM document (sorted keys, no indent)

Filters: keep only pairs where validate(eom, source).passed is True.

Splits: 90% train / 5% val / 5% test (test = held-out, never seen during training).

Saves as JSONL:
- data/train/sft.jsonl
- data/train/val.jsonl
- data/train/test.jsonl

Usage:
    uv run python scripts/prepare_sft_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from eom.compilers.prompt_template import SYSTEM_PROMPT, build_user_prompt_with_spans
from eom.compilers.scaffolding import extract_reference_spans, format_spans_for_prompt
from eom.harness import validate
from eom.normalise import normalise
from eom.schema import EOMDocument

GOLD = Path("data/gold")
SYNTHETIC = Path("data/synthetic")
OUT = Path("data/train")
SEED = 42
TRAIN_FRAC = 0.90
VAL_FRAC = 0.05
# TEST_FRAC implicit = 1 - TRAIN_FRAC - VAL_FRAC


def _walk_pairs(root: Path):
    """Yield (source_path, eom_path, doc_type, slug) for every (md, eom.json) pair."""
    if not root.exists():
        return
    for type_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            ej = type_dir / f"{md.stem}.eom.json"
            if ej.exists():
                yield (md, ej, type_dir.name, md.stem)


def _build_example(source: str, eom: EOMDocument, doc_type: str) -> dict:
    """Build one (input, target) text pair."""
    spans = extract_reference_spans(source)
    user_prompt = build_user_prompt_with_spans(
        source_text=source,
        document_type=doc_type,
        render_profile=eom.render_profile,
        few_shots="(none — fine-tuned model)",
        spans_menu=format_spans_for_prompt(spans),
    )
    target = json.dumps(eom.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return {
        "input": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
        "target": target,
        "doc_type": doc_type,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    examples: list[dict] = []
    n_skipped = 0
    for root, label in [(GOLD, "gold"), (SYNTHETIC, "synthetic")]:
        for md, ej, doc_type, slug in _walk_pairs(root):
            source = normalise(md.read_text(encoding="utf-8"))
            try:
                eom = EOMDocument.model_validate_json(ej.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  ! skipping {label}/{doc_type}/{slug}: invalid schema ({e})")
                n_skipped += 1
                continue
            report = validate(eom, source)
            if not report.passed:
                print(f"  ! skipping {label}/{doc_type}/{slug}: harness fail "
                      f"({len(report.failures)} rules)")
                n_skipped += 1
                continue
            ex = _build_example(source, eom, doc_type)
            ex["source_label"] = label
            ex["slug"] = slug
            examples.append(ex)

    if not examples:
        raise SystemExit("No valid (source, EOM) pairs found.")

    random.seed(SEED)
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train, val, test = examples[:n_train], examples[n_train:n_train + n_val], examples[n_train + n_val:]

    for name, items in [("sft", train), ("val", val), ("test", test)]:
        out_path = OUT / f"{name}.jsonl"
        with out_path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        # Tally per-source for reporting
        labels = {}
        for it in items:
            labels[it["source_label"]] = labels.get(it["source_label"], 0) + 1
        print(f"  {name:>6}: {len(items):>3} examples → {out_path} ({labels})")

    print(f"\nTotal: {n} valid pairs (skipped {n_skipped})")
    print(f"  train/val/test = {len(train)}/{len(val)}/{len(test)}")


if __name__ == "__main__":
    main()
