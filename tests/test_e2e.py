# tests/test_e2e.py
"""End-to-end: compile every gold .md with rules, validate, render both views."""

from pathlib import Path

import pytest

from eom.harness import validate
from eom.normalise import normalise
from eom.renderers import render_context_pack, render_newspaper

GOLD_DIR = Path("data/gold")


def _gold_pairs():
    if not GOLD_DIR.exists():
        return []
    pairs = []
    for type_dir in sorted(p for p in GOLD_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            eom_path = type_dir / f"{md.stem}.eom.json"
            if eom_path.exists():
                pairs.append((type_dir.name, md, eom_path))
    return pairs


@pytest.mark.parametrize("doc_type,md_path,eom_path", _gold_pairs())
def test_gold_eom_passes_harness(doc_type, md_path, eom_path):
    from eom.schema import EOMDocument
    source = normalise(md_path.read_text(encoding="utf-8"))
    eom = EOMDocument.model_validate_json(eom_path.read_text(encoding="utf-8"))
    report = validate(eom, source)
    if not report.passed:
        for f in report.failures:
            print(f"{f.rule} [{f.block_id}]: {f.message}")
    assert report.passed


@pytest.mark.parametrize("doc_type,md_path,eom_path", _gold_pairs())
def test_gold_eom_renders(doc_type, md_path, eom_path):
    from eom.schema import EOMDocument
    eom = EOMDocument.model_validate_json(eom_path.read_text(encoding="utf-8"))
    html = render_newspaper(eom)
    assert html.lower().startswith(("<!doctype", "<html"))
    text = render_context_pack(eom, token_budget=3000)
    assert "## headline" in text.lower()
