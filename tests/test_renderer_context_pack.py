# tests/test_renderer_context_pack.py
import pytest  # noqa: F401

from eom.renderers.context_pack import render_context_pack
from eom.tokens import count_tokens
from tests.fixtures.loader import load_pair


def test_returns_string():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    assert isinstance(out, str)
    assert len(out) > 0


def test_includes_headline_and_lead_always():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=300)
    headline = next(b for b in eom.blocks if b.type == "headline")
    lead = next(b for b in eom.blocks if b.type == "lead")
    assert headline.content in out
    assert lead.content in out


def test_respects_token_budget():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=200)
    # Some slack for headers and citation markers — within 1.5x budget
    assert count_tokens(out) <= 300


def test_section_headers_present():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    assert "## headline" in out.lower()
    assert "## lead" in out.lower()


def test_citations_present_for_evidence():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    # Each evidence/factbox block in tier A or B should appear with [src:id]
    for b in eom.blocks:
        if b.type in ("evidence", "factbox") and b.attention_tier in ("A", "B"):
            assert f"[src:{b.id}]" in out


def test_tier_a_always_included():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=10_000)
    for b in eom.blocks:
        if b.attention_tier == "A":
            # Block content (or at least its first 30 chars) must be in output.
            head30 = b.content[:30]
            assert head30 in out, f"tier A block {b.id} not rendered"


def test_tier_d_never_included():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=10_000)
    for b in eom.blocks:
        if b.attention_tier == "D":
            assert b.content not in out


def test_summary_header_with_compression_ratio():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    assert "source_tokens" in out or "compression" in out
