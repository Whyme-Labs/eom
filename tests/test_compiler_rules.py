# tests/test_compiler_rules.py
import pytest

from eom.compilers.rules import RulesCompiler
from eom.harness import validate
from tests.fixtures.loader import load_pair


@pytest.fixture
def compiler():
    return RulesCompiler()


def test_compile_returns_eom_document(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    assert eom.version == "0.1"
    assert eom.document_type == "memo"


def test_compile_produces_one_headline(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    n_headline = sum(1 for b in eom.blocks if b.type == "headline")
    assert n_headline == 1


def test_compile_produces_one_lead(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    n_lead = sum(1 for b in eom.blocks if b.type == "lead")
    assert n_lead == 1


def test_compile_passes_structural_invariants(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    report = validate(eom, source)
    structural = {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
    structural_failures = [f for f in report.failures if f.rule in structural]
    assert structural_failures == [], structural_failures


def test_compile_fills_source_metadata(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source)
    assert eom.source.chars == len(source)
    assert eom.source.checksum.startswith("sha256:")


def test_default_render_profile_when_no_hints(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source)
    assert eom.render_profile == "executive_brief"


def test_compile_respects_render_profile_hint(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"render_profile": "analytical_brief"})
    assert eom.render_profile == "analytical_brief"
    assert eom.attention_budget.B_A == 400
