from eom.harness import (
    FailureRecord,
    ValidationReport,
    WarningRecord,
    check_h1,
    check_h2,
)
from eom.schema import Block, EOMDocument, AttentionBudget, SourceMetadata, SourceSpan


def _block(id: str, type: str, ro: int, tier: str = "C", with_span: bool = True) -> Block:
    return Block(
        id=id,
        type=type,
        content=f"content for {id}",
        attention_tier=tier,
        priority=0.5,
        reading_order=ro,
        source_span=SourceSpan(start=0, end=10, quote="0123456789") if with_span else None,
    )


def _doc(blocks: list[Block]) -> EOMDocument:
    return EOMDocument(
        version="0.1",
        document_type="memo",
        summary="x",
        render_profile="executive_brief",
        attention_budget=AttentionBudget(B_A=200, B_AB=800),
        blocks=blocks,
        source=SourceMetadata(checksum="sha256:x", chars=100, lang="en"),
    )


class TestValidationReport:
    def test_passed_when_no_failures(self):
        r = ValidationReport(failures=[], warnings=[], metrics={})
        assert r.passed is True

    def test_failed_when_failures_present(self):
        f = FailureRecord(rule="H1", message="x", block_id=None)
        r = ValidationReport(failures=[f], warnings=[], metrics={})
        assert r.passed is False

    def test_passed_with_only_warnings(self):
        w = WarningRecord(rule="H13", message="not checkable here")
        r = ValidationReport(failures=[], warnings=[w], metrics={})
        assert r.passed is True


class TestFailureRecord:
    def test_creates_failure(self):
        f = FailureRecord(rule="H3", message="too many tier A", block_id=None)
        assert f.rule == "H3"
        assert f.block_id is None

    def test_creates_failure_with_block_id(self):
        f = FailureRecord(rule="H11", message="span out of range", block_id="evidence-1")
        assert f.block_id == "evidence-1"


class TestH1:
    def test_passes_with_one_headline(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h1(d) == []

    def test_fails_with_zero_headlines(self):
        d = _doc([_block("lead-1", "lead", 0, tier="A")])
        f = check_h1(d)
        assert len(f) == 1
        assert f[0].rule == "H1"

    def test_fails_with_two_headlines(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("headline-2", "headline", 1, tier="A"),
            _block("lead-1", "lead", 2, tier="A"),
        ])
        f = check_h1(d)
        assert len(f) == 1
        assert "2" in f[0].message


class TestH2:
    def test_passes_with_lead_at_ro_1(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h2(d) == []

    def test_passes_with_lead_at_ro_3(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("appendix-x", "appendix", 1, tier="C"),
            _block("appendix-y", "appendix", 2, tier="C"),
            _block("lead-1", "lead", 3, tier="A"),
        ])
        assert check_h2(d) == []

    def test_fails_with_lead_at_ro_4(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("appendix-1", "appendix", 1, tier="C"),
            _block("appendix-2", "appendix", 2, tier="C"),
            _block("appendix-3", "appendix", 3, tier="C"),
            _block("lead-1", "lead", 4, tier="A"),
        ])
        f = check_h2(d)
        assert len(f) == 1
        assert f[0].rule == "H2"

    def test_fails_with_zero_leads(self):
        d = _doc([_block("headline-1", "headline", 0, tier="A")])
        f = check_h2(d)
        assert len(f) == 1

    def test_fails_with_two_leads(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
            _block("lead-2", "lead", 2, tier="A"),
        ])
        f = check_h2(d)
        assert len(f) >= 1
