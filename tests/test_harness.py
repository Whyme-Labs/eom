from eom.harness import (
    FailureRecord,
    ValidationReport,
    WarningRecord,
    check_h1,
    check_h2,
    check_h3,
    check_h4,
    check_h5,
    check_h6,
    check_h7,
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


class TestH3:
    def _doc_with_tiers(self, tiers: list[str]) -> EOMDocument:
        # Build a document where each appended block has the given tier.
        # First block is headline (always A), second is lead (always A).
        blocks = [
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ]
        for i, t in enumerate(tiers):
            blocks.append(_block(f"claim-{i}", "claim", 2 + i, tier=t))
        return _doc(blocks)

    def test_passes_with_balanced_tiers(self):
        # 25 total: 2 A (head+lead, 8%) + 5 B (20%) + 14 C (56%) + 4 D (16%).
        # All within caps.
        tiers = ["B"] * 5 + ["C"] * 14 + ["D"] * 4
        d = self._doc_with_tiers(tiers)
        assert check_h3(d) == []

    def test_fails_when_A_exceeds_cap(self):
        # 11 blocks total (head+lead+9 A-tier claims): A=11/11=100%
        tiers = ["A"] * 9
        d = self._doc_with_tiers(tiers)
        f = check_h3(d)
        assert any(r.rule == "H3" and "tier A" in r.message for r in f)

    def test_fails_when_B_exceeds_cap(self):
        # 10 blocks: 2 A + 8 B = 80% B, well past 25%
        tiers = ["B"] * 8
        d = self._doc_with_tiers(tiers)
        f = check_h3(d)
        assert any(r.rule == "H3" and "tier B" in r.message for r in f)

    def test_passes_with_only_AB_when_within_caps(self):
        # 25 total: 2 A (8%) + 5 B (20%) + 18 D
        tiers = ["B"] * 5 + ["D"] * 18
        d = self._doc_with_tiers(tiers)
        assert check_h3(d) == []


class TestH4:
    def test_passes_with_total_order(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
            _block("claim-1", "claim", 2),
        ])
        assert check_h4(d) == []

    def test_fails_with_duplicate_reading_order(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 0, tier="A"),
        ])
        f = check_h4(d)
        assert any(r.rule == "H4" for r in f)

    def test_fails_with_gap(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 2, tier="A"),
        ])
        f = check_h4(d)
        assert any(r.rule == "H4" for r in f)


class TestH5:
    def test_passes_unique_ids(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h5(d) == []

    def test_fails_duplicate_ids(self):
        # Pydantic Block doesn't enforce cross-block uniqueness, so we can build a doc with dupes.
        b1 = _block("claim-1", "claim", 0)
        b2 = _block("claim-1", "claim", 1)
        b_head = _block("headline-1", "headline", 2, tier="A")
        b_lead = _block("lead-1", "lead", 3, tier="A")
        d = _doc([b1, b2, b_head, b_lead])
        f = check_h5(d)
        assert any(r.rule == "H5" for r in f)


class TestH6:
    def test_passes_when_all_have_content(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h6(d) == []
    # Empty content is already rejected by Block validators; H6 is a belt-and-braces.


class TestH7:
    # Block validator already restricts type; H7 is a runtime sanity check.
    def test_passes_canonical_types(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h7(d) == []
