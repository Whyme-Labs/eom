import pytest
from pydantic import ValidationError

from eom.schema import SourceSpan, SourceMetadata
from eom.schema import AttentionBudget, Block
from eom.schema import EOMDocument, RENDER_PROFILES


class TestSourceSpan:
    def test_creates_valid_span(self):
        span = SourceSpan(start=10, end=20, quote="hello world")
        assert span.start == 10
        assert span.end == 20
        assert span.quote == "hello world"

    def test_rejects_negative_start(self):
        with pytest.raises(ValidationError):
            SourceSpan(start=-1, end=5, quote="x")

    def test_rejects_end_before_start(self):
        with pytest.raises(ValidationError):
            SourceSpan(start=10, end=5, quote="x")

    def test_rejects_empty_quote(self):
        with pytest.raises(ValidationError):
            SourceSpan(start=0, end=5, quote="")


class TestSourceMetadata:
    def test_creates_valid_metadata(self):
        meta = SourceMetadata(
            checksum="sha256:abc123",
            chars=512,
            lang="en",
        )
        assert meta.lang == "en"

    def test_rejects_negative_chars(self):
        with pytest.raises(ValidationError):
            SourceMetadata(checksum="sha256:x", chars=-1, lang="en")

    def test_rejects_invalid_lang_format(self):
        with pytest.raises(ValidationError):
            SourceMetadata(checksum="sha256:x", chars=10, lang="english")

    @pytest.mark.parametrize("bad_lang", ["EN", "e1", "éà", "ñé", "αβ", "e", "eng", " en"])
    def test_rejects_non_ascii_or_non_lowercase_lang(self, bad_lang):
        with pytest.raises(ValidationError):
            SourceMetadata(checksum="sha256:x", chars=10, lang=bad_lang)


class TestAttentionBudget:
    def test_creates_valid_budget(self):
        b = AttentionBudget(B_A=200, B_AB=800)
        assert b.B_A == 200
        assert b.B_AB == 800

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            AttentionBudget(B_A=-1, B_AB=800)

    def test_rejects_BAB_smaller_than_BA(self):
        with pytest.raises(ValidationError):
            AttentionBudget(B_A=500, B_AB=200)


class TestBlock:
    def _ok_block_kwargs(self, **overrides):
        defaults = dict(
            id="claim-1",
            type="claim",
            content="The sky is blue.",
            attention_tier="B",
            priority=0.5,
            reading_order=2,
            source_span=SourceSpan(start=0, end=18, quote="The sky is blue.  "),
        )
        defaults.update(overrides)
        return defaults

    def test_creates_valid_block(self):
        b = Block(**self._ok_block_kwargs())
        assert b.id == "claim-1"
        assert b.is_inferred is False
        assert b.inference_basis == []
        assert b.parent_id is None

    def test_priority_in_unit_interval(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(priority=1.1))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(priority=-0.1))

    def test_reading_order_non_negative(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(reading_order=-1))

    def test_rejects_unknown_type(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(type="quote"))

    def test_rejects_unknown_tier(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(attention_tier="E"))

    def test_is_inferred_only_for_claim_or_decision(self):
        # is_inferred=True on an evidence block must fail at the block level
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(
                type="evidence",
                is_inferred=True,
                source_span=None,
            ))

    def test_inference_basis_requires_is_inferred(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(
                is_inferred=False,
                inference_basis=["evidence-1"],
            ))

    def test_inferred_claim_with_basis(self):
        b = Block(**self._ok_block_kwargs(
            is_inferred=True,
            inference_basis=["evidence-1", "factbox-1"],
            source_span=None,
        ))
        assert b.is_inferred is True
        assert b.inference_basis == ["evidence-1", "factbox-1"]

    def test_id_must_be_slug(self):
        # IDs are type-prefixed, lowercase, alphanumeric + hyphen only.
        Block(**self._ok_block_kwargs(id="claim-1"))
        Block(**self._ok_block_kwargs(id="evidence-12"))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(id="Claim 1"))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(id="claim_1"))

    @pytest.mark.parametrize("bad_id", ["claim-1\n", "claim-1\t", "claim-1 ", " claim-1"])
    def test_id_rejects_embedded_whitespace(self, bad_id):
        # re.match would silently allow a trailing newline; we use fullmatch.
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(id=bad_id))

    def test_content_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(content=""))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(content="   "))


class TestRenderProfiles:
    def test_two_profiles_defined(self):
        assert "executive_brief" in RENDER_PROFILES
        assert "analytical_brief" in RENDER_PROFILES

    def test_executive_brief_budgets(self):
        b = RENDER_PROFILES["executive_brief"]
        assert b.B_A == 200
        assert b.B_AB == 800

    def test_analytical_brief_budgets(self):
        b = RENDER_PROFILES["analytical_brief"]
        assert b.B_A == 400
        assert b.B_AB == 2000


class TestEOMDocument:
    def _ok_doc(self, **overrides):
        head = Block(
            id="headline-1", type="headline",
            content="Test", attention_tier="A",
            priority=1.0, reading_order=0,
            source_span=SourceSpan(start=0, end=4, quote="Test"),
        )
        defaults = dict(
            version="0.1",
            document_type="memo",
            summary="A test document.",
            render_profile="executive_brief",
            attention_budget=AttentionBudget(B_A=200, B_AB=800),
            blocks=[head],
            source=SourceMetadata(checksum="sha256:test", chars=4, lang="en"),
        )
        defaults.update(overrides)
        return defaults

    def test_creates_valid_doc(self):
        d = EOMDocument(**self._ok_doc())
        assert d.version == "0.1"
        assert d.document_type == "memo"

    def test_rejects_unknown_version(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(version="0.2"))

    def test_rejects_unknown_document_type(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(document_type="email"))

    def test_rejects_unknown_render_profile(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(render_profile="custom"))

    def test_rejects_empty_blocks_list(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(blocks=[]))

    def test_round_trip_json(self):
        d = EOMDocument(**self._ok_doc())
        roundtripped = EOMDocument.model_validate_json(d.model_dump_json())
        assert roundtripped == d

    @pytest.mark.parametrize("bad_summary", ["", "   ", "\t\n"])
    def test_rejects_blank_summary(self, bad_summary):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(summary=bad_summary))

    def test_render_profiles_is_immutable_view(self):
        # Read-only mapping: cannot add or replace keys.
        with pytest.raises(TypeError):
            RENDER_PROFILES["new_profile"] = AttentionBudget(B_A=1, B_AB=2)  # type: ignore[index]

    def test_attention_budget_is_frozen(self):
        # Frozen models reject in-place mutation; use model_copy(update=...) instead.
        b = RENDER_PROFILES["executive_brief"]
        with pytest.raises(ValidationError):
            b.B_A = 999  # type: ignore[misc]
