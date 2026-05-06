import pytest
from pydantic import ValidationError

from eom.schema import SourceSpan, SourceMetadata
from eom.schema import AttentionBudget, Block


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
