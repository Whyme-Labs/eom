import pytest
from pydantic import ValidationError

from eom.schema import SourceSpan, SourceMetadata


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
