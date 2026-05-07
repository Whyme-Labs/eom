# tests/test_scaffolding.py
from eom.compilers.scaffolding import (
    extract_reference_spans,
    format_spans_for_prompt,
)
from eom.normalise import normalise


def test_simple_doc_yields_headings_and_paragraphs():
    src = normalise("# Title\n\nFirst paragraph.\n\n## Section\n\nSecond paragraph.\n")
    spans = extract_reference_spans(src)
    types = [s.type for s in spans]
    assert types == ["heading", "paragraph", "heading", "paragraph"]
    assert all(src[s.start:s.end] == s.quote for s in spans)


def test_offsets_are_byte_accurate():
    src = normalise("# H\n\nP1.\n\nP2 with **bold**.\n")
    spans = extract_reference_spans(src)
    for s in spans:
        assert src[s.start:s.end] == s.quote, f"Span {s.id} mismatch"


def test_empty_doc_returns_empty():
    assert extract_reference_spans("") == []


def test_only_whitespace_returns_empty():
    assert extract_reference_spans("\n\n   \n") == []


def test_ids_are_contiguous_from_1():
    src = normalise("# A\n\nB\n\n## C\n\nD\n\nE\n")
    spans = extract_reference_spans(src)
    assert [s.id for s in spans] == list(range(1, len(spans) + 1))


def test_format_spans_for_prompt_is_one_line_per_span():
    src = normalise("# Title\n\nPara.\n")
    spans = extract_reference_spans(src)
    out = format_spans_for_prompt(spans)
    assert out.count("\n") == 1  # 2 spans → 1 newline between them
    assert "[1]" in out and "[2]" in out


def test_format_handles_newlines_in_paragraph():
    """Paragraphs in markdown can span lines via soft breaks."""
    src = normalise("Line one\nstill same paragraph.\n\nNew paragraph.\n")
    spans = extract_reference_spans(src)
    assert spans[0].type == "paragraph"
    formatted = format_spans_for_prompt(spans)
    # Newlines inside quotes are escaped
    assert "\\n" in formatted


def test_real_freight_memo_fixture():
    """Sanity-check on the canonical fixture."""
    from tests.fixtures.loader import load_pair
    source, _ = load_pair("freight_memo")
    spans = extract_reference_spans(source)
    assert len(spans) >= 6  # heading + 5 sections at minimum
    for s in spans:
        assert source[s.start:s.end] == s.quote
