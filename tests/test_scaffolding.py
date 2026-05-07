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


def test_long_paragraph_splits_into_sentences():
    """A paragraph >200 chars with multiple sentences emits paragraph + sentence spans."""
    long_para = (
        "We propose a new sequence-to-sequence architecture called the Transformer. "
        "It dispenses with recurrence and convolutions entirely, relying solely on attention. "
        "Experiments show BLEU 28.4 on the WMT 2014 EN-DE translation task. "
        "The model trains in 12 hours on 8 GPUs."
    )
    src = normalise(f"# Paper\n\n{long_para}\n")
    spans = extract_reference_spans(src)
    types = [s.type for s in spans]
    # Expect: heading, paragraph, sentence, sentence, sentence, sentence
    assert types[0] == "heading"
    assert types[1] == "paragraph"
    n_sentences = types.count("sentence")
    assert n_sentences == 4, f"expected 4 sentences, got {n_sentences} ({types})"
    # Every span (including sentences) byte-matches source[start:end]
    for s in spans:
        assert src[s.start:s.end] == s.quote, f"span {s.id} ({s.type}) mismatch"


def test_short_paragraph_does_not_split():
    """A paragraph <200 chars should NOT emit sentence spans."""
    src = normalise("Short. Para. With sentences.\n")
    spans = extract_reference_spans(src)
    assert all(s.type != "sentence" for s in spans)


def test_single_sentence_paragraph_does_not_split():
    """Long paragraph with only ONE sentence (no boundaries) returns no sentence spans."""
    src = normalise(
        "This is one very long sentence that goes on and on and on without "
        "any sentence-ending punctuation to indicate where a new sentence "
        "would begin and so the sentence splitter should leave it alone\n"
    )
    spans = extract_reference_spans(src)
    assert all(s.type != "sentence" for s in spans)


def test_sentence_spans_byte_accurate_on_real_paper():
    """Run on a real arXiv abstract to confirm sentence offsets align."""
    src = normalise(
        "# Mistral 7B\n\n"
        "We introduce Mistral 7B, a 7-billion-parameter language model. "
        "Mistral 7B outperforms Llama-2-13B across all benchmarks. "
        "It uses grouped-query attention for fast inference. "
        "The model is released under Apache 2.0 license.\n"
    )
    spans = extract_reference_spans(src)
    for s in spans:
        assert src[s.start:s.end] == s.quote, (
            f"span {s.id} ({s.type}) mismatch: {s.quote!r} vs {src[s.start:s.end]!r}"
        )
    # Should have heading + paragraph + 4 sentences
    sentence_spans = [s for s in spans if s.type == "sentence"]
    assert len(sentence_spans) == 4
