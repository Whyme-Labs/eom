"""Markdown-AST-based scaffolding for the prompted compiler.

Extracts every paragraph and heading from a normalised markdown source,
computing their character offset ranges. Long paragraphs are also
sub-divided into individual sentence spans, so the teacher can cite
sentence-level evidence (e.g., a single benchmark result inside a long
arXiv abstract) rather than only the whole paragraph.

The PromptedCompiler passes these as a "reference spans" menu in the
user prompt so the teacher LLM picks valid (start, end, quote) tuples
by reference instead of inventing offsets — directly targeting the H11
source-span quote mismatch failure mode.

List items and code blocks are still skipped for v0.1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from markdown_it import MarkdownIt

# Sentence boundary: a sentence-ending punctuation followed by whitespace and
# either an uppercase ASCII letter or an opening bracket / quote / digit.
# This is intentionally simple — it will mis-split on abbreviations like
# "et al." or "Fig. 1" sometimes, but the teacher can still pick the correct
# whole-paragraph span if a sentence split is bad.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z(\["\'`0-9])')

# Paragraphs shorter than this are not split into sentences (already small
# enough to cite as-is).
_MIN_PARA_LEN_FOR_SPLIT = 200


@dataclass(frozen=True)
class ReferenceSpan:
    """A valid (start, end, quote) tuple the teacher may cite in source_span."""

    id: int  # 1-based, contiguous
    type: str  # "heading", "paragraph", or "sentence"
    start: int  # char offset into normalised source
    end: int  # char offset (exclusive)
    quote: str  # source[start:end] verbatim


def _sentence_subspans(start: int, end: int, source: str) -> list[tuple[int, int]]:
    """Split [start, end) into sentence-level subranges. Returns absolute offsets.

    Each returned (s, e) satisfies source[s:e] == sentence text (no leading or
    trailing whitespace).
    """
    text = source[start:end]
    if len(text) < _MIN_PARA_LEN_FOR_SPLIT:
        return []
    out: list[tuple[int, int]] = []
    pos = 0
    for match in _SENT_SPLIT_RE.finditer(text):
        sent_end = match.start()  # before the whitespace
        if sent_end > pos:
            out.append((pos, sent_end))
        pos = match.end()  # after the whitespace
    if pos < len(text):
        out.append((pos, len(text)))
    # Skip if only one sentence (paragraph-level span already covers it)
    if len(out) <= 1:
        return []
    return [(start + s, start + e) for s, e in out]


def extract_reference_spans(source: str) -> list[ReferenceSpan]:
    """Walk the markdown source and emit one ReferenceSpan per heading/paragraph.

    Offsets are character offsets into `source`. Caller is expected to have
    already passed `source` through `eom.normalise.normalise`.

    Implementation: markdown-it-py emits tokens with `.map = [start_line, end_line]`
    referring to line numbers (0-indexed) in the source. We turn line numbers into
    character offsets via a precomputed line-start table.
    """
    md = MarkdownIt("commonmark")
    tokens = md.parse(source)

    # Precompute: for each line index, the character offset of its first char.
    line_starts = [0]
    for i, ch in enumerate(source):
        if ch == "\n":
            line_starts.append(i + 1)
    n_lines = len(line_starts)

    def offsets_for_line_range(start_line: int, end_line: int) -> tuple[int, int]:
        """Convert (start_line_inclusive, end_line_exclusive) to char offsets.

        markdown-it's .map[1] is exclusive (i.e., points at the line *after* the block).
        Returns (start_char, end_char) where end_char excludes the trailing newline.
        """
        s = line_starts[start_line]
        if end_line >= n_lines:
            e = len(source)
        else:
            e = line_starts[end_line]
            # Drop trailing newlines from the block's end
            while e > s and source[e - 1] in ("\n", "\r"):
                e -= 1
        return s, e

    spans: list[ReferenceSpan] = []
    next_id = 1
    for tok in tokens:
        if tok.type == "heading_open" and tok.map:
            start_line, end_line = tok.map
            s, e = offsets_for_line_range(start_line, end_line)
            quote = source[s:e]
            # Keep verbatim including the # markers.
            # The teacher knows headings include the marker.
            if quote.strip():
                spans.append(
                    ReferenceSpan(
                        id=next_id,
                        type="heading",
                        start=s,
                        end=e,
                        quote=quote,
                    )
                )
                next_id += 1
        elif tok.type == "paragraph_open" and tok.map:
            start_line, end_line = tok.map
            s, e = offsets_for_line_range(start_line, end_line)
            quote = source[s:e]
            if quote.strip():
                spans.append(
                    ReferenceSpan(
                        id=next_id,
                        type="paragraph",
                        start=s,
                        end=e,
                        quote=quote,
                    )
                )
                next_id += 1
                # Long paragraphs get sub-divided into sentence-level spans
                for sub_s, sub_e in _sentence_subspans(s, e, source):
                    spans.append(
                        ReferenceSpan(
                            id=next_id,
                            type="sentence",
                            start=sub_s,
                            end=sub_e,
                            quote=source[sub_s:sub_e],
                        )
                    )
                    next_id += 1
    return spans


def format_spans_for_prompt(spans: list[ReferenceSpan], max_quote_chars: int = 0) -> str:
    """Render the span menu as a compact prompt section.

    Each line: `[N] type=<type> start=<s> end=<e> quote=<full quote>`.
    The teacher MUST copy the start/end/quote triple verbatim; do not
    truncate the quote since the teacher will need the full string.
    """
    if not spans:
        return "(no reference spans extracted from source)"
    lines: list[str] = []
    for sp in spans:
        # Escape newlines in the quote for single-line presentation; the teacher
        # is told these represent literal \n in the source.
        q = sp.quote.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")
        lines.append(
            f'[{sp.id}] type={sp.type} start={sp.start} end={sp.end} quote="{q}"'
        )
    return "\n".join(lines)
