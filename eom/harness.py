"""EOM harness — per-document validators (H1–H12).

The harness is the standard. A document is EOM-conformant iff
`validate(eom, source_text).passed is True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eom.schema import EOMDocument
from eom.tokens import count_tokens


@dataclass(frozen=True)
class FailureRecord:
    """One harness failure."""
    rule: str          # e.g. "H3"
    message: str
    block_id: str | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class WarningRecord:
    """A warning (e.g., a property not checkable at this layer)."""
    rule: str
    message: str


@dataclass
class ValidationReport:
    failures: list[FailureRecord] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    metrics: dict[str, int | float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0


def check_h1(doc: EOMDocument) -> list[FailureRecord]:
    """H1: exactly one block of type=headline."""
    n = sum(1 for b in doc.blocks if b.type == "headline")
    if n == 1:
        return []
    return [FailureRecord(rule="H1", message=f"expected 1 headline, found {n}")]


def check_h2(doc: EOMDocument) -> list[FailureRecord]:
    """H2: exactly one lead, with reading_order <= 3."""
    leads = [b for b in doc.blocks if b.type == "lead"]
    out: list[FailureRecord] = []
    if len(leads) != 1:
        out.append(FailureRecord(rule="H2", message=f"expected 1 lead, found {len(leads)}"))
        return out
    lead = leads[0]
    if lead.reading_order > 3:
        out.append(FailureRecord(
            rule="H2",
            message=f"lead reading_order={lead.reading_order} > 3",
            block_id=lead.id,
        ))
    return out


def check_h3(doc: EOMDocument) -> list[FailureRecord]:
    """H3: tier distribution caps. |A|/N <= 0.10, |B|/N <= 0.25, |A|+|B|+|C| <= N."""
    n = len(doc.blocks)
    if n == 0:
        return []  # H6 will catch the empty case
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for b in doc.blocks:
        counts[b.attention_tier] += 1
    out: list[FailureRecord] = []
    if counts["A"] / n > 0.10 + 1e-9:
        out.append(FailureRecord(
            rule="H3",
            message=f"tier A fraction {counts['A']}/{n}={counts['A']/n:.2%} exceeds cap 10%",
        ))
    if counts["B"] / n > 0.25 + 1e-9:
        out.append(FailureRecord(
            rule="H3",
            message=f"tier B fraction {counts['B']}/{n}={counts['B']/n:.2%} exceeds cap 25%",
        ))
    return out


def check_h4(doc: EOMDocument) -> list[FailureRecord]:
    """H4: reading_order is a total order in [0, N) with no duplicates or gaps."""
    n = len(doc.blocks)
    orders = sorted(b.reading_order for b in doc.blocks)
    expected = list(range(n))
    if orders == expected:
        return []
    out: list[FailureRecord] = []
    seen: set[int] = set()
    for b in doc.blocks:
        if b.reading_order in seen:
            out.append(FailureRecord(
                rule="H4",
                message=f"duplicate reading_order {b.reading_order}",
                block_id=b.id,
            ))
        seen.add(b.reading_order)
    if not out:
        # No duplicates, but the sequence has gaps or is out of range.
        out.append(FailureRecord(
            rule="H4",
            message=f"reading_order is not [0, N); got {orders}, expected {expected}",
        ))
    return out


def check_h5(doc: EOMDocument) -> list[FailureRecord]:
    """H5: block IDs unique within document."""
    seen: set[str] = set()
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.id in seen:
            out.append(FailureRecord(rule="H5", message=f"duplicate id {b.id!r}", block_id=b.id))
        seen.add(b.id)
    return out


def check_h6(doc: EOMDocument) -> list[FailureRecord]:
    """H6: every block has non-empty content (re-check at harness layer)."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if not b.content.strip():
            out.append(FailureRecord(
                rule="H6",
                message="block content is empty or whitespace-only",
                block_id=b.id,
            ))
    return out


CANONICAL_BLOCK_TYPES = {
    "headline", "lead", "claim", "evidence",
    "factbox", "caveat", "decision", "appendix",
}


def check_h7(doc: EOMDocument) -> list[FailureRecord]:
    """H7: every block.type is one of the eight canonical types."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.type not in CANONICAL_BLOCK_TYPES:
            out.append(FailureRecord(
                rule="H7",
                message=f"unknown block type {b.type!r}",
                block_id=b.id,
            ))
    return out


def check_h8(doc: EOMDocument) -> list[FailureRecord]:
    """H8: headline <= 100 chars; lead <= 60 words (English)."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.type == "headline" and len(b.content) > 100:
            out.append(FailureRecord(
                rule="H8",
                message=f"headline length {len(b.content)} > 100",
                block_id=b.id,
            ))
        if b.type == "lead":
            n_words = len(b.content.split())
            if n_words > 60:
                out.append(FailureRecord(
                    rule="H8",
                    message=f"lead word count {n_words} > 60",
                    block_id=b.id,
                ))
    return out


def check_h9(doc: EOMDocument) -> list[FailureRecord]:
    """H9: sum of tokens across tier A blocks <= attention_budget.B_A."""
    total = sum(count_tokens(b.content) for b in doc.blocks if b.attention_tier == "A")
    if total > doc.attention_budget.B_A:
        return [FailureRecord(
            rule="H9",
            message=f"tier A total tokens {total} > B_A {doc.attention_budget.B_A}",
        )]
    return []


def check_h10(doc: EOMDocument) -> list[FailureRecord]:
    """H10: sum of tokens across tier A and B blocks <= attention_budget.B_AB."""
    total = sum(
        count_tokens(b.content) for b in doc.blocks
        if b.attention_tier in ("A", "B")
    )
    if total > doc.attention_budget.B_AB:
        return [FailureRecord(
            rule="H10",
            message=f"tier A+B total tokens {total} > B_AB {doc.attention_budget.B_AB}",
        )]
    return []


def check_h11(doc: EOMDocument, source_text: str) -> list[FailureRecord]:
    """H11: every evidence/factbox has a valid source_span (offsets in range, quote matches)."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.type not in ("evidence", "factbox"):
            continue
        if b.source_span is None:
            out.append(FailureRecord(
                rule="H11",
                message=f"{b.type} block missing source_span",
                block_id=b.id,
            ))
            continue
        span = b.source_span
        if span.end > len(source_text):
            out.append(FailureRecord(
                rule="H11",
                message=f"source_span [{span.start},{span.end}) out of range "
                        f"(source has {len(source_text)} chars)",
                block_id=b.id,
                span=(span.start, span.end),
            ))
            continue
        actual = source_text[span.start : span.end]
        if actual != span.quote:
            out.append(FailureRecord(
                rule="H11",
                message=f"source_span quote mismatch: expected {span.quote!r}, "
                        f"got {actual!r}",
                block_id=b.id,
                span=(span.start, span.end),
            ))
    return out


def check_h12(doc: EOMDocument) -> list[FailureRecord]:
    """H12: claim/decision must have source_span or be is_inferred with valid basis.

    Inference basis must reference existing evidence/factbox blocks.
    Span validity (offsets, quote match) is H11's job at the corpus level for
    these block types when source_span is provided; here we check structural
    consistency only.
    """
    out: list[FailureRecord] = []
    by_id = {b.id: b for b in doc.blocks}
    for b in doc.blocks:
        if b.type not in ("claim", "decision"):
            continue
        if b.is_inferred:
            if not b.inference_basis:
                out.append(FailureRecord(
                    rule="H12",
                    message=f"{b.type} is_inferred=True but empty inference_basis",
                    block_id=b.id,
                ))
                continue
            for ref_id in b.inference_basis:
                ref = by_id.get(ref_id)
                if ref is None:
                    out.append(FailureRecord(
                        rule="H12",
                        message=f"inference_basis contains unknown id {ref_id!r}",
                        block_id=b.id,
                    ))
                elif ref.type not in ("evidence", "factbox"):
                    out.append(FailureRecord(
                        rule="H12",
                        message=f"inference_basis target {ref_id!r} is type {ref.type!r}; "
                                f"must be evidence or factbox",
                        block_id=b.id,
                    ))
        else:
            if b.source_span is None:
                out.append(FailureRecord(
                    rule="H12",
                    message=f"{b.type} lacks source_span and is not is_inferred",
                    block_id=b.id,
                ))
    return out


def validate(doc: EOMDocument, source_text: str) -> ValidationReport:
    """Run H1-H12 against the document and source text; return ValidationReport."""
    failures: list[FailureRecord] = []
    failures += check_h1(doc)
    failures += check_h2(doc)
    failures += check_h3(doc)
    failures += check_h4(doc)
    failures += check_h5(doc)
    failures += check_h6(doc)
    failures += check_h7(doc)
    failures += check_h8(doc)
    failures += check_h9(doc)
    failures += check_h10(doc)
    failures += check_h11(doc, source_text)
    failures += check_h12(doc)

    metrics: dict[str, int | float] = {
        "n_blocks": len(doc.blocks),
        "tier_a_count": sum(1 for b in doc.blocks if b.attention_tier == "A"),
        "tier_b_count": sum(1 for b in doc.blocks if b.attention_tier == "B"),
        "tier_c_count": sum(1 for b in doc.blocks if b.attention_tier == "C"),
        "tier_d_count": sum(1 for b in doc.blocks if b.attention_tier == "D"),
        "tier_a_tokens": sum(
            count_tokens(b.content) for b in doc.blocks if b.attention_tier == "A"
        ),
        "tier_ab_tokens": sum(
            count_tokens(b.content) for b in doc.blocks
            if b.attention_tier in ("A", "B")
        ),
    }

    warnings = [
        WarningRecord(
            rule="H13",
            message="salience monotonicity is corpus-level; not checked here",
        ),
        WarningRecord(
            rule="H14",
            message="lead centrality is corpus-level; not checked here",
        ),
    ]

    return ValidationReport(failures=failures, warnings=warnings, metrics=metrics)
