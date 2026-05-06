"""EOM harness — per-document validators (H1–H12).

The harness is the standard. A document is EOM-conformant iff
`validate(eom, source_text).passed is True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eom.schema import EOMDocument


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
