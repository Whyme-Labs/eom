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
