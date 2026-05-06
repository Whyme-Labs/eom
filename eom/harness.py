"""EOM harness — per-document validators (H1–H12).

The harness is the standard. A document is EOM-conformant iff
`validate(eom, source_text).passed is True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eom.schema import EOMDocument  # noqa: F401


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
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0
