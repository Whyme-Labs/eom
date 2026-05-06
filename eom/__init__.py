"""EOM — Editorial Object Model.

Public API:
    compile(source_text, hints) -> EOMDocument
    validate(eom, source_text) -> ValidationReport
    render_newspaper(eom) -> str
    render_context_pack(eom, token_budget) -> str
"""

__version__ = "0.1.0"

from eom.harness import validate, ValidationReport, FailureRecord, WarningRecord
from eom.schema import (
    EOMDocument,
    Block,
    SourceSpan,
    SourceMetadata,
    AttentionBudget,
    RENDER_PROFILES,
)

__all__ = [
    "validate",
    "ValidationReport",
    "FailureRecord",
    "WarningRecord",
    "EOMDocument",
    "Block",
    "SourceSpan",
    "SourceMetadata",
    "AttentionBudget",
    "RENDER_PROFILES",
]
