"""Compiler interface and shared types."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, runtime_checkable

from eom.schema import EOMDocument


class CompileHints(TypedDict, total=False):
    """Optional hints passed to a compiler. All fields are optional."""

    document_type: Literal[
        "memo", "report", "paper", "transcript", "news", "policy", "other"
    ]
    audience: Literal["executive", "researcher", "general"]
    render_profile: Literal["executive_brief", "analytical_brief"]
    token_budget: int


@runtime_checkable
class Compiler(Protocol):
    """Common interface for all EOM compilers."""

    def compile(
        self,
        source_text: str,
        hints: CompileHints | None = None,
    ) -> EOMDocument: ...
