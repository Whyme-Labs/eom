"""Repair loop for LLM-based compilers.

If the compiler output fails the harness, summarise failures and re-prompt
the compiler with the failure feedback. Cap at `max_attempts` total
compile invocations. Return (best_eom, attempts) where `best_eom` is the
last attempt's output (which may still fail; check `validate(eom, source).passed`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from eom.compilers.base import Compiler, CompileHints
from eom.harness import FailureRecord, validate
from eom.schema import EOMDocument


def summarise_failures(failures: list[FailureRecord]) -> str:
    """Group failures by rule and produce an actionable summary string."""
    if not failures:
        return ""
    by_rule: dict[str, list[FailureRecord]] = defaultdict(list)
    for f in failures:
        by_rule[f.rule].append(f)
    lines = ["Your previous output failed the EOM harness. Fix these issues:"]
    for rule, fs in sorted(by_rule.items()):
        lines.append(f"\n[{rule}] {len(fs)} failure(s):")
        for f in fs[:5]:  # cap per-rule details
            tag = f" (block {f.block_id})" if f.block_id else ""
            lines.append(f"  - {f.message}{tag}")
        if len(fs) > 5:
            lines.append(f"  - ...and {len(fs) - 5} more.")
    lines.append("\nProduce a corrected EOM JSON. Output JSON only.")
    return "\n".join(lines)


class _CompilerWithFeedback(Protocol):
    """Marker protocol: a compiler that supports an optional feedback string."""
    def compile(
        self, source_text: str, hints: CompileHints | None = ...,
        feedback: str | None = ...,
    ) -> EOMDocument: ...


def compile_with_repair(
    compiler: Compiler,
    source_text: str,
    hints: CompileHints | None = None,
    max_attempts: int = 3,
) -> tuple[EOMDocument, int]:
    """Run compiler with up to `max_attempts` attempts; return (eom, attempts)."""
    attempts = 0
    last_eom: EOMDocument | None = None
    feedback: str | None = None
    for _ in range(max_attempts):
        attempts += 1
        if feedback is None:
            eom = compiler.compile(source_text, hints=hints)
        else:
            # If compiler doesn't accept feedback kw, fall back to plain compile.
            try:
                eom = compiler.compile(source_text, hints=hints, feedback=feedback)  # type: ignore[call-arg]
            except TypeError:
                eom = compiler.compile(source_text, hints=hints)
        last_eom = eom
        report = validate(eom, source_text)
        if report.passed:
            return eom, attempts
        feedback = summarise_failures(report.failures)
    assert last_eom is not None  # max_attempts >= 1
    return last_eom, attempts
