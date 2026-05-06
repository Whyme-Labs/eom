"""EOM compiler implementations.

Use `get_compiler(kind)` to select a backend by name.
"""

from __future__ import annotations

from typing import Literal

from eom.compilers.base import Compiler, CompileHints

CompilerKind = Literal["rules", "prompted", "finetuned"]


def get_compiler(kind: CompilerKind, **kwargs) -> Compiler:
    """Factory: import lazily so optional ML deps don't break the rules path."""
    if kind == "rules":
        from eom.compilers.rules import RulesCompiler
        return RulesCompiler(**kwargs)
    if kind == "prompted":
        from eom.compilers.prompted import PromptedCompiler
        return PromptedCompiler(**kwargs)
    if kind == "finetuned":
        from eom.compilers.finetuned import FineTunedCompiler
        return FineTunedCompiler(**kwargs)
    raise ValueError(f"unknown compiler kind: {kind!r}")


__all__ = ["Compiler", "CompileHints", "get_compiler"]
