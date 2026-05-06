# tests/test_compiler_base.py
from eom.compilers import Compiler, CompileHints, get_compiler  # noqa: F401


def test_get_compiler_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        get_compiler("magical")


def test_compile_hints_is_typed_dict():
    h: CompileHints = {"document_type": "memo"}
    assert h["document_type"] == "memo"
