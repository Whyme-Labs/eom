# tests/test_compiler_prompted.py
import json

from eom.compilers.llm_client import StubLLMClient
from eom.compilers.prompted import PromptedCompiler
from eom.schema import EOMDocument
from tests.fixtures.loader import load_pair


def _stub_eom_for(source: str, expected_eom: EOMDocument) -> str:
    """Serialise expected EOM but with placeholder source.checksum/chars (compiler will fill)."""
    d = expected_eom.model_dump(mode="json")
    d["source"]["checksum"] = "PLACEHOLDER"
    d["source"]["chars"] = 0
    return json.dumps(d)


def test_compile_with_stub_returns_validated_eom():
    source, expected = load_pair("freight_memo")
    stub = StubLLMClient(response=_stub_eom_for(source, expected))
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source, hints={"document_type": "memo"})
    assert eom.version == "0.1"


def test_compile_calls_llm_with_source_in_prompt():
    source, expected = load_pair("freight_memo")
    stub = StubLLMClient(response=_stub_eom_for(source, expected))
    compiler = PromptedCompiler(client=stub, few_shots=[])
    compiler.compile(source, hints={"document_type": "memo"})
    assert source[:80] in stub.last_request.user


def test_compile_strips_code_fences_around_json():
    source, expected = load_pair("freight_memo")
    response_with_fence = "```json\n" + _stub_eom_for(source, expected) + "\n```"
    stub = StubLLMClient(response=response_with_fence)
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source, hints={"document_type": "memo"})
    assert eom.version == "0.1"


def test_compile_falls_back_to_rules_on_invalid_json():
    source, _ = load_pair("freight_memo")
    stub = StubLLMClient(response="this is not json at all")
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source, hints={"document_type": "memo"})
    # Fallback path should still produce a valid EOM (from RulesCompiler).
    assert eom.version == "0.1"


def test_fills_source_metadata():
    source, expected = load_pair("freight_memo")
    stub = StubLLMClient(response=_stub_eom_for(source, expected))
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source)
    assert eom.source.chars == len(source)
    assert eom.source.checksum.startswith("sha256:")
