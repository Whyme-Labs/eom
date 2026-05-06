# tests/test_llm_client.py
from eom.compilers.llm_client import LLMRequest, StubLLMClient


def test_stub_returns_preset():
    stub = StubLLMClient(response="hello")
    req = LLMRequest(system="sys", user="usr")
    out = stub.complete(req)
    assert out == "hello"
    assert stub.last_request is req


def test_request_defaults():
    req = LLMRequest(system="s", user="u")
    assert req.temperature == 0.0
    assert "gemma" in req.model.lower()
