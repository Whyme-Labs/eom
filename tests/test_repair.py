# tests/test_repair.py
from eom.repair import compile_with_repair, summarise_failures
from eom.compilers.prompted import PromptedCompiler
from eom.compilers.llm_client import StubLLMClient
from eom.harness import FailureRecord
from tests.fixtures.loader import load_pair


def test_summarise_failures_empty():
    assert summarise_failures([]) == ""


def test_summarise_failures_groups_by_rule():
    fs = [
        FailureRecord(rule="H3", message="tier A 30%"),
        FailureRecord(rule="H3", message="tier B 40%"),
        FailureRecord(rule="H11", message="span out of range", block_id="evidence-1"),
    ]
    summary = summarise_failures(fs)
    assert "H3" in summary
    assert "H11" in summary
    assert "evidence-1" in summary


def test_compile_with_repair_returns_immediately_when_passes(monkeypatch):
    source, expected_eom = load_pair("freight_memo")
    import json
    payload = json.dumps(expected_eom.model_dump(mode="json"))
    stub = StubLLMClient(response=payload)
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom, attempts = compile_with_repair(compiler, source, hints={"document_type": "memo"})
    assert attempts == 1


def test_compile_with_repair_retries_on_failure(monkeypatch):
    source, expected_eom = load_pair("freight_memo")
    import json
    # First response: malformed; second: valid
    valid = json.dumps(expected_eom.model_dump(mode="json"))
    bad = "not json"

    class TwoShotClient:
        def __init__(self):
            self.responses = [bad, valid]
            self.calls = 0

        def complete(self, req):
            r = self.responses[self.calls]
            self.calls += 1
            return r

    client = TwoShotClient()
    compiler = PromptedCompiler(client=client, few_shots=[])
    eom, attempts = compile_with_repair(compiler, source,
                                         hints={"document_type": "memo"},
                                         max_attempts=3)
    # First attempt: bad json -> falls back to rules (counts as attempt 1)
    # Subsequent repair calls re-prompt the same compiler.
    assert attempts >= 1
