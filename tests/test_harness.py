from eom.harness import (
    FailureRecord,
    ValidationReport,
    WarningRecord,
)


class TestValidationReport:
    def test_passed_when_no_failures(self):
        r = ValidationReport(failures=[], warnings=[], metrics={})
        assert r.passed is True

    def test_failed_when_failures_present(self):
        f = FailureRecord(rule="H1", message="x", block_id=None)
        r = ValidationReport(failures=[f], warnings=[], metrics={})
        assert r.passed is False

    def test_passed_with_only_warnings(self):
        w = WarningRecord(rule="H13", message="not checkable here")
        r = ValidationReport(failures=[], warnings=[w], metrics={})
        assert r.passed is True


class TestFailureRecord:
    def test_creates_failure(self):
        f = FailureRecord(rule="H3", message="too many tier A", block_id=None)
        assert f.rule == "H3"
        assert f.block_id is None

    def test_creates_failure_with_block_id(self):
        f = FailureRecord(rule="H11", message="span out of range", block_id="evidence-1")
        assert f.block_id == "evidence-1"
