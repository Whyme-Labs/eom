# tests/test_cli.py
import json

import pytest
from click.testing import CliRunner

from eom.cli import cli
from tests.fixtures.loader import load_pair


@pytest.fixture
def runner():
    return CliRunner()


def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "compile" in result.output
    assert "validate" in result.output
    assert "render" in result.output


def test_compile_with_rules(tmp_path, runner):
    source, _ = load_pair("freight_memo")
    md = tmp_path / "doc.md"
    md.write_text(source)
    out = tmp_path / "doc.eom.json"
    result = runner.invoke(cli, [
        "compile", "--input", str(md),
        "--compiler", "rules",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["version"] == "0.1"


def test_validate_fixture(tmp_path, runner):
    source, eom = load_pair("freight_memo")
    md = tmp_path / "doc.md"
    md.write_text(source)
    eom_path = tmp_path / "doc.eom.json"
    eom_path.write_text(eom.model_dump_json())
    result = runner.invoke(cli, [
        "validate", "--eom", str(eom_path),
        "--source", str(md),
    ])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_render_newspaper(tmp_path, runner):
    _, eom = load_pair("freight_memo")
    eom_path = tmp_path / "doc.eom.json"
    eom_path.write_text(eom.model_dump_json())
    out = tmp_path / "doc.html"
    result = runner.invoke(cli, [
        "render", "--eom", str(eom_path),
        "--target", "newspaper",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.read_text().lower().startswith(("<!doctype", "<html"))


def test_render_context_pack(tmp_path, runner):
    _, eom = load_pair("freight_memo")
    eom_path = tmp_path / "doc.eom.json"
    eom_path.write_text(eom.model_dump_json())
    out = tmp_path / "doc.txt"
    result = runner.invoke(cli, [
        "render", "--eom", str(eom_path),
        "--target", "context-pack",
        "--budget", "3000",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "## headline" in text.lower()
