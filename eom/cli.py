"""eom CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from eom.compilers import get_compiler
from eom.compilers.llm_client import OpenRouterClient
from eom.harness import validate as run_validate
from eom.normalise import normalise
from eom.renderers import render_context_pack, render_newspaper
from eom.repair import compile_with_repair
from eom.schema import EOMDocument


@click.group()
@click.version_option()
def cli():
    """EOM — Editorial Object Model."""


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--compiler", "compiler_kind", type=click.Choice(["rules", "prompted"]),
              default="rules")
@click.option("--profile", type=click.Choice(["executive_brief", "analytical_brief"]),
              default="executive_brief")
@click.option("--document-type",
              type=click.Choice(["memo", "report", "paper", "transcript",
                                  "news", "policy", "other"]),
              default="other")
@click.option("--max-attempts", type=int, default=3)
def compile(input_path, output_path, compiler_kind, profile, document_type, max_attempts):
    """Compile a markdown / text document into EOM JSON."""
    source = normalise(Path(input_path).read_text(encoding="utf-8"))
    if compiler_kind == "prompted":
        client = OpenRouterClient()
        compiler_obj = get_compiler("prompted", client=client, few_shots=[])
    else:
        compiler_obj = get_compiler("rules")

    hints = {"document_type": document_type, "render_profile": profile}
    eom, attempts = compile_with_repair(compiler_obj, source, hints=hints, max_attempts=max_attempts)
    Path(output_path).write_text(eom.model_dump_json(indent=2))
    click.echo(f"Compiled in {attempts} attempt(s) -> {output_path}")


@cli.command()
@click.option("--eom", "eom_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--source", "source_path", required=True, type=click.Path(exists=True, dir_okay=False))
def validate(eom_path, source_path):
    """Validate an EOM JSON against its source."""
    source = normalise(Path(source_path).read_text(encoding="utf-8"))
    eom = EOMDocument.model_validate_json(Path(eom_path).read_text(encoding="utf-8"))
    report = run_validate(eom, source)
    if report.passed:
        click.echo("PASS")
        click.echo(f"  blocks: {int(report.metrics['n_blocks'])}")
        click.echo(f"  tier A: {int(report.metrics['tier_a_count'])}")
        click.echo(f"  tier A tokens: {int(report.metrics['tier_a_tokens'])}")
    else:
        click.echo(f"FAIL ({len(report.failures)} failures)")
        for f in report.failures:
            tag = f" [{f.block_id}]" if f.block_id else ""
            click.echo(f"  {f.rule}{tag}: {f.message}")
        sys.exit(1)


@cli.command()
@click.option("--eom", "eom_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--target", type=click.Choice(["newspaper", "context-pack"]), required=True)
@click.option("--output", "-o", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--budget", type=int, default=3000, help="Token budget for context-pack")
def render(eom_path, target, output_path, budget):
    """Render an EOM to HTML or text."""
    eom = EOMDocument.model_validate_json(Path(eom_path).read_text(encoding="utf-8"))
    if target == "newspaper":
        out = render_newspaper(eom)
    else:
        out = render_context_pack(eom, token_budget=budget)
    Path(output_path).write_text(out, encoding="utf-8")
    click.echo(f"Rendered -> {output_path}")
