"""Render EOM as a printable, single-page newspaper-style HTML view."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from eom.schema import Block, EOMDocument

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(default_for_string=True),
        keep_trailing_newline=True,
    )


def _partition(blocks: list[Block]) -> dict[str, list[Block]]:
    out: dict[str, list[Block]] = {"A": [], "B": [], "C": [], "D": []}
    for b in blocks:
        out[b.attention_tier].append(b)
    for tier in out.values():
        tier.sort(key=lambda b: (-b.priority, b.reading_order))
    return out


def render_newspaper(eom: EOMDocument) -> str:
    parts = _partition(eom.blocks)
    headline = next((b for b in parts["A"] if b.type == "headline"), None)
    if headline is None:
        # Caller has bypassed harness; render anyway with placeholder.
        headline = Block(
            id="headline-missing", type="headline",
            content="(missing headline)", attention_tier="A",
            priority=1.0, reading_order=0,
        )
    lead = next((b for b in eom.blocks if b.type == "lead"), None)
    tier_a_extras = [b for b in parts["A"] if b.type not in ("headline", "lead")]

    # Exclude lead from tier-grouped sections (already in hero).
    tier_b = [b for b in parts["B"] if b.type != "lead"]
    tier_c = [b for b in parts["C"] if b.type != "lead"]
    tier_d = [b for b in parts["D"] if b.type != "lead"]

    css = (_TEMPLATE_DIR / "newspaper.css").read_text(encoding="utf-8")
    template = _env().get_template("newspaper.html")
    return template.render(
        eom=eom,
        css=css,
        headline=headline,
        lead=lead,
        tier_a_extras=tier_a_extras,
        tier_b=tier_b,
        tier_c=tier_c,
        tier_d=tier_d,
    )
