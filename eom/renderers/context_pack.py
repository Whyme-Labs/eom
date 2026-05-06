"""Render EOM as a token-budgeted text payload for LLM ingestion.

Tier-A blocks are always included. Then tier B by priority desc, until
budget is tight. Then tier C as one-line summaries. Tier D omitted.
"""

from __future__ import annotations

from eom.schema import Block, EOMDocument
from eom.tokens import count_tokens

_SECTION_ORDER = (
    ("headline", "## headline"),
    ("lead", "## lead"),
    ("decision", "## decisions"),
    ("factbox", "## facts"),
    ("evidence", "## evidence"),
    ("claim", "## claims"),
    ("caveat", "## caveats"),
    ("appendix", "## appendix"),
)


def _format_block(b: Block, *, with_citation: bool) -> str:
    """One line per block. Citation marker for evidence/factbox in tier A/B."""
    citation = f" [src:{b.id}]" if with_citation else ""
    return f"- {b.content}{citation}"


def _summary_header(eom: EOMDocument, body_tokens: int) -> str:
    compression = body_tokens / max(1, eom.source.chars)
    return (
        f"<!-- eom_v{eom.version} | profile={eom.render_profile} | "
        f"document_type={eom.document_type} | "
        f"source_chars={eom.source.chars} | "
        f"context_tokens={body_tokens} | "
        f"compression={compression:.3f} -->\n"
        f"{eom.summary}\n"
    )


def render_context_pack(eom: EOMDocument, token_budget: int) -> str:
    """Build a context pack respecting `token_budget` (tiktoken cl100k_base)."""
    by_tier: dict[str, list[Block]] = {"A": [], "B": [], "C": [], "D": []}
    for b in eom.blocks:
        by_tier[b.attention_tier].append(b)
    for tier in by_tier.values():
        tier.sort(key=lambda x: (-x.priority, x.reading_order))

    chosen: list[Block] = []
    chosen.extend(by_tier["A"])  # always include tier A

    used = sum(count_tokens(b.content) for b in chosen)
    headroom = max(0, token_budget - used - 100)  # 100-token safety margin for headers

    # Greedy add tier B by priority desc.
    for b in by_tier["B"]:
        cost = count_tokens(b.content)
        if cost <= headroom:
            chosen.append(b)
            headroom -= cost

    # Tier C as one-line summaries (truncated to first sentence + ellipsis).
    for b in by_tier["C"]:
        first_sent = b.content.split(". ", 1)[0]
        truncated = first_sent if first_sent.endswith(".") else first_sent + "…"
        cost = count_tokens(truncated)
        if cost <= headroom:
            chosen.append(b.model_copy(update={"content": truncated}))
            headroom -= cost

    # Group by section in canonical order.
    by_type: dict[str, list[Block]] = {t: [] for t, _ in _SECTION_ORDER}
    for b in chosen:
        if b.type in by_type:
            by_type[b.type].append(b)

    body_lines: list[str] = []
    for type_key, header in _SECTION_ORDER:
        blocks = by_type[type_key]
        if not blocks:
            continue
        body_lines.append(header)
        for b in blocks:
            with_cite = b.type in ("evidence", "factbox") and b.attention_tier in ("A", "B")
            body_lines.append(_format_block(b, with_citation=with_cite))
        body_lines.append("")

    body = "\n".join(body_lines).rstrip() + "\n"
    body_tokens = count_tokens(body)

    return _summary_header(eom, body_tokens) + "\n" + body
