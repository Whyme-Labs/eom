"""Deterministic post-processors for compiler output.

Both RulesCompiler and PromptedCompiler can over-emit tier-A and tier-B blocks
relative to harness caps (H3) or token budgets (H9, H10). These functions
demote excess blocks by priority so the harness passes without re-prompting.

Note: H3's small-doc floor is exactly mirrored here:
    cap_a = max(1, int(0.10 * N))
    cap_b = max(2, int(0.25 * N))
"""

from __future__ import annotations

from eom.schema import AttentionBudget, Block
from eom.tokens import count_tokens


def enforce_tier_caps(blocks: list[Block]) -> list[Block]:
    """Demote excess tier-A blocks to B, then excess B to C, by priority asc."""
    n = len(blocks)
    cap_a = max(1, int(0.10 * n))
    cap_b = max(2, int(0.25 * n))

    tier_a = sorted(
        (b for b in blocks if b.attention_tier == "A"),
        key=lambda b: -b.priority,
    )
    keep_a = {b.id for b in tier_a[:cap_a]}
    new_blocks: list[Block] = []
    for b in blocks:
        if b.attention_tier == "A" and b.id not in keep_a:
            b = b.model_copy(update={"attention_tier": "B"})
        new_blocks.append(b)

    tier_b = sorted(
        (b for b in new_blocks if b.attention_tier == "B"),
        key=lambda b: -b.priority,
    )
    keep_b = {b.id for b in tier_b[:cap_b]}
    final: list[Block] = []
    for b in new_blocks:
        if b.attention_tier == "B" and b.id not in keep_b:
            b = b.model_copy(update={"attention_tier": "C"})
        final.append(b)
    return final


def enforce_token_budget(blocks: list[Block], budget: AttentionBudget) -> list[Block]:
    """Demote tier-A blocks until tier-A token sum ≤ B_A; same for A∪B vs B_AB."""
    # Pass 1: B_A
    while True:
        tier_a = [b for b in blocks if b.attention_tier == "A"]
        total = sum(count_tokens(b.content) for b in tier_a)
        if total <= budget.B_A or len(tier_a) <= 1:
            break
        victim = min(tier_a, key=lambda b: b.priority)
        blocks = [
            b.model_copy(update={"attention_tier": "B"}) if b.id == victim.id else b
            for b in blocks
        ]
    # Pass 2: B_AB
    while True:
        ab_total = sum(count_tokens(b.content) for b in blocks
                       if b.attention_tier in ("A", "B"))
        tier_b = [b for b in blocks if b.attention_tier == "B"]
        if ab_total <= budget.B_AB or not tier_b:
            break
        victim = min(tier_b, key=lambda b: b.priority)
        blocks = [
            b.model_copy(update={"attention_tier": "C"}) if b.id == victim.id else b
            for b in blocks
        ]
    return blocks
