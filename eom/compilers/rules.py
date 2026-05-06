"""Deterministic rule-based EOM compiler.

Heuristics:
- First H1 (or first non-empty line if no H1) -> headline.
- First paragraph after the headline -> lead candidate (truncated to 60 words).
- Sentences with hedging language ("however", "limited", "may not", "depends on",
  "uncertain", "caveat") -> caveat.
- Bulleted/numeric lists or paragraphs containing >=3 numbers/percentages -> factbox.
- Paragraphs containing imperative recommendation language ("recommend", "approve",
  "should", "must") in early/mid position -> decision.
- Everything else -> claim or appendix by position.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from eom.compilers.base import CompileHints
from eom.normalise import normalise
from eom.schema import (
    AttentionBudget,
    Block,
    EOMDocument,
    RENDER_PROFILES,
    SourceMetadata,
    SourceSpan,
)
from eom.tokens import count_tokens

_HEDGING = re.compile(
    r"\b(however|limited|may not|depends on|uncertain|caveat|reassess|"
    r"based on .*alone|risk|nevertheless|but)\b",
    re.IGNORECASE,
)
_DECISION_VERBS = re.compile(
    r"\b(recommend|approve|reject|implement|adopt|should|must|propose|require)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")


@dataclass
class _Para:
    """A normalised paragraph with its source offsets."""
    text: str
    start: int
    end: int


def _segment_paragraphs(source_text: str) -> list[_Para]:
    """Split into paragraphs (double-newline separators) with offsets."""
    paragraphs: list[_Para] = []
    pos = 0
    for chunk in re.split(r"\n\n+", source_text):
        # Find this chunk in the source from pos onward.
        idx = source_text.find(chunk, pos)
        if idx < 0:
            idx = pos
        text = chunk.strip()
        if text:
            paragraphs.append(_Para(text=text, start=idx, end=idx + len(chunk)))
        pos = idx + len(chunk)
    return paragraphs


def _strip_md_heading(text: str) -> str:
    """Remove leading '#'-style heading markers."""
    return re.sub(r"^#+\s*", "", text).strip()


def _is_heading(text: str) -> bool:
    return bool(re.match(r"^#+\s+", text))


class RulesCompiler:
    """Deterministic compiler. No LLM calls."""

    def compile(
        self,
        source_text: str,
        hints: CompileHints | None = None,
    ) -> EOMDocument:
        hints = hints or {}
        source = normalise(source_text)
        document_type = hints.get("document_type", "other")
        render_profile = hints.get("render_profile", "executive_brief")
        budget = RENDER_PROFILES[render_profile]

        paragraphs = _segment_paragraphs(source)
        if not paragraphs:
            # Edge case: empty doc. Synthesise minimal blocks.
            return self._minimal_doc(source, render_profile, budget, document_type)

        blocks: list[Block] = []
        ro = 0

        # Headline: first heading, or first paragraph if no heading.
        first = paragraphs[0]
        if _is_heading(first.text):
            head_text = _strip_md_heading(first.text)
            head_start = first.start + first.text.find(head_text)
            head_end = head_start + len(head_text)
            paragraphs = paragraphs[1:]
        else:
            # Use the first sentence as headline candidate.
            first_sent = first.text.split(". ", 1)[0][:100]
            head_text = first_sent
            head_start = first.start
            head_end = first.start + len(first_sent)
        head_text = head_text[:100]  # H8 cap

        blocks.append(Block(
            id="headline-1", type="headline",
            content=head_text, attention_tier="A",
            priority=1.0, reading_order=ro,
            source_span=SourceSpan(
                start=head_start, end=head_end,
                quote=source[head_start:head_end],
            ),
        ))
        ro += 1

        # Lead: first non-heading paragraph.
        lead_para = next((p for p in paragraphs if not _is_heading(p.text)), None)
        if lead_para is not None:
            lead_text = lead_para.text
            words = lead_text.split()
            if len(words) > 60:
                lead_text = " ".join(words[:60])
            blocks.append(Block(
                id="lead-1", type="lead",
                content=lead_text, attention_tier="A",
                priority=0.95, reading_order=ro,
                source_span=SourceSpan(
                    start=lead_para.start, end=lead_para.start + len(lead_text),
                    quote=source[lead_para.start : lead_para.start + len(lead_text)],
                ),
            ))
            ro += 1
            paragraphs = [p for p in paragraphs if p is not lead_para]
        else:
            # Synthesise a minimal lead.
            blocks.append(Block(
                id="lead-1", type="lead",
                content=head_text[:60],
                attention_tier="A",
                priority=0.95, reading_order=ro,
                source_span=SourceSpan(
                    start=head_start, end=head_end,
                    quote=source[head_start:head_end],
                ),
            ))
            ro += 1

        # Classify remaining paragraphs.
        type_counts = {"claim": 0, "evidence": 0, "factbox": 0,
                       "caveat": 0, "decision": 0, "appendix": 0}
        for i, p in enumerate(paragraphs):
            text = _strip_md_heading(p.text) if _is_heading(p.text) else p.text
            if not text:
                continue

            block_type = self._classify(text, position=i, total=len(paragraphs))
            type_counts[block_type] += 1
            n = type_counts[block_type]
            tier = self._tier_for(block_type, position=i, total=len(paragraphs))
            priority = self._priority_for(block_type, position=i, total=len(paragraphs))

            # Truncate content if it's too large for tier A.
            content = text
            if tier == "A":
                if count_tokens(content) > budget.B_A // 4:
                    content = self._truncate_to_tokens(content, budget.B_A // 4)

            blocks.append(Block(
                id=f"{block_type}-{n}", type=block_type,
                content=content, attention_tier=tier,
                priority=priority, reading_order=ro,
                source_span=SourceSpan(
                    start=p.start, end=p.end,
                    quote=source[p.start : p.end],
                ),
            ))
            ro += 1

        # Enforce H3 caps after the fact: if too many tier A, demote lowest priority.
        blocks = self._enforce_tier_caps(blocks)
        # Ensure H10 budget.
        blocks = self._enforce_token_budget(blocks, budget)

        eom = EOMDocument(
            version="0.1",
            document_type=document_type,
            summary=blocks[1].content[:140] if len(blocks) > 1 else blocks[0].content[:140],
            render_profile=render_profile,
            attention_budget=budget,
            blocks=blocks,
            source=SourceMetadata(
                checksum="sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
                chars=len(source),
                lang="en",
            ),
        )
        return eom

    def _classify(self, text: str, position: int, total: int) -> str:
        if _DECISION_VERBS.search(text) and position <= total // 2:
            return "decision"
        if _HEDGING.search(text):
            return "caveat"
        if len(_NUMBER_RE.findall(text)) >= 3:
            return "factbox"
        if position >= max(2, int(total * 0.7)):
            return "appendix"
        # Long paragraph with no other signal -> evidence; short -> claim.
        if len(text.split()) > 40:
            return "evidence"
        return "claim"

    def _tier_for(self, block_type: str, position: int, total: int) -> str:
        if block_type in ("decision",):
            return "A" if position <= 2 else "B"
        if block_type == "factbox":
            return "A" if position <= 2 else "B"
        if block_type == "caveat":
            return "B"
        if block_type == "appendix":
            return "D"
        if block_type == "evidence":
            return "B" if position <= total * 0.5 else "C"
        return "C"  # default for claim

    def _priority_for(self, block_type: str, position: int, total: int) -> float:
        depth = position / max(1, total)
        if block_type == "headline":
            return 1.0
        if block_type == "lead":
            return 0.95
        return max(0.05, 0.85 - depth * 0.6)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        while words and count_tokens(" ".join(words)) > max_tokens:
            words.pop()
        result = " ".join(words)
        if len(result) < len(text):
            result = result.rstrip(".") + "…"
        return result or text[:60]

    def _enforce_tier_caps(self, blocks: list[Block]) -> list[Block]:
        n = len(blocks)
        cap_a = max(1, int(0.10 * n))
        cap_b = max(2, int(0.25 * n))
        # Sort tier A blocks by priority desc; keep top cap_a, demote rest to B.
        tier_a = sorted(
            [b for b in blocks if b.attention_tier == "A"],
            key=lambda b: -b.priority,
        )
        keep_a = set(b.id for b in tier_a[:cap_a])
        # Same for B.
        new_blocks = []
        for b in blocks:
            if b.attention_tier == "A" and b.id not in keep_a:
                b = b.model_copy(update={"attention_tier": "B"})
            new_blocks.append(b)
        tier_b = sorted(
            [b for b in new_blocks if b.attention_tier == "B"],
            key=lambda b: -b.priority,
        )
        keep_b = set(b.id for b in tier_b[:cap_b])
        final = []
        for b in new_blocks:
            if b.attention_tier == "B" and b.id not in keep_b:
                b = b.model_copy(update={"attention_tier": "C"})
            final.append(b)
        return final

    def _enforce_token_budget(self, blocks: list[Block], budget: AttentionBudget) -> list[Block]:
        """Truncate tier A/B content to fit B_A and B_AB."""
        # Pass 1: enforce B_A by demoting lowest-priority tier-A blocks until under budget.
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
        # Pass 2: enforce B_AB by demoting lowest-priority tier-B blocks.
        while True:
            ab = [b for b in blocks if b.attention_tier in ("A", "B")]
            total = sum(count_tokens(b.content) for b in ab)
            tier_b = [b for b in blocks if b.attention_tier == "B"]
            if total <= budget.B_AB or not tier_b:
                break
            victim = min(tier_b, key=lambda b: b.priority)
            blocks = [
                b.model_copy(update={"attention_tier": "C"}) if b.id == victim.id else b
                for b in blocks
            ]
        return blocks

    def _minimal_doc(
        self,
        source: str,
        render_profile: str,
        budget: AttentionBudget,
        document_type: str,
    ) -> EOMDocument:
        head = Block(
            id="headline-1", type="headline",
            content="(empty document)", attention_tier="A",
            priority=1.0, reading_order=0,
        )
        lead = Block(
            id="lead-1", type="lead",
            content="(empty document)", attention_tier="A",
            priority=0.95, reading_order=1,
        )
        return EOMDocument(
            version="0.1", document_type=document_type,
            summary="empty document",
            render_profile=render_profile, attention_budget=budget,
            blocks=[head, lead],
            source=SourceMetadata(
                checksum="sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
                chars=len(source), lang="en",
            ),
        )
