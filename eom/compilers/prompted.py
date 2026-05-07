"""LLM-driven EOM compiler. Uses an injectable LLMClient."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import ValidationError

from eom.compilers.base import CompileHints
from eom.compilers.llm_client import DEFAULT_MODEL, LLMClient, LLMRequest
from eom.compilers.prompt_template import (
    SYSTEM_PROMPT,
    build_user_prompt,
    build_user_prompt_with_spans,
)
from eom.compilers.rules import RulesCompiler
from eom.normalise import normalise
from eom.schema import (
    EOMDocument,
    RENDER_PROFILES,
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = _FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def _format_few_shots(few_shots: Sequence[tuple[str, EOMDocument]]) -> str:
    """Format (source_text, eom) pairs as few-shot exemplars for the prompt."""
    if not few_shots:
        return "(no examples provided)"
    parts = []
    for i, (src, eom) in enumerate(few_shots, start=1):
        eom_json = json.dumps(eom.model_dump(mode="json"), indent=2)
        parts.append(
            f"Example {i}:\nSource:\n<<<\n{src[:1500]}\n>>>\n\nEOM JSON:\n{eom_json}\n"
        )
    return "\n---\n".join(parts)


@dataclass
class PromptedCompiler:
    client: LLMClient
    few_shots: Sequence[tuple[str, EOMDocument]] = field(default_factory=list)
    model: str = DEFAULT_MODEL  # pinned default; OpenRouter routing
    use_scaffolding: bool = True  # pre-extract reference spans and inline them

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

        few_shot_text = _format_few_shots(self.few_shots)
        if self.use_scaffolding:
            from eom.compilers.scaffolding import (
                extract_reference_spans,
                format_spans_for_prompt,
            )
            spans = extract_reference_spans(source)
            user = build_user_prompt_with_spans(
                source_text=source,
                document_type=document_type,
                render_profile=render_profile,
                few_shots=few_shot_text,
                spans_menu=format_spans_for_prompt(spans),
            )
        else:
            user = build_user_prompt(
                source_text=source,
                document_type=document_type,
                render_profile=render_profile,
                few_shots=few_shot_text,
            )
        req = LLMRequest(
            system=SYSTEM_PROMPT,
            user=user,
            model=self.model,
            max_tokens=4096,
            temperature=0.0,
        )
        try:
            raw = self.client.complete(req)
        except Exception:
            return self._fallback(source, hints)

        body = _strip_fences(raw)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return self._fallback(source, hints)

        # Force source metadata to match the actual normalised source.
        payload["source"] = {
            "checksum": "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "chars": len(source),
            "lang": payload.get("source", {}).get("lang", "en"),
        }
        # Force render profile and budget to match hints if provided.
        payload["render_profile"] = render_profile
        payload["attention_budget"] = {"B_A": budget.B_A, "B_AB": budget.B_AB}

        try:
            eom = EOMDocument.model_validate(payload)
        except ValidationError:
            return self._fallback(source, hints)

        # Post-process: deterministically enforce H3 tier caps and H9/H10 token
        # budgets. Teacher LLMs frequently over-tag importance; demoting by
        # priority is harmless when the priorities themselves are reasonable
        # and converts H3/H9/H10 from harness failures into clean fixups.
        from eom.compilers.post_process import enforce_tier_caps, enforce_token_budget
        adjusted = enforce_tier_caps(list(eom.blocks))
        adjusted = enforce_token_budget(adjusted, eom.attention_budget)
        if adjusted != list(eom.blocks):
            eom = eom.model_copy(update={"blocks": adjusted})
        return eom

    def _fallback(self, source: str, hints: CompileHints) -> EOMDocument:
        rules = RulesCompiler()
        return rules.compile(source, hints)
