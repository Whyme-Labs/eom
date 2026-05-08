"""Fine-tuned EOM compiler. Loads a Gemma + LoRA adapter via transformers/peft.

Usage:
    export EOM_FINETUNED_CKPT=/path/to/eom-sft-adapter
    from eom.compilers.finetuned import FineTunedCompiler
    c = FineTunedCompiler()
    eom = c.compile(source_text, hints={"document_type": "memo"})
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from eom.compilers.base import CompileHints
from eom.compilers.post_process import enforce_tier_caps, enforce_token_budget
from eom.compilers.prompt_template import SYSTEM_PROMPT, build_user_prompt_with_spans
from eom.compilers.rules import RulesCompiler
from eom.compilers.scaffolding import extract_reference_spans, format_spans_for_prompt
from eom.normalise import normalise
from eom.schema import EOMDocument, RENDER_PROFILES

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


def _strip_fences(s: str) -> str:
    s = s.strip()
    m = _FENCE_RE.match(s)
    return m.group(1).strip() if m else s


@dataclass
class FineTunedCompiler:
    """Loads Gemma base + LoRA adapter; produces EOM JSON.

    The model is loaded lazily in ``__post_init__`` via transformers + peft
    (not Unsloth), so it works on the 1080 Ti without 4-bit bitsandbytes
    support. Use ``base_model_name="unsloth/gemma-2-2b-it-bnb-4bit"`` if you
    want to run with 4-bit quantisation on a compatible GPU.
    """

    checkpoint_path: str | None = None
    base_model_name: str = "google/gemma-3-4b-it"  # full-precision; swap to bnb-4bit variant as needed
    max_new_tokens: int = 2048
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        ckpt = self.checkpoint_path or os.environ.get("EOM_FINETUNED_CKPT")
        if not ckpt:
            raise RuntimeError(
                "EOM_FINETUNED_CKPT env var not set and no checkpoint_path arg provided. "
                "Either run scripts/prepare_sft_dataset.py + train via "
                "notebooks/03-train-sft.ipynb on Kaggle and download the adapter, "
                "or pass checkpoint_path= to the constructor."
            )
        self.checkpoint_path = ckpt
        self._load()

    def _load(self) -> None:
        # Lazy imports — heavy deps; only loaded when this compiler is used
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint_path)
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base, self.checkpoint_path)
        self._model.eval()

    def compile(self, source_text: str, hints: CompileHints | None = None) -> EOMDocument:
        hints = hints or {}
        source = normalise(source_text)
        document_type = hints.get("document_type", "other")
        render_profile = hints.get("render_profile", "executive_brief")
        budget = RENDER_PROFILES[render_profile]

        spans = extract_reference_spans(source)
        user = build_user_prompt_with_spans(
            source_text=source,
            document_type=document_type,
            render_profile=render_profile,
            few_shots="(none — fine-tuned model)",
            spans_menu=format_spans_for_prompt(spans),
        )
        prompt = f"{SYSTEM_PROMPT}\n\n{user}"

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        import torch

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        generated = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        body = _strip_fences(generated)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return self._fallback(source, hints)

        # Force source metadata + render profile / budget like PromptedCompiler does
        payload["source"] = {
            "checksum": "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "chars": len(source),
            "lang": payload.get("source", {}).get("lang", "en"),
        }
        payload["render_profile"] = render_profile
        payload["attention_budget"] = {"B_A": budget.B_A, "B_AB": budget.B_AB}

        try:
            eom = EOMDocument.model_validate(payload)
        except ValidationError:
            return self._fallback(source, hints)

        # Same post-processor pass as PromptedCompiler
        adjusted = enforce_tier_caps(list(eom.blocks))
        adjusted = enforce_token_budget(adjusted, eom.attention_budget)
        if adjusted != list(eom.blocks):
            eom = eom.model_copy(update={"blocks": adjusted})
        return eom

    def _fallback(self, source: str, hints: CompileHints) -> EOMDocument:
        return RulesCompiler().compile(source, hints)
