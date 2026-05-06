"""Abstract LLM client + Together AI implementation.

The client returns plain text. JSON parsing is the caller's job (the
prompted compiler handles that with explicit error recovery).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass
class LLMRequest:
    system: str
    user: str
    max_tokens: int = 4096
    temperature: float = 0.0
    model: str = "google/gemma-2-27b-it"
    extra: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def complete(self, req: LLMRequest) -> str: ...


class TogetherClient:
    """Together AI's OpenAI-compatible chat completions endpoint.

    Set TOGETHER_API_KEY env var. As of plan-write date, Together hosts the
    Gemma family under model id `google/gemma-2-27b-it`. When Gemma 4 lands
    on Together, update the default in LLMRequest.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise RuntimeError("TOGETHER_API_KEY not set")
        self.base_url = base_url or "https://api.together.xyz/v1"
        self.timeout = timeout

    def complete(self, req: LLMRequest) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": req.model,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            **req.extra,
        }
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"]


class StubLLMClient:
    """Test double: returns a pre-set response, records the request."""

    def __init__(self, response: str):
        self.response = response
        self.last_request: LLMRequest | None = None

    def complete(self, req: LLMRequest) -> str:
        self.last_request = req
        return self.response
