"""Abstract LLM client + OpenAI-compatible implementations.

The client returns plain text. JSON parsing is the caller's job (the
prompted compiler handles that with explicit error recovery).

Supported providers (all OpenAI-compatible chat-completions):
- OpenRouter (default; routes to Gemma 4 via google/gemma-4-31b-it)
- Together AI (legacy; kept for back-compat)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol

import httpx

DEFAULT_MODEL = "google/gemma-4-31b-it"


@dataclass
class LLMRequest:
    system: str
    user: str
    max_tokens: int = 4096
    temperature: float = 0.0
    model: str = DEFAULT_MODEL
    extra: dict = field(default_factory=dict)


class LLMClient(Protocol):
    def complete(self, req: LLMRequest) -> str: ...


class _OpenAICompatibleClient:
    """Shared OpenAI-compatible chat-completions client.

    Subclasses set `BASE_URL` and `ENV_VAR`.
    """

    BASE_URL: str = ""
    ENV_VAR: str = ""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0):
        self.api_key = api_key or os.environ.get(self.ENV_VAR)
        if not self.api_key:
            raise RuntimeError(f"{self.ENV_VAR} not set")
        self.base_url = base_url or self.BASE_URL
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


class OpenRouterClient(_OpenAICompatibleClient):
    """OpenRouter chat-completions endpoint.

    Set OPENROUTER_API_KEY env var. OpenRouter routes to many backends; pin a
    specific model via LLMRequest.model (e.g. ``google/gemma-4-31b-it`` or
    ``google/gemma-4-31b-it:free``).
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    ENV_VAR = "OPENROUTER_API_KEY"


class TogetherClient(_OpenAICompatibleClient):
    """Together AI chat-completions endpoint.

    Set TOGETHER_API_KEY env var. Kept for back-compat; OpenRouter is the
    project default.
    """

    BASE_URL = "https://api.together.xyz/v1"
    ENV_VAR = "TOGETHER_API_KEY"


class StubLLMClient:
    """Test double: returns a pre-set response, records the request."""

    def __init__(self, response: str):
        self.response = response
        self.last_request: LLMRequest | None = None

    def complete(self, req: LLMRequest) -> str:
        self.last_request = req
        return self.response
