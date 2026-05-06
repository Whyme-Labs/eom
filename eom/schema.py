"""EOM Pydantic schema (v0.1)."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_LANG_RE = re.compile(r"^[a-z]{2}$")


class SourceSpan(BaseModel):
    """A character-offset range into the normalised source_text."""

    model_config = ConfigDict(extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def _end_after_start(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self


class SourceMetadata(BaseModel):
    """Metadata about the source document (post-normalisation)."""

    model_config = ConfigDict(extra="forbid")

    checksum: str = Field(min_length=1)
    chars: int = Field(ge=0)
    lang: str  # ISO-639-1, exactly 2 lowercase letters

    @field_validator("lang")
    @classmethod
    def _check_lang(cls, v: str) -> str:
        if not _LANG_RE.fullmatch(v):
            raise ValueError(
                f"lang must be ISO-639-1 (2 lowercase ASCII letters), got: {v!r}"
            )
        return v
