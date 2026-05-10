"""EOM Pydantic schema.

v0.1 is the original; v0.2 is strictly additive — every v0.1 document is a
valid v0.2 document. New v0.2 fields are optional with v0.1-equivalent
defaults. See `docs/SPEC-v0.2.md`.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Literal, Mapping

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


BLOCK_TYPES = (
    "headline", "lead", "claim", "evidence",
    "factbox", "caveat", "decision", "appendix",
)

BlockType = Literal[
    "headline", "lead", "claim", "evidence",
    "factbox", "caveat", "decision", "appendix",
]
AttentionTier = Literal["A", "B", "C", "D"]

_ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")


class AttentionBudget(BaseModel):
    """Token budgets enforced by H9 / H10. v0.2 adds optional `token_budget`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    B_A: int = Field(ge=0)
    B_AB: int = Field(ge=0)
    token_budget: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _BAB_at_least_BA(self) -> AttentionBudget:
        if self.B_AB < self.B_A:
            raise ValueError(f"B_AB ({self.B_AB}) must be >= B_A ({self.B_A})")
        return self


RelationType = Literal[
    "supports", "qualifies", "contradicts", "derived_from", "cites", "refines",
]


class Relation(BaseModel):
    """Typed directed edge between blocks (v0.2). Replaces v0.1's untyped
    `inference_basis: [str]` over the migration window — both coexist."""

    model_config = ConfigDict(extra="forbid")

    type: RelationType
    target: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


RoleTag = Literal["ground_truth", "claim", "speculation", "citation"]
EvidenceLayer = Literal["surface", "drill", "archive"]
SystemIntent = Literal["question", "summarize", "extract", "compare", "decide"]
Dialect = Literal["outbound", "inbound"]


class Block(BaseModel):
    """One editorial unit in an EOM document."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: BlockType
    content: str
    attention_tier: AttentionTier
    priority: float = Field(ge=0.0, le=1.0)
    reading_order: int = Field(ge=0)
    source_span: SourceSpan | None = None
    is_inferred: bool = False
    inference_basis: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    relations: list[Relation] = Field(default_factory=list)
    role_tag: RoleTag | None = None
    evidence_layer: EvidenceLayer | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID_RE.fullmatch(v):
            raise ValueError(f"id must be slug (lowercase, alnum, hyphen): {v!r}")
        return v

    @field_validator("content")
    @classmethod
    def _content_non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must be non-empty (and non-whitespace)")
        return v

    @model_validator(mode="after")
    def _check_inference_consistency(self) -> Block:
        if self.is_inferred and self.type not in ("claim", "decision"):
            raise ValueError(
                f"is_inferred=True only allowed for claim/decision, got {self.type}"
            )
        if self.inference_basis and not self.is_inferred:
            raise ValueError(
                "inference_basis is only valid when is_inferred=True"
            )
        return self


DocumentType = Literal[
    "memo", "report", "paper", "transcript", "news", "policy", "other",
]

RenderProfileName = Literal["executive_brief", "analytical_brief"]


class EOMDocument(BaseModel):
    """A complete EOM document. `version` widens to "0.2" for the new
    optional fields (dialect, system_intent). v0.1 docs validate unchanged."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["0.1", "0.2"]
    document_type: DocumentType
    summary: str = Field(min_length=1)
    render_profile: RenderProfileName
    attention_budget: AttentionBudget
    blocks: list[Block] = Field(min_length=1)
    source: SourceMetadata
    dialect: Dialect = "outbound"
    system_intent: SystemIntent | None = None

    @field_validator("summary")
    @classmethod
    def _summary_non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("summary must be non-empty (and non-whitespace)")
        return v


RENDER_PROFILES: Mapping[str, AttentionBudget] = MappingProxyType({
    "executive_brief": AttentionBudget(B_A=200, B_AB=800),
    "analytical_brief": AttentionBudget(B_A=400, B_AB=2000),
})
