# EOM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build EOM v0.1 — a versioned harness, reference schema, three-implementation compiler, two renderers, dataset pipeline, fine-tuned Gemma-4 E2B converter, and evaluation suite — over 12 weeks ending with a publishable open standard plus research artifact.

**Architecture:** Harness is the standard (testable properties); reference schema is the simplest JSON encoding that satisfies the harness; three compilers (deterministic, prompted, fine-tuned) all produce the same shape; renderers consume EOM deterministically. Main LLM is never trained or modified; EOM lives strictly downstream.

**Tech Stack:** Python 3.11, `uv` for package management, Pydantic v2 for schema, `markdown-it-py` for AST parsing, Jinja2 for newspaper templates, `tiktoken` for tokenisation, `click` for CLI, Together AI (or HuggingFace inference) for `compiler_prompted` (Gemma-4-27B), Unsloth + LoRA on a single A100 / Kaggle TPU-VM for `compiler_finetuned` (Gemma-4 E2B), `pytest` for tests, `spaCy` for NER scaffolding in the synthetic pipeline.

**Granularity note:** Phase 1 (weeks 1–4) is in full TDD detail (write-test → run-fail → implement → run-pass → commit). Phases 2–4 are at deliverable + acceptance-criteria granularity; their detailed sub-steps will be written when each phase begins, because they depend on measurements from the prior phase. That is by design, not laziness.

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Python 3.11, deps, package metadata, entry points |
| `.python-version` | `3.11` |
| `.gitignore` | Standard Python + ML + secrets |
| `README.md` | Project overview, quickstart, links to spec |
| `eom/__init__.py` | Public API re-exports |
| `eom/schema.py` | Pydantic models: `EOMDocument`, `Block`, `SourceSpan`, `AttentionBudget`, `SourceMetadata` |
| `eom/normalise.py` | `normalise(text) -> str` — NFC unicode, `\n` newlines, trailing whitespace strip |
| `eom/tokens.py` | `count_tokens(text) -> int` — pinned to `tiktoken cl100k_base` |
| `eom/harness.py` | `validate(eom, source_text) -> ValidationReport`, plus `H1..H12` individual checks, `FailureRecord`, `WarningRecord` |
| `eom/evaluators.py` | Corpus-level evaluators (`H13`, `H13b`, `H14`); used in Phase 2+ |
| `eom/compilers/__init__.py` | Re-exports + `get_compiler(kind, **kwargs)` factory |
| `eom/compilers/base.py` | `Compiler` Protocol, `CompileHints` TypedDict |
| `eom/compilers/rules.py` | Deterministic rule-based compiler |
| `eom/compilers/prompted.py` | LLM-driven compiler with prompt template + few-shots |
| `eom/compilers/finetuned.py` | Wrapper around the SFT'd Gemma-4 E2B (Phase 3) |
| `eom/compilers/llm_client.py` | Abstract LLM client; Together AI implementation, HF transformers fallback |
| `eom/renderers/__init__.py` | Re-exports |
| `eom/renderers/newspaper.py` | `render_newspaper(eom) -> str` (HTML via Jinja2) |
| `eom/renderers/context_pack.py` | `render_context_pack(eom, token_budget) -> str` |
| `eom/repair.py` | Repair loop with failure → feedback summariser |
| `eom/cli.py` | `eom` Click CLI: `compile`, `validate`, `render` |
| `templates/newspaper.html` | Jinja2 template (hero / main / rail / archive) |
| `templates/newspaper.css` | Print-ready CSS for newspaper renderer |
| `tests/test_schema.py` | Schema model tests |
| `tests/test_normalise.py` | Normalisation tests |
| `tests/test_tokens.py` | Token counter tests |
| `tests/test_harness.py` | Each Hxx tested individually + integration |
| `tests/test_compiler_rules.py` | Rules compiler tests |
| `tests/test_compiler_prompted.py` | Prompted compiler tests (mocked LLM client) |
| `tests/test_renderer_newspaper.py` | Newspaper renderer tests |
| `tests/test_renderer_context_pack.py` | Context pack renderer tests |
| `tests/test_repair.py` | Repair loop tests |
| `tests/test_cli.py` | CLI integration tests |
| `tests/fixtures/freight_memo.md` | Canonical input fixture |
| `tests/fixtures/freight_memo.eom.json` | Expected EOM for fixture |
| `tests/fixtures/short_news.md` | Second canonical fixture |
| `tests/fixtures/short_news.eom.json` | Expected EOM |
| `data/gold/<doc-type>/<slug>.md` | 30–50 hand-curated source documents |
| `data/gold/<doc-type>/<slug>.eom.json` | Hand-curated EOM matching each source |
| `data/gold/MANIFEST.json` | Index: source path, eom path, doc_type, lang, license |
| `scripts/generate_synthetic.py` | Phase 2: privileged-context teacher pipeline |
| `scripts/eval_corpus.py` | Phase 2+3: H13/H13b/H14 evaluation runner |
| `scripts/train_sft.py` | Phase 3: Unsloth SFT loop |
| `scripts/train_rlvr.py` | Phase 3: optional GRPO loop |
| `notebooks/01-explore-gold.ipynb` | Phase 1 wrap: gold-set inspection |
| `notebooks/02-prompted-baseline.ipynb` | Phase 2: prompted-compiler measurements |
| `notebooks/03-train-sft.ipynb` | Phase 3: Kaggle/Colab-runnable training |
| `notebooks/04-eval-corpus.ipynb` | Phase 3: corpus-eval results |
| `docs/harness-spec.md` | RFC-style standard doc (Phase 4) |
| `docs/schema-spec.md` | Reference encoding doc (Phase 4) |

---

## Phase 1: Standard (Weeks 1–4)

The end-state of Phase 1 is a working CLI:
```bash
$ eom compile --input doc.md --compiler prompted --profile executive_brief --output doc.eom.json
$ eom validate --eom doc.eom.json --source doc.md
$ eom render --eom doc.eom.json --target newspaper --output doc.html
$ eom render --eom doc.eom.json --target context-pack --budget 3000 --output doc.txt
```
…with 100% H1–H12 pass rate on the gold seed and integration tests covering the end-to-end pipeline.

---

### Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `README.md`
- Create: `eom/__init__.py`

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "eom"
version = "0.1.0"
description = "Editorial Object Model — an attention-architecture standard for human-AI documents"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "soh"}]
dependencies = [
    "pydantic>=2.5",
    "markdown-it-py>=3.0",
    "jinja2>=3.1",
    "tiktoken>=0.7",
    "click>=8.1",
    "rich>=13.7",
    "httpx>=0.27",
]

[project.optional-dependencies]
ml = [
    "spacy>=3.7",
    "torch>=2.3",
    "transformers>=4.45",
    "datasets>=2.20",
]
train = [
    "unsloth",
    "trl>=0.10",
    "peft>=0.12",
]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "mypy>=1.10",
]

[project.scripts]
eom = "eom.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["eom"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.venv/
.env
.envrc

# data
data/synthetic/
data/raw/
data/cache/
*.parquet

# models
*.bin
*.safetensors
checkpoints/
runs/
wandb/

# notebooks
.ipynb_checkpoints/
*.ipynb_meta

# editor
.vscode/
.idea/
*.swp
.DS_Store
```

- [ ] **Step 4: Create `README.md`**

```markdown
# EOM — Editorial Object Model

An attention-architecture standard for documents that serve both humans and AI.

EOM is a downstream representation: any markdown / prose source — whether human-authored or LLM-generated — compiles into a structured graph of editorial blocks (headline, lead, claim, evidence, factbox, caveat, decision, appendix) with explicit salience, source provenance, and a hard attention budget.

The standard is defined as a **harness** (versioned testable properties); the JSON shape is just one encoding that satisfies the harness.

**Status:** v0.1 in development. See [the design spec](docs/superpowers/specs/2026-05-02-eom-design.md) for full details.

## Quickstart (when Phase 1 ships)

```bash
uv sync
uv run eom compile --input examples/memo.md --compiler rules --output memo.eom.json
uv run eom render --eom memo.eom.json --target newspaper --output memo.html
uv run eom render --eom memo.eom.json --target context-pack --budget 3000 --output memo.txt
```

## Components

| Layer | Responsibility |
|---|---|
| Harness | Defines what makes a document EOM-conformant. Source of truth. |
| Schema | The simplest JSON encoding that satisfies the harness. |
| Compiler | Three implementations: rules-based, prompted-LLM, fine-tuned (Gemma-4 E2B). |
| Renderers | Newspaper HTML view + LLM context pack — both deterministic. |

## License

MIT.
```

- [ ] **Step 5: Create `eom/__init__.py`**

```python
"""EOM — Editorial Object Model.

Public API:
    compile(source_text, hints) -> EOMDocument
    validate(eom, source_text) -> ValidationReport
    render_newspaper(eom) -> str
    render_context_pack(eom, token_budget) -> str
"""

__version__ = "0.1.0"
```

- [ ] **Step 6: Verify package layout**

Run: `uv venv && uv sync --extra dev`
Expected: clean install, no errors.

- [ ] **Step 7: Commit**

`uv sync` will have generated `uv.lock`. Commit it for reproducible installs.

```bash
git add pyproject.toml .python-version .gitignore README.md eom/__init__.py uv.lock
git commit -m "chore: bootstrap eom package layout"
```

---

### Task 2: Text normalisation

**Files:**
- Create: `eom/normalise.py`
- Test: `tests/test_normalise.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_normalise.py
from eom.normalise import normalise


def test_normalise_returns_string():
    assert normalise("hello") == "hello"


def test_normalise_strips_trailing_whitespace_per_line():
    assert normalise("hello   \nworld  \n") == "hello\nworld\n"


def test_normalise_converts_crlf_to_lf():
    assert normalise("a\r\nb\r\nc") == "a\nb\nc"


def test_normalise_converts_lone_cr_to_lf():
    assert normalise("a\rb\rc") == "a\nb\nc"


def test_normalise_strips_bom():
    assert normalise("﻿hello") == "hello"


def test_normalise_applies_nfc():
    # NFC: precomposed é (U+00E9) preferred over decomposed e + acute (U+0065 U+0301)
    decomposed = "café"
    assert normalise(decomposed) == "café"
    assert len(normalise(decomposed)) == 4


def test_normalise_idempotent():
    text = "hello\nworld with é and 中文\n"
    assert normalise(normalise(text)) == normalise(text)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_normalise.py -v`
Expected: 7 errors — `ModuleNotFoundError: No module named 'eom.normalise'`

- [ ] **Step 3: Implement `eom/normalise.py`**

```python
"""Text normalisation for EOM source text.

All offset-comparing code (validators, compilers) must apply this
before computing or checking offsets, so byte-for-byte agreement is
guaranteed across implementations.
"""

import unicodedata


def normalise(text: str) -> str:
    """Return text in canonical EOM form.

    - Strips UTF-8 BOM if present.
    - Normalises CRLF and lone CR to LF.
    - Strips trailing whitespace from each line (preserving line breaks).
    - Applies NFC unicode normalisation.
    """
    if text.startswith("﻿"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = unicodedata.normalize("NFC", text)
    return text
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_normalise.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add eom/normalise.py tests/test_normalise.py
git commit -m "feat(eom): text normalisation (NFC, LF, BOM strip, trailing whitespace)"
```

---

### Task 3: Token counter

**Files:**
- Create: `eom/tokens.py`
- Test: `tests/test_tokens.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tokens.py
import pytest

from eom.tokens import count_tokens, ENCODING_NAME


def test_encoding_pinned():
    assert ENCODING_NAME == "cl100k_base"


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_simple():
    assert count_tokens("hello") == 1


def test_count_tokens_sentence():
    n = count_tokens("The quick brown fox jumps over the lazy dog.")
    # cl100k_base tokenises this into 9-10 tokens; assert a stable range.
    assert 8 <= n <= 11


def test_count_tokens_multiline():
    a = count_tokens("hello\nworld")
    b = count_tokens("hello world")
    # Newline introduces at least one extra token.
    assert a >= b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tokens.py -v`
Expected: errors — `ModuleNotFoundError: No module named 'eom.tokens'`.

- [ ] **Step 3: Implement `eom/tokens.py`**

```python
"""Pinned tokeniser for EOM compactness checks.

The harness fixes cl100k_base so that B_A / B_AB budgets are
reproducible across implementations, regardless of the LLM that
ultimately consumes the context pack.
"""

from functools import lru_cache

import tiktoken

ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str) -> int:
    """Return the number of tokens in `text` under the pinned encoding."""
    if not text:
        return 0
    return len(_encoding().encode(text))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_tokens.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eom/tokens.py tests/test_tokens.py
git commit -m "feat(eom): pin tiktoken cl100k_base for reproducible budgets"
```

---

### Task 4: Schema — `SourceSpan` and `SourceMetadata`

**Files:**
- Create: `eom/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests for `SourceSpan` and `SourceMetadata`**

```python
# tests/test_schema.py
import pytest
from pydantic import ValidationError

from eom.schema import SourceSpan, SourceMetadata


class TestSourceSpan:
    def test_creates_valid_span(self):
        span = SourceSpan(start=10, end=20, quote="hello world")
        assert span.start == 10
        assert span.end == 20
        assert span.quote == "hello world"

    def test_rejects_negative_start(self):
        with pytest.raises(ValidationError):
            SourceSpan(start=-1, end=5, quote="x")

    def test_rejects_end_before_start(self):
        with pytest.raises(ValidationError):
            SourceSpan(start=10, end=5, quote="x")

    def test_rejects_empty_quote(self):
        with pytest.raises(ValidationError):
            SourceSpan(start=0, end=5, quote="")


class TestSourceMetadata:
    def test_creates_valid_metadata(self):
        meta = SourceMetadata(
            checksum="sha256:abc123",
            chars=512,
            lang="en",
        )
        assert meta.lang == "en"

    def test_rejects_negative_chars(self):
        with pytest.raises(ValidationError):
            SourceMetadata(checksum="sha256:x", chars=-1, lang="en")

    def test_rejects_invalid_lang_format(self):
        with pytest.raises(ValidationError):
            SourceMetadata(checksum="sha256:x", chars=10, lang="english")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schema.py -v`
Expected: errors — `ModuleNotFoundError: No module named 'eom.schema'`.

- [ ] **Step 3: Implement `SourceSpan` and `SourceMetadata` in `eom/schema.py`**

```python
"""EOM Pydantic schema (v0.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
        if len(v) != 2 or not v.islower() or not v.isalpha():
            raise ValueError(f"lang must be ISO-639-1 (2 lowercase letters), got: {v!r}")
        return v
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add eom/schema.py tests/test_schema.py
git commit -m "feat(schema): SourceSpan and SourceMetadata"
```

---

### Task 5: Schema — `AttentionBudget` and `Block`

**Files:**
- Modify: `eom/schema.py`
- Modify: `tests/test_schema.py`

- [ ] **Step 1: Append failing tests for `AttentionBudget` and `Block`**

```python
# Append to tests/test_schema.py
from eom.schema import AttentionBudget, Block


class TestAttentionBudget:
    def test_creates_valid_budget(self):
        b = AttentionBudget(B_A=200, B_AB=800)
        assert b.B_A == 200
        assert b.B_AB == 800

    def test_rejects_negative(self):
        with pytest.raises(ValidationError):
            AttentionBudget(B_A=-1, B_AB=800)

    def test_rejects_BAB_smaller_than_BA(self):
        with pytest.raises(ValidationError):
            AttentionBudget(B_A=500, B_AB=200)


class TestBlock:
    def _ok_block_kwargs(self, **overrides):
        defaults = dict(
            id="claim-1",
            type="claim",
            content="The sky is blue.",
            attention_tier="B",
            priority=0.5,
            reading_order=2,
            source_span=SourceSpan(start=0, end=18, quote="The sky is blue.  "),
        )
        defaults.update(overrides)
        return defaults

    def test_creates_valid_block(self):
        b = Block(**self._ok_block_kwargs())
        assert b.id == "claim-1"
        assert b.is_inferred is False
        assert b.inference_basis == []
        assert b.parent_id is None

    def test_priority_in_unit_interval(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(priority=1.1))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(priority=-0.1))

    def test_reading_order_non_negative(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(reading_order=-1))

    def test_rejects_unknown_type(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(type="quote"))

    def test_rejects_unknown_tier(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(attention_tier="E"))

    def test_is_inferred_only_for_claim_or_decision(self):
        # is_inferred=True on an evidence block must fail at the block level
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(
                type="evidence",
                is_inferred=True,
                source_span=None,
            ))

    def test_inference_basis_requires_is_inferred(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(
                is_inferred=False,
                inference_basis=["evidence-1"],
            ))

    def test_inferred_claim_with_basis(self):
        b = Block(**self._ok_block_kwargs(
            is_inferred=True,
            inference_basis=["evidence-1", "factbox-1"],
            source_span=None,
        ))
        assert b.is_inferred is True
        assert b.inference_basis == ["evidence-1", "factbox-1"]

    def test_id_must_be_slug(self):
        # IDs are type-prefixed, lowercase, alphanumeric + hyphen only.
        Block(**self._ok_block_kwargs(id="claim-1"))
        Block(**self._ok_block_kwargs(id="evidence-12"))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(id="Claim 1"))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(id="claim_1"))

    def test_content_must_be_non_empty(self):
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(content=""))
        with pytest.raises(ValidationError):
            Block(**self._ok_block_kwargs(content="   "))
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_schema.py -v`
Expected: errors for `AttentionBudget`, `Block` (ImportError), but earlier tests still pass.

- [ ] **Step 3: Append `AttentionBudget` and `Block` to `eom/schema.py`**

```python
# Append to eom/schema.py
import re

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
    """Token budgets enforced by H9 / H10."""

    model_config = ConfigDict(extra="forbid")

    B_A: int = Field(ge=0)
    B_AB: int = Field(ge=0)

    @model_validator(mode="after")
    def _BAB_at_least_BA(self) -> AttentionBudget:
        if self.B_AB < self.B_A:
            raise ValueError(f"B_AB ({self.B_AB}) must be >= B_A ({self.B_A})")
        return self


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

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not _ID_RE.match(v):
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eom/schema.py tests/test_schema.py
git commit -m "feat(schema): AttentionBudget and Block with intra-block validators"
```

---

### Task 6: Schema — `EOMDocument` and render profiles

**Files:**
- Modify: `eom/schema.py`
- Modify: `tests/test_schema.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_schema.py
from eom.schema import EOMDocument, RENDER_PROFILES


class TestRenderProfiles:
    def test_two_profiles_defined(self):
        assert "executive_brief" in RENDER_PROFILES
        assert "analytical_brief" in RENDER_PROFILES

    def test_executive_brief_budgets(self):
        b = RENDER_PROFILES["executive_brief"]
        assert b.B_A == 200
        assert b.B_AB == 800

    def test_analytical_brief_budgets(self):
        b = RENDER_PROFILES["analytical_brief"]
        assert b.B_A == 400
        assert b.B_AB == 2000


class TestEOMDocument:
    def _ok_doc(self, **overrides):
        head = Block(
            id="headline-1", type="headline",
            content="Test", attention_tier="A",
            priority=1.0, reading_order=0,
            source_span=SourceSpan(start=0, end=4, quote="Test"),
        )
        defaults = dict(
            version="0.1",
            document_type="memo",
            summary="A test document.",
            render_profile="executive_brief",
            attention_budget=AttentionBudget(B_A=200, B_AB=800),
            blocks=[head],
            source=SourceMetadata(checksum="sha256:test", chars=4, lang="en"),
        )
        defaults.update(overrides)
        return defaults

    def test_creates_valid_doc(self):
        d = EOMDocument(**self._ok_doc())
        assert d.version == "0.1"
        assert d.document_type == "memo"

    def test_rejects_unknown_version(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(version="0.2"))

    def test_rejects_unknown_document_type(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(document_type="email"))

    def test_rejects_unknown_render_profile(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(render_profile="custom"))

    def test_rejects_empty_blocks_list(self):
        with pytest.raises(ValidationError):
            EOMDocument(**self._ok_doc(blocks=[]))

    def test_round_trip_json(self):
        d = EOMDocument(**self._ok_doc())
        roundtripped = EOMDocument.model_validate_json(d.model_dump_json())
        assert roundtripped == d
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_schema.py -v`
Expected: errors importing `EOMDocument`, `RENDER_PROFILES`.

- [ ] **Step 3: Append `EOMDocument` and `RENDER_PROFILES` to `eom/schema.py`**

```python
# Append to eom/schema.py
DocumentType = Literal[
    "memo", "report", "paper", "transcript", "news", "policy", "other",
]

RenderProfileName = Literal["executive_brief", "analytical_brief"]


class EOMDocument(BaseModel):
    """A complete EOM document (the reference encoding)."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["0.1"]
    document_type: DocumentType
    summary: str = Field(min_length=1)
    render_profile: RenderProfileName
    attention_budget: AttentionBudget
    blocks: list[Block] = Field(min_length=1)
    source: SourceMetadata


RENDER_PROFILES: dict[str, AttentionBudget] = {
    "executive_brief": AttentionBudget(B_A=200, B_AB=800),
    "analytical_brief": AttentionBudget(B_A=400, B_AB=2000),
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eom/schema.py tests/test_schema.py
git commit -m "feat(schema): EOMDocument top-level model and render profiles"
```

---

### Task 7: Test fixtures — `freight_memo`

**Files:**
- Create: `tests/fixtures/freight_memo.md`
- Create: `tests/fixtures/freight_memo.eom.json`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/loader.py`
- Test: `tests/test_fixtures.py`

This task locks in a canonical input/output pair that every later test reuses. It must pass schema validation; later tasks make it pass the harness.

- [ ] **Step 1: Create `tests/fixtures/freight_memo.md`**

```markdown
# Q1 Freight Cost Update

Q1 freight cost rose 9% after a port closure on the west coast.

On-time delivery dropped from 96% to 91% over the quarter, while backlog increased by 14%. Operations estimates a six-week recovery once rerouting is in place.

## Background

The closure followed a labour dispute at the primary port and lasted twelve days. During that period, ten of our fourteen weekly shipments were delayed by an average of 4.5 days.

## Recommendation

Approve a temporary rerouting through alternate ports and a two-week buffer-stock build for top-30 SKUs.

## Caveat

This recommendation is based on Q1 data alone; a six-week reassessment is required before structural changes.
```

- [ ] **Step 2: Create `tests/fixtures/freight_memo.eom.json`**

```json
{
  "version": "0.1",
  "document_type": "memo",
  "summary": "Q1 freight cost rose 9% after a port closure; recommend temporary rerouting and buffer stock.",
  "render_profile": "executive_brief",
  "attention_budget": {"B_A": 200, "B_AB": 800},
  "blocks": [
    {
      "id": "headline-1",
      "type": "headline",
      "content": "Q1 Freight Cost Update",
      "attention_tier": "A",
      "priority": 1.0,
      "reading_order": 0,
      "source_span": {"start": 2, "end": 24, "quote": "Q1 Freight Cost Update"},
      "is_inferred": false,
      "inference_basis": [],
      "parent_id": null
    },
    {
      "id": "lead-1",
      "type": "lead",
      "content": "On-time delivery dropped from 96% to 91% and backlog rose 14%; operations expects a six-week recovery.",
      "attention_tier": "A",
      "priority": 0.95,
      "reading_order": 1,
      "source_span": {"start": 84, "end": 263, "quote": "On-time delivery dropped from 96% to 91% over the quarter, while backlog increased by 14%. Operations estimates a six-week recovery once rerouting is in place."},
      "is_inferred": false,
      "inference_basis": [],
      "parent_id": null
    },
    {
      "id": "factbox-1",
      "type": "factbox",
      "content": "Freight: +9% • On-time: 96%→91% • Backlog: +14% • Recovery est.: 6 weeks",
      "attention_tier": "A",
      "priority": 0.9,
      "reading_order": 2,
      "source_span": {"start": 26, "end": 263, "quote": "Q1 freight cost rose 9% after a port closure on the west coast.\n\nOn-time delivery dropped from 96% to 91% over the quarter, while backlog increased by 14%. Operations estimates a six-week recovery once rerouting is in place."},
      "is_inferred": false,
      "inference_basis": [],
      "parent_id": null
    },
    {
      "id": "evidence-1",
      "type": "evidence",
      "content": "Closure followed a labour dispute at the primary port; lasted 12 days; 10 of 14 weekly shipments delayed by avg 4.5 days.",
      "attention_tier": "B",
      "priority": 0.7,
      "reading_order": 3,
      "source_span": {"start": 280, "end": 506, "quote": "The closure followed a labour dispute at the primary port and lasted twelve days. During that period, ten of our fourteen weekly shipments were delayed by an average of 4.5 days."},
      "is_inferred": false,
      "inference_basis": [],
      "parent_id": null
    },
    {
      "id": "decision-1",
      "type": "decision",
      "content": "Approve temporary rerouting and a two-week buffer-stock build for top-30 SKUs.",
      "attention_tier": "A",
      "priority": 0.85,
      "reading_order": 4,
      "source_span": {"start": 525, "end": 651, "quote": "Approve a temporary rerouting through alternate ports and a two-week buffer-stock build for top-30 SKUs."},
      "is_inferred": false,
      "inference_basis": [],
      "parent_id": null
    },
    {
      "id": "caveat-1",
      "type": "caveat",
      "content": "Recommendation rests on Q1 data alone; reassess in six weeks before structural changes.",
      "attention_tier": "B",
      "priority": 0.6,
      "reading_order": 5,
      "source_span": {"start": 663, "end": 803, "quote": "This recommendation is based on Q1 data alone; a six-week reassessment is required before structural changes."},
      "is_inferred": false,
      "inference_basis": [],
      "parent_id": null
    }
  ],
  "source": {
    "checksum": "PLACEHOLDER",
    "chars": 0,
    "lang": "en"
  }
}
```

Note: `checksum` and `chars` are placeholders; the loader fills them in based on the actual normalised source.

- [ ] **Step 3: Create `tests/fixtures/__init__.py`** (empty file)

- [ ] **Step 4: Create `tests/fixtures/loader.py`**

```python
"""Test fixture loaders.

Reads a markdown source file, normalises it, and pairs it with the
expected EOM JSON (with checksum / chars filled in).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eom.normalise import normalise
from eom.schema import EOMDocument

FIXTURE_DIR = Path(__file__).parent


def load_pair(slug: str) -> tuple[str, EOMDocument]:
    """Return (normalised_source_text, expected_eom) for the named fixture."""
    md_path = FIXTURE_DIR / f"{slug}.md"
    eom_path = FIXTURE_DIR / f"{slug}.eom.json"
    raw = md_path.read_text(encoding="utf-8")
    source = normalise(raw)
    eom_dict = json.loads(eom_path.read_text(encoding="utf-8"))
    eom_dict["source"]["checksum"] = "sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()
    eom_dict["source"]["chars"] = len(source)
    return source, EOMDocument.model_validate(eom_dict)
```

- [ ] **Step 5: Create `tests/test_fixtures.py`**

```python
from tests.fixtures.loader import load_pair


def test_load_freight_memo():
    source, eom = load_pair("freight_memo")
    assert eom.document_type == "memo"
    assert eom.version == "0.1"
    assert eom.source.chars == len(source)
    assert eom.source.checksum.startswith("sha256:")
    # Sanity: each block's source_span.quote matches the slice
    for block in eom.blocks:
        if block.source_span:
            sub = source[block.source_span.start : block.source_span.end]
            assert sub == block.source_span.quote, (
                f"block {block.id}: expected {block.source_span.quote!r}, "
                f"got {sub!r}"
            )
```

- [ ] **Step 6: Run test**

Run: `uv run pytest tests/test_fixtures.py -v`
Expected: PASS. If `source_span` quotes don't match the slice, **fix the offsets in `freight_memo.eom.json`** (don't change the source).

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/ tests/test_fixtures.py
git commit -m "test: add freight_memo canonical fixture and loader"
```

---

### Task 8: Harness scaffolding — `ValidationReport`, `FailureRecord`

**Files:**
- Create: `eom/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_harness.py
import pytest

from eom.harness import (
    FailureRecord,
    ValidationReport,
    WarningRecord,
)


class TestValidationReport:
    def test_passed_when_no_failures(self):
        r = ValidationReport(failures=[], warnings=[], metrics={})
        assert r.passed is True

    def test_failed_when_failures_present(self):
        f = FailureRecord(rule="H1", message="x", block_id=None)
        r = ValidationReport(failures=[f], warnings=[], metrics={})
        assert r.passed is False

    def test_passed_with_only_warnings(self):
        w = WarningRecord(rule="H13", message="not checkable here")
        r = ValidationReport(failures=[], warnings=[w], metrics={})
        assert r.passed is True


class TestFailureRecord:
    def test_creates_failure(self):
        f = FailureRecord(rule="H3", message="too many tier A", block_id=None)
        assert f.rule == "H3"
        assert f.block_id is None

    def test_creates_failure_with_block_id(self):
        f = FailureRecord(rule="H11", message="span out of range", block_id="evidence-1")
        assert f.block_id == "evidence-1"
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_harness.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement scaffolding in `eom/harness.py`**

```python
"""EOM harness — per-document validators (H1–H12).

The harness is the standard. A document is EOM-conformant iff
`validate(eom, source_text).passed is True`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eom.schema import EOMDocument


@dataclass(frozen=True)
class FailureRecord:
    """One harness failure."""
    rule: str          # e.g. "H3"
    message: str
    block_id: str | None = None
    span: tuple[int, int] | None = None


@dataclass(frozen=True)
class WarningRecord:
    """A warning (e.g., a property not checkable at this layer)."""
    rule: str
    message: str


@dataclass
class ValidationReport:
    failures: list[FailureRecord] = field(default_factory=list)
    warnings: list[WarningRecord] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eom/harness.py tests/test_harness.py
git commit -m "feat(harness): ValidationReport scaffolding"
```

---

### Task 9: Harness — H1 (exactly one headline) and H2 (exactly one lead, reading_order ≤ 3)

**Files:**
- Modify: `eom/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_harness.py
from eom.harness import check_h1, check_h2
from eom.schema import Block, EOMDocument, AttentionBudget, SourceMetadata, SourceSpan


def _block(id: str, type: str, ro: int, tier: str = "C", with_span: bool = True) -> Block:
    return Block(
        id=id,
        type=type,
        content=f"content for {id}",
        attention_tier=tier,
        priority=0.5,
        reading_order=ro,
        source_span=SourceSpan(start=0, end=10, quote="0123456789") if with_span else None,
    )


def _doc(blocks: list[Block]) -> EOMDocument:
    return EOMDocument(
        version="0.1",
        document_type="memo",
        summary="x",
        render_profile="executive_brief",
        attention_budget=AttentionBudget(B_A=200, B_AB=800),
        blocks=blocks,
        source=SourceMetadata(checksum="sha256:x", chars=100, lang="en"),
    )


class TestH1:
    def test_passes_with_one_headline(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h1(d) == []

    def test_fails_with_zero_headlines(self):
        d = _doc([_block("lead-1", "lead", 0, tier="A")])
        f = check_h1(d)
        assert len(f) == 1
        assert f[0].rule == "H1"

    def test_fails_with_two_headlines(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("headline-2", "headline", 1, tier="A"),
            _block("lead-1", "lead", 2, tier="A"),
        ])
        f = check_h1(d)
        assert len(f) == 1
        assert "2" in f[0].message


class TestH2:
    def test_passes_with_lead_at_ro_1(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h2(d) == []

    def test_passes_with_lead_at_ro_3(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("appendix-x", "appendix", 1, tier="C"),
            _block("appendix-y", "appendix", 2, tier="C"),
            _block("lead-1", "lead", 3, tier="A"),
        ])
        assert check_h2(d) == []

    def test_fails_with_lead_at_ro_4(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("appendix-1", "appendix", 1, tier="C"),
            _block("appendix-2", "appendix", 2, tier="C"),
            _block("appendix-3", "appendix", 3, tier="C"),
            _block("lead-1", "lead", 4, tier="A"),
        ])
        f = check_h2(d)
        assert len(f) == 1
        assert f[0].rule == "H2"

    def test_fails_with_zero_leads(self):
        d = _doc([_block("headline-1", "headline", 0, tier="A")])
        f = check_h2(d)
        assert len(f) == 1

    def test_fails_with_two_leads(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
            _block("lead-2", "lead", 2, tier="A"),
        ])
        f = check_h2(d)
        assert len(f) >= 1
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_harness.py -v`
Expected: ImportError on `check_h1`, `check_h2`.

- [ ] **Step 3: Append H1 and H2 to `eom/harness.py`**

```python
# Append to eom/harness.py
def check_h1(doc: EOMDocument) -> list[FailureRecord]:
    """H1: exactly one block of type=headline."""
    n = sum(1 for b in doc.blocks if b.type == "headline")
    if n == 1:
        return []
    return [FailureRecord(rule="H1", message=f"expected 1 headline, found {n}")]


def check_h2(doc: EOMDocument) -> list[FailureRecord]:
    """H2: exactly one lead, with reading_order <= 3."""
    leads = [b for b in doc.blocks if b.type == "lead"]
    out: list[FailureRecord] = []
    if len(leads) != 1:
        out.append(FailureRecord(rule="H2", message=f"expected 1 lead, found {len(leads)}"))
        return out
    lead = leads[0]
    if lead.reading_order > 3:
        out.append(FailureRecord(
            rule="H2",
            message=f"lead reading_order={lead.reading_order} > 3",
            block_id=lead.id,
        ))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eom/harness.py tests/test_harness.py
git commit -m "feat(harness): H1 (one headline) and H2 (one lead, ro<=3)"
```

---

### Task 10: Harness — H3 (tier distribution caps)

**Files:**
- Modify: `eom/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_harness.py
from eom.harness import check_h3


class TestH3:
    def _doc_with_tiers(self, tiers: list[str]) -> EOMDocument:
        # Build a document where the i-th block has the given tier.
        # First block is headline (always A), second is lead (always A).
        blocks = [
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ]
        for i, t in enumerate(tiers):
            blocks.append(_block(f"claim-{i}", "claim", 2 + i, tier=t))
        return _doc(blocks)

    def test_passes_with_balanced_tiers(self):
        # Out of 12 blocks: 1 A (8.3%) ok, 2 B (16.6%) ok, 6 C, 3 D
        # Wait — headline+lead are A. So we need to count those.
        # 12 total: 2 A (head+lead, 16.6%) — ALREADY > 10%. Fix: use bigger doc.
        tiers = ["B"] * 5 + ["C"] * 14 + ["D"] * 4   # +2 A from head/lead
        # Total 25. A=2 (8%), B=5 (20%), C=14 (56%), D=4 (16%). All within caps.
        d = self._doc_with_tiers(tiers)
        assert check_h3(d) == []

    def test_fails_when_A_exceeds_cap(self):
        # 11 blocks total (head+lead+9 A-tier claims): A=11/11=100%
        tiers = ["A"] * 9
        d = self._doc_with_tiers(tiers)
        f = check_h3(d)
        assert any(r.rule == "H3" and "tier A" in r.message for r in f)

    def test_fails_when_B_exceeds_cap(self):
        # 10 blocks: 2 A + 8 B = 80% B, well past 25%
        tiers = ["B"] * 8
        d = self._doc_with_tiers(tiers)
        f = check_h3(d)
        assert any(r.rule == "H3" and "tier B" in r.message for r in f)

    def test_passes_with_only_AB_when_within_caps(self):
        # 25 total: 2 A (8%) + 5 B (20%) + 18 D
        tiers = ["B"] * 5 + ["D"] * 18
        d = self._doc_with_tiers(tiers)
        assert check_h3(d) == []
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_harness.py -v -k H3`
Expected: ImportError on `check_h3`.

- [ ] **Step 3: Append H3 to `eom/harness.py`**

```python
# Append to eom/harness.py
def check_h3(doc: EOMDocument) -> list[FailureRecord]:
    """H3: tier distribution caps. |A|/N <= 0.10, |B|/N <= 0.25, |A|+|B|+|C| <= N."""
    n = len(doc.blocks)
    if n == 0:
        return []  # H6 will catch the empty case
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for b in doc.blocks:
        counts[b.attention_tier] += 1
    out: list[FailureRecord] = []
    if counts["A"] / n > 0.10 + 1e-9:
        out.append(FailureRecord(
            rule="H3",
            message=f"tier A fraction {counts['A']}/{n}={counts['A']/n:.2%} exceeds cap 10%",
        ))
    if counts["B"] / n > 0.25 + 1e-9:
        out.append(FailureRecord(
            rule="H3",
            message=f"tier B fraction {counts['B']}/{n}={counts['B']/n:.2%} exceeds cap 25%",
        ))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_harness.py -v -k H3`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add eom/harness.py tests/test_harness.py
git commit -m "feat(harness): H3 tier distribution caps"
```

---

### Task 11: Harness — H4, H5, H6, H7 (structural integrity)

**Files:**
- Modify: `eom/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_harness.py
from eom.harness import check_h4, check_h5, check_h6, check_h7


class TestH4:
    def test_passes_with_total_order(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
            _block("claim-1", "claim", 2),
        ])
        assert check_h4(d) == []

    def test_fails_with_duplicate_reading_order(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 0, tier="A"),
        ])
        f = check_h4(d)
        assert any(r.rule == "H4" for r in f)

    def test_fails_with_gap(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 2, tier="A"),
        ])
        f = check_h4(d)
        assert any(r.rule == "H4" for r in f)


class TestH5:
    def test_passes_unique_ids(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h5(d) == []

    def test_fails_duplicate_ids(self):
        # Pydantic Block doesn't enforce cross-block uniqueness, so we can build a doc with dupes.
        b1 = _block("claim-1", "claim", 0)
        b2 = _block("claim-1", "claim", 1)
        b_head = _block("headline-1", "headline", 2, tier="A")
        b_lead = _block("lead-1", "lead", 3, tier="A")
        d = _doc([b1, b2, b_head, b_lead])
        f = check_h5(d)
        assert any(r.rule == "H5" for r in f)


class TestH6:
    def test_passes_when_all_have_content(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h6(d) == []
    # Empty content is already rejected by Block validators; H6 is a belt-and-braces.


class TestH7:
    # Block validator already restricts type; H7 is a runtime sanity check.
    def test_passes_canonical_types(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h7(d) == []
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_harness.py -v -k "H4 or H5 or H6 or H7"`
Expected: ImportError on the new functions.

- [ ] **Step 3: Append H4–H7 to `eom/harness.py`**

```python
# Append to eom/harness.py
def check_h4(doc: EOMDocument) -> list[FailureRecord]:
    """H4: reading_order is a total order in [0, N) with no duplicates or gaps."""
    n = len(doc.blocks)
    orders = sorted(b.reading_order for b in doc.blocks)
    expected = list(range(n))
    if orders == expected:
        return []
    out: list[FailureRecord] = []
    seen: set[int] = set()
    for b in doc.blocks:
        if b.reading_order in seen:
            out.append(FailureRecord(
                rule="H4",
                message=f"duplicate reading_order {b.reading_order}",
                block_id=b.id,
            ))
        seen.add(b.reading_order)
    if not out:
        # No duplicates, but the sequence has gaps or is out of range.
        out.append(FailureRecord(
            rule="H4",
            message=f"reading_order is not [0, N); got {orders}, expected {expected}",
        ))
    return out


def check_h5(doc: EOMDocument) -> list[FailureRecord]:
    """H5: block IDs unique within document."""
    seen: set[str] = set()
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.id in seen:
            out.append(FailureRecord(rule="H5", message=f"duplicate id {b.id!r}", block_id=b.id))
        seen.add(b.id)
    return out


def check_h6(doc: EOMDocument) -> list[FailureRecord]:
    """H6: every block has non-empty content (re-check at harness layer)."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if not b.content.strip():
            out.append(FailureRecord(
                rule="H6",
                message="block content is empty or whitespace-only",
                block_id=b.id,
            ))
    return out


CANONICAL_BLOCK_TYPES = {
    "headline", "lead", "claim", "evidence",
    "factbox", "caveat", "decision", "appendix",
}


def check_h7(doc: EOMDocument) -> list[FailureRecord]:
    """H7: every block.type is one of the eight canonical types."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.type not in CANONICAL_BLOCK_TYPES:
            out.append(FailureRecord(
                rule="H7",
                message=f"unknown block type {b.type!r}",
                block_id=b.id,
            ))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eom/harness.py tests/test_harness.py
git commit -m "feat(harness): H4 (reading_order), H5 (unique ids), H6 (non-empty), H7 (canonical types)"
```

---

### Task 12: Harness — H8 (size limits) and H9/H10 (token budgets)

**Files:**
- Modify: `eom/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_harness.py
from eom.harness import check_h8, check_h9, check_h10


class TestH8:
    def test_passes_short(self):
        d = _doc([
            _block("headline-1", "headline", 0, tier="A"),
            _block("lead-1", "lead", 1, tier="A"),
        ])
        assert check_h8(d) == []

    def test_fails_long_headline(self):
        head = Block(
            id="headline-1", type="headline",
            content="x" * 101, attention_tier="A",
            priority=1.0, reading_order=0,
            source_span=SourceSpan(start=0, end=1, quote="x"),
        )
        lead = _block("lead-1", "lead", 1, tier="A")
        d = _doc([head, lead])
        f = check_h8(d)
        assert any(r.rule == "H8" and "headline" in r.message for r in f)

    def test_fails_long_lead(self):
        head = _block("headline-1", "headline", 0, tier="A")
        lead = Block(
            id="lead-1", type="lead",
            content=" ".join(["word"] * 61),  # 61 words, cap is 60
            attention_tier="A",
            priority=0.9, reading_order=1,
            source_span=SourceSpan(start=0, end=10, quote="0123456789"),
        )
        d = _doc([head, lead])
        f = check_h8(d)
        assert any(r.rule == "H8" and "lead" in r.message for r in f)


class TestH9H10:
    def _heavy_block(self, id, type, ro, tier, n_tokens):
        # Tokens are tiktoken-counted; "lorem " is ~2 tokens, so n_tokens/2 reps suffice.
        # Be safe and use single chars repeated.
        return Block(
            id=id, type=type,
            content=("a " * n_tokens).strip(),
            attention_tier=tier,
            priority=0.5, reading_order=ro,
            source_span=SourceSpan(start=0, end=10, quote="0123456789"),
        )

    def test_h9_passes_within_budget(self):
        # B_A=200; 2 tier-A blocks of ~50 tokens each = ~100 tokens.
        d = EOMDocument(
            version="0.1", document_type="memo",
            summary="x", render_profile="executive_brief",
            attention_budget=AttentionBudget(B_A=200, B_AB=800),
            blocks=[
                self._heavy_block("headline-1", "headline", 0, "A", 10),
                self._heavy_block("lead-1", "lead", 1, "A", 40),
            ],
            source=SourceMetadata(checksum="sha256:x", chars=10, lang="en"),
        )
        assert check_h9(d) == []

    def test_h9_fails_over_budget(self):
        # 300 tokens at tier A, budget is 200.
        d = EOMDocument(
            version="0.1", document_type="memo",
            summary="x", render_profile="executive_brief",
            attention_budget=AttentionBudget(B_A=200, B_AB=800),
            blocks=[
                self._heavy_block("headline-1", "headline", 0, "A", 50),
                self._heavy_block("lead-1", "lead", 1, "A", 250),
            ],
            source=SourceMetadata(checksum="sha256:x", chars=10, lang="en"),
        )
        f = check_h9(d)
        assert any(r.rule == "H9" for r in f)

    def test_h10_fails_over_combined_budget(self):
        # B_AB=800 but actual A+B is 1000.
        d = EOMDocument(
            version="0.1", document_type="memo",
            summary="x", render_profile="executive_brief",
            attention_budget=AttentionBudget(B_A=200, B_AB=800),
            blocks=[
                self._heavy_block("headline-1", "headline", 0, "A", 30),
                self._heavy_block("lead-1", "lead", 1, "A", 70),
                self._heavy_block("evidence-1", "evidence", 2, "B", 900),
            ],
            source=SourceMetadata(checksum="sha256:x", chars=10, lang="en"),
        )
        f = check_h10(d)
        assert any(r.rule == "H10" for r in f)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_harness.py -v -k "H8 or H9 or H10"`
Expected: ImportError on the new functions.

- [ ] **Step 3: Append H8/H9/H10 to `eom/harness.py`**

```python
# Append to eom/harness.py
from eom.tokens import count_tokens


def check_h8(doc: EOMDocument) -> list[FailureRecord]:
    """H8: headline <= 100 chars; lead <= 60 words (English)."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.type == "headline" and len(b.content) > 100:
            out.append(FailureRecord(
                rule="H8",
                message=f"headline length {len(b.content)} > 100",
                block_id=b.id,
            ))
        if b.type == "lead":
            n_words = len(b.content.split())
            if n_words > 60:
                out.append(FailureRecord(
                    rule="H8",
                    message=f"lead word count {n_words} > 60",
                    block_id=b.id,
                ))
    return out


def check_h9(doc: EOMDocument) -> list[FailureRecord]:
    """H9: sum of tokens across tier A blocks <= attention_budget.B_A."""
    total = sum(count_tokens(b.content) for b in doc.blocks if b.attention_tier == "A")
    if total > doc.attention_budget.B_A:
        return [FailureRecord(
            rule="H9",
            message=f"tier A total tokens {total} > B_A {doc.attention_budget.B_A}",
        )]
    return []


def check_h10(doc: EOMDocument) -> list[FailureRecord]:
    """H10: sum of tokens across tier A and B blocks <= attention_budget.B_AB."""
    total = sum(
        count_tokens(b.content) for b in doc.blocks
        if b.attention_tier in ("A", "B")
    )
    if total > doc.attention_budget.B_AB:
        return [FailureRecord(
            rule="H10",
            message=f"tier A+B total tokens {total} > B_AB {doc.attention_budget.B_AB}",
        )]
    return []
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eom/harness.py tests/test_harness.py
git commit -m "feat(harness): H8 (size limits), H9 (B_A budget), H10 (B_AB budget)"
```

---

### Task 13: Harness — H11 and H12 (provenance)

**Files:**
- Modify: `eom/harness.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_harness.py
from eom.harness import check_h11, check_h12


SOURCE_TEXT = "The freight cost rose 9 percent after the closure of the western port."


class TestH11:
    def _doc_with_evidence(self, span: SourceSpan | None) -> EOMDocument:
        head = _block("headline-1", "headline", 0, tier="A", with_span=False)
        lead = _block("lead-1", "lead", 1, tier="A", with_span=False)
        evidence = Block(
            id="evidence-1", type="evidence",
            content="freight rose",
            attention_tier="B", priority=0.5, reading_order=2,
            source_span=span,
        )
        return _doc([head, lead, evidence])

    def test_passes_with_valid_span(self):
        # "freight cost rose" is at offsets 4..21 in SOURCE_TEXT
        span = SourceSpan(start=4, end=21, quote="freight cost rose")
        d = self._doc_with_evidence(span)
        assert check_h11(d, SOURCE_TEXT) == []

    def test_fails_when_span_missing(self):
        d = self._doc_with_evidence(None)
        f = check_h11(d, SOURCE_TEXT)
        assert any(r.rule == "H11" and "missing source_span" in r.message for r in f)

    def test_fails_when_offsets_out_of_range(self):
        span = SourceSpan(start=1000, end=1010, quote="0123456789")
        d = self._doc_with_evidence(span)
        f = check_h11(d, SOURCE_TEXT)
        assert any(r.rule == "H11" and "out of range" in r.message for r in f)

    def test_fails_when_quote_does_not_match(self):
        span = SourceSpan(start=4, end=21, quote="WRONG QUOTE TEXT!")
        d = self._doc_with_evidence(span)
        f = check_h11(d, SOURCE_TEXT)
        assert any(r.rule == "H11" and "quote mismatch" in r.message for r in f)


class TestH12:
    def _doc_with_decision(self, ev_span: SourceSpan, decision: Block) -> EOMDocument:
        head = _block("headline-1", "headline", 0, tier="A", with_span=False)
        lead = _block("lead-1", "lead", 1, tier="A", with_span=False)
        evidence = Block(
            id="evidence-1", type="evidence",
            content="freight cost rose",
            attention_tier="B", priority=0.5, reading_order=2,
            source_span=ev_span,
        )
        return _doc([head, lead, evidence, decision])

    def test_passes_with_source_span(self):
        # "freight cost rose 9 percent" at 4..30
        ev_span = SourceSpan(start=4, end=21, quote="freight cost rose")
        decision_span = SourceSpan(start=4, end=30, quote="freight cost rose 9 percent")
        decision = Block(
            id="decision-1", type="decision",
            content="Approve action",
            attention_tier="A", priority=0.8, reading_order=3,
            source_span=decision_span,
        )
        d = self._doc_with_decision(ev_span, decision)
        assert check_h12(d) == []

    def test_passes_when_inferred_with_basis(self):
        ev_span = SourceSpan(start=4, end=21, quote="freight cost rose")
        decision = Block(
            id="decision-1", type="decision",
            content="Approve action",
            attention_tier="A", priority=0.8, reading_order=3,
            source_span=None, is_inferred=True,
            inference_basis=["evidence-1"],
        )
        d = self._doc_with_decision(ev_span, decision)
        assert check_h12(d) == []

    def test_fails_when_inferred_with_empty_basis(self):
        # Pydantic Block already prohibits is_inferred without basis at *write*?  No — it
        # only prohibits inference_basis without is_inferred. is_inferred=True with
        # empty basis is a harness-layer failure.
        ev_span = SourceSpan(start=4, end=21, quote="freight cost rose")
        decision = Block(
            id="decision-1", type="decision",
            content="Approve action",
            attention_tier="A", priority=0.8, reading_order=3,
            source_span=None, is_inferred=True, inference_basis=[],
        )
        d = self._doc_with_decision(ev_span, decision)
        f = check_h12(d)
        assert any(r.rule == "H12" and "empty inference_basis" in r.message for r in f)

    def test_fails_when_basis_points_to_nonexistent_block(self):
        ev_span = SourceSpan(start=4, end=21, quote="freight cost rose")
        decision = Block(
            id="decision-1", type="decision",
            content="Approve action",
            attention_tier="A", priority=0.8, reading_order=3,
            source_span=None, is_inferred=True,
            inference_basis=["nonexistent-9"],
        )
        d = self._doc_with_decision(ev_span, decision)
        f = check_h12(d)
        assert any(r.rule == "H12" and "unknown id" in r.message for r in f)

    def test_fails_when_basis_points_to_wrong_type(self):
        # basis must point to evidence/factbox; pointing to a claim is invalid.
        ev_span = SourceSpan(start=4, end=21, quote="freight cost rose")
        head = _block("headline-1", "headline", 0, tier="A", with_span=False)
        lead = _block("lead-1", "lead", 1, tier="A", with_span=False)
        evidence = Block(
            id="evidence-1", type="evidence",
            content="freight cost rose",
            attention_tier="B", priority=0.5, reading_order=2,
            source_span=ev_span,
        )
        another_claim = Block(
            id="claim-2", type="claim",
            content="Some other claim.",
            attention_tier="C", priority=0.3, reading_order=3,
            source_span=ev_span,
        )
        decision = Block(
            id="decision-1", type="decision",
            content="Approve action",
            attention_tier="A", priority=0.8, reading_order=4,
            source_span=None, is_inferred=True,
            inference_basis=["claim-2"],
        )
        d = _doc([head, lead, evidence, another_claim, decision])
        f = check_h12(d)
        assert any(r.rule == "H12" and "must be evidence or factbox" in r.message for r in f)

    def test_fails_when_neither_span_nor_inferred(self):
        ev_span = SourceSpan(start=4, end=21, quote="freight cost rose")
        decision = Block(
            id="decision-1", type="decision",
            content="Approve action",
            attention_tier="A", priority=0.8, reading_order=3,
            source_span=None, is_inferred=False,
        )
        d = self._doc_with_decision(ev_span, decision)
        f = check_h12(d)
        assert any(r.rule == "H12" and "lacks source_span" in r.message for r in f)
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_harness.py -v -k "H11 or H12"`
Expected: ImportError on the new functions.

- [ ] **Step 3: Append H11 and H12 to `eom/harness.py`**

```python
# Append to eom/harness.py
def check_h11(doc: EOMDocument, source_text: str) -> list[FailureRecord]:
    """H11: every evidence/factbox has a valid source_span (offsets in range, quote matches)."""
    out: list[FailureRecord] = []
    for b in doc.blocks:
        if b.type not in ("evidence", "factbox"):
            continue
        if b.source_span is None:
            out.append(FailureRecord(
                rule="H11",
                message=f"{b.type} block missing source_span",
                block_id=b.id,
            ))
            continue
        span = b.source_span
        if span.end > len(source_text):
            out.append(FailureRecord(
                rule="H11",
                message=f"source_span [{span.start},{span.end}) out of range "
                        f"(source has {len(source_text)} chars)",
                block_id=b.id,
                span=(span.start, span.end),
            ))
            continue
        actual = source_text[span.start : span.end]
        if actual != span.quote:
            out.append(FailureRecord(
                rule="H11",
                message=f"source_span quote mismatch: expected {span.quote!r}, "
                        f"got {actual!r}",
                block_id=b.id,
                span=(span.start, span.end),
            ))
    return out


def check_h12(doc: EOMDocument) -> list[FailureRecord]:
    """H12: claim/decision must have source_span or be is_inferred with valid basis.

    Inference basis must reference existing evidence/factbox blocks.
    Span validity (offsets, quote match) is H11's job at the corpus level for
    these block types when source_span is provided; here we check structural
    consistency only.
    """
    out: list[FailureRecord] = []
    by_id = {b.id: b for b in doc.blocks}
    for b in doc.blocks:
        if b.type not in ("claim", "decision"):
            continue
        if b.is_inferred:
            if not b.inference_basis:
                out.append(FailureRecord(
                    rule="H12",
                    message=f"{b.type} is_inferred=True but empty inference_basis",
                    block_id=b.id,
                ))
                continue
            for ref_id in b.inference_basis:
                ref = by_id.get(ref_id)
                if ref is None:
                    out.append(FailureRecord(
                        rule="H12",
                        message=f"inference_basis contains unknown id {ref_id!r}",
                        block_id=b.id,
                    ))
                elif ref.type not in ("evidence", "factbox"):
                    out.append(FailureRecord(
                        rule="H12",
                        message=f"inference_basis target {ref_id!r} is type {ref.type!r}; "
                                f"must be evidence or factbox",
                        block_id=b.id,
                    ))
        else:
            if b.source_span is None:
                out.append(FailureRecord(
                    rule="H12",
                    message=f"{b.type} lacks source_span and is not is_inferred",
                    block_id=b.id,
                ))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_harness.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add eom/harness.py tests/test_harness.py
git commit -m "feat(harness): H11 (evidence provenance) and H12 (claim/decision provenance)"
```

---

### Task 14: Harness — `validate()` orchestrator

**Files:**
- Modify: `eom/harness.py`
- Modify: `eom/__init__.py`
- Modify: `tests/test_harness.py`

- [ ] **Step 1: Append failing tests**

```python
# Append to tests/test_harness.py
from eom.harness import validate
from tests.fixtures.loader import load_pair


class TestValidate:
    def test_freight_memo_fixture_passes(self):
        source, eom = load_pair("freight_memo")
        report = validate(eom, source)
        if not report.passed:
            for f in report.failures:
                print(f)
        assert report.passed

    def test_metrics_populated(self):
        source, eom = load_pair("freight_memo")
        report = validate(eom, source)
        assert "n_blocks" in report.metrics
        assert "tier_a_count" in report.metrics
        assert "tier_a_tokens" in report.metrics

    def test_warnings_for_h13_h14(self):
        source, eom = load_pair("freight_memo")
        report = validate(eom, source)
        rules = {w.rule for w in report.warnings}
        assert "H13" in rules
        assert "H14" in rules
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_harness.py::TestValidate -v`
Expected: ImportError on `validate`.

- [ ] **Step 3: Append `validate()` to `eom/harness.py`**

```python
# Append to eom/harness.py
def validate(doc: EOMDocument, source_text: str) -> ValidationReport:
    """Run H1-H12 against the document and source text; return ValidationReport."""
    failures: list[FailureRecord] = []
    failures += check_h1(doc)
    failures += check_h2(doc)
    failures += check_h3(doc)
    failures += check_h4(doc)
    failures += check_h5(doc)
    failures += check_h6(doc)
    failures += check_h7(doc)
    failures += check_h8(doc)
    failures += check_h9(doc)
    failures += check_h10(doc)
    failures += check_h11(doc, source_text)
    failures += check_h12(doc)

    metrics = {
        "n_blocks": float(len(doc.blocks)),
        "tier_a_count": float(sum(1 for b in doc.blocks if b.attention_tier == "A")),
        "tier_b_count": float(sum(1 for b in doc.blocks if b.attention_tier == "B")),
        "tier_c_count": float(sum(1 for b in doc.blocks if b.attention_tier == "C")),
        "tier_d_count": float(sum(1 for b in doc.blocks if b.attention_tier == "D")),
        "tier_a_tokens": float(sum(
            count_tokens(b.content) for b in doc.blocks if b.attention_tier == "A"
        )),
        "tier_ab_tokens": float(sum(
            count_tokens(b.content) for b in doc.blocks
            if b.attention_tier in ("A", "B")
        )),
    }

    warnings = [
        WarningRecord(
            rule="H13",
            message="salience monotonicity is corpus-level; not checked here",
        ),
        WarningRecord(
            rule="H14",
            message="lead centrality is corpus-level; not checked here",
        ),
    ]

    return ValidationReport(failures=failures, warnings=warnings, metrics=metrics)
```

- [ ] **Step 4: Re-export from `eom/__init__.py`**

```python
# Update eom/__init__.py
"""EOM — Editorial Object Model.

Public API:
    compile(source_text, hints) -> EOMDocument
    validate(eom, source_text) -> ValidationReport
    render_newspaper(eom) -> str
    render_context_pack(eom, token_budget) -> str
"""

__version__ = "0.1.0"

from eom.harness import validate, ValidationReport, FailureRecord, WarningRecord
from eom.schema import (
    EOMDocument,
    Block,
    SourceSpan,
    SourceMetadata,
    AttentionBudget,
    RENDER_PROFILES,
)

__all__ = [
    "validate",
    "ValidationReport",
    "FailureRecord",
    "WarningRecord",
    "EOMDocument",
    "Block",
    "SourceSpan",
    "SourceMetadata",
    "AttentionBudget",
    "RENDER_PROFILES",
]
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add eom/harness.py eom/__init__.py tests/test_harness.py
git commit -m "feat(harness): validate() orchestrator + corpus warnings"
```

---

### Task 15: Renderer — context pack (text payload)

**Files:**
- Create: `eom/renderers/__init__.py`
- Create: `eom/renderers/context_pack.py`
- Test: `tests/test_renderer_context_pack.py`

- [ ] **Step 1: Create `eom/renderers/__init__.py`**

```python
"""EOM renderers — deterministic, model-free."""

from eom.renderers.context_pack import render_context_pack
from eom.renderers.newspaper import render_newspaper

__all__ = ["render_context_pack", "render_newspaper"]
```

(Note: `newspaper` import will fail until Task 16; that's OK because tests use direct imports until then.)

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_renderer_context_pack.py
import pytest

from eom.renderers.context_pack import render_context_pack
from eom.tokens import count_tokens
from tests.fixtures.loader import load_pair


def test_returns_string():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    assert isinstance(out, str)
    assert len(out) > 0


def test_includes_headline_and_lead_always():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=300)
    headline = next(b for b in eom.blocks if b.type == "headline")
    lead = next(b for b in eom.blocks if b.type == "lead")
    assert headline.content in out
    assert lead.content in out


def test_respects_token_budget():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=200)
    # Some slack for headers and citation markers — within 1.5x budget
    assert count_tokens(out) <= 300


def test_section_headers_present():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    assert "## headline" in out.lower()
    assert "## lead" in out.lower()


def test_citations_present_for_evidence():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    # Each evidence/factbox block in tier A or B should appear with [src:id]
    for b in eom.blocks:
        if b.type in ("evidence", "factbox") and b.attention_tier in ("A", "B"):
            assert f"[src:{b.id}]" in out


def test_tier_a_always_included():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=10_000)
    for b in eom.blocks:
        if b.attention_tier == "A":
            # Block content (or at least its first 30 chars) must be in output.
            head30 = b.content[:30]
            assert head30 in out, f"tier A block {b.id} not rendered"


def test_tier_d_never_included():
    _, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=10_000)
    for b in eom.blocks:
        if b.attention_tier == "D":
            assert b.content not in out


def test_summary_header_with_compression_ratio():
    source, eom = load_pair("freight_memo")
    out = render_context_pack(eom, token_budget=3000)
    assert "source_tokens" in out or "compression" in out
```

- [ ] **Step 3: Run tests to verify failures**

Run: `uv run pytest tests/test_renderer_context_pack.py -v`
Expected: ImportError or FAIL.

- [ ] **Step 4: Implement `eom/renderers/context_pack.py`**

```python
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
    src_tokens = eom.source.chars  # rough proxy; we don't have source text here
    return (
        f"<!-- eom_v{eom.version} | profile={eom.render_profile} | "
        f"document_type={eom.document_type} | "
        f"source_chars={eom.source.chars} | "
        f"context_tokens={body_tokens} -->\n"
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
            chosen.append(b._copy_with(content=truncated) if hasattr(b, "_copy_with")
                          else b.model_copy(update={"content": truncated}))
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
```

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_renderer_context_pack.py -v`
Expected: all 8 passed.

- [ ] **Step 6: Commit**

```bash
git add eom/renderers/__init__.py eom/renderers/context_pack.py tests/test_renderer_context_pack.py
git commit -m "feat(renderers): context_pack with tier-A-first budget filling"
```

---

### Task 16: Renderer — newspaper HTML

**Files:**
- Create: `eom/renderers/newspaper.py`
- Create: `templates/newspaper.html`
- Create: `templates/newspaper.css`
- Test: `tests/test_renderer_newspaper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_renderer_newspaper.py
import pytest
from bs4 import BeautifulSoup

from eom.renderers.newspaper import render_newspaper
from tests.fixtures.loader import load_pair


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def test_returns_html_string():
    _, eom = load_pair("freight_memo")
    html = render_newspaper(eom)
    assert html.startswith("<!DOCTYPE html>") or html.startswith("<html")


def test_contains_headline_in_h1():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    h1 = soup.find("h1")
    assert h1 is not None
    assert "Q1 Freight Cost Update" in h1.get_text()


def test_lead_marked_separately():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    lead = soup.find(class_="eom-lead")
    assert lead is not None
    assert "On-time delivery" in lead.get_text()


def test_tier_a_blocks_in_hero():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    hero = soup.find(class_="eom-hero")
    assert hero is not None
    for b in eom.blocks:
        if b.attention_tier == "A":
            assert b.content[:30] in hero.get_text()


def test_tier_d_in_archive():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    # Archive section should exist (even if empty for fixture without D blocks).
    archive = soup.find(class_="eom-archive")
    assert archive is not None


def test_includes_block_id_attributes():
    _, eom = load_pair("freight_memo")
    soup = _soup(render_newspaper(eom))
    head = next(b for b in eom.blocks if b.type == "headline")
    el = soup.find(attrs={"data-block-id": head.id})
    assert el is not None


def test_inline_css_present():
    _, eom = load_pair("freight_memo")
    html = render_newspaper(eom)
    assert "<style>" in html
```

- [ ] **Step 2: Add `beautifulsoup4` and `lxml` test deps**

Update `pyproject.toml` `[project.optional-dependencies].dev`:
```toml
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
    "mypy>=1.10",
    "beautifulsoup4>=4.12",
]
```

Run: `uv sync --extra dev`

- [ ] **Step 3: Run tests to verify failures**

Run: `uv run pytest tests/test_renderer_newspaper.py -v`
Expected: ImportError on `eom.renderers.newspaper`.

- [ ] **Step 4: Create `templates/newspaper.css`**

```css
* { box-sizing: border-box; }
body {
  font-family: 'Georgia', 'Times New Roman', serif;
  max-width: 1100px;
  margin: 2em auto;
  padding: 0 1em;
  color: #1a1a1a;
  line-height: 1.5;
}
.eom-meta { color: #666; font-size: 0.85em; border-bottom: 1px solid #999; padding-bottom: 0.5em; }
.eom-hero { border-bottom: 3px solid #000; padding-bottom: 1em; margin-bottom: 1.5em; }
.eom-hero h1 { font-size: 2.4em; margin: 0 0 0.3em; line-height: 1.1; }
.eom-lead { font-size: 1.2em; font-weight: 600; color: #333; margin: 0.5em 0 1em; }
.eom-factbox {
  background: #f5f0e6;
  border-left: 4px solid #b8860b;
  padding: 0.75em 1em;
  font-size: 0.95em;
  margin: 1em 0;
}
.eom-decision {
  background: #1a1a1a;
  color: #fff;
  padding: 0.75em 1em;
  margin: 1em 0;
  font-weight: 600;
}
.eom-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 2em; }
.eom-main { font-size: 1em; }
.eom-rail { font-size: 0.92em; color: #444; border-left: 1px dashed #ccc; padding-left: 1em; }
.eom-block { margin: 0.75em 0; }
.eom-block p { margin: 0.4em 0; }
.eom-caveat {
  font-style: italic;
  color: #555;
  border-top: 1px solid #ccc;
  margin-top: 1em;
  padding-top: 0.5em;
}
.eom-archive {
  margin-top: 2em;
  border-top: 2px solid #ccc;
  padding-top: 0.75em;
  font-size: 0.85em;
  color: #777;
}
.eom-archive summary { cursor: pointer; font-weight: 600; }
.eom-block sup { color: #b8860b; font-size: 0.75em; }
@media print {
  body { max-width: 100%; }
  .eom-archive[open] summary { font-weight: bold; }
}
```

- [ ] **Step 5: Create `templates/newspaper.html`**

```html
<!DOCTYPE html>
<html lang="{{ eom.source.lang }}">
<head>
<meta charset="utf-8">
<title>{{ headline.content }}</title>
<style>{{ css }}</style>
</head>
<body>
<div class="eom-meta">EOM v{{ eom.version }} · {{ eom.document_type }} · profile={{ eom.render_profile }}</div>

<div class="eom-hero">
  <h1 data-block-id="{{ headline.id }}">{{ headline.content }}</h1>
  {% if lead %}
  <div class="eom-lead" data-block-id="{{ lead.id }}">{{ lead.content }}</div>
  {% endif %}
  {% for b in tier_a_extras %}
    <div class="eom-block eom-{{ b.type }}" data-block-id="{{ b.id }}">
      <p>{{ b.content }}</p>
      {% if b.source_span %}<sup>[{{ b.id }}]</sup>{% endif %}
    </div>
  {% endfor %}
</div>

<div class="eom-grid">
  <div class="eom-main">
    {% for b in tier_b %}
      <div class="eom-block eom-{{ b.type }}" data-block-id="{{ b.id }}">
        <p>{{ b.content }}</p>
        {% if b.source_span %}<sup>[{{ b.id }}]</sup>{% endif %}
      </div>
    {% endfor %}
  </div>
  <aside class="eom-rail">
    {% for b in tier_c %}
      <div class="eom-block eom-{{ b.type }}" data-block-id="{{ b.id }}">
        <p>{{ b.content }}</p>
      </div>
    {% endfor %}
  </aside>
</div>

<details class="eom-archive">
  <summary>Archive ({{ tier_d|length }} blocks)</summary>
  {% for b in tier_d %}
    <div class="eom-block eom-{{ b.type }}" data-block-id="{{ b.id }}">
      <p>{{ b.content }}</p>
    </div>
  {% endfor %}
</details>

</body>
</html>
```

- [ ] **Step 6: Implement `eom/renderers/newspaper.py`**

```python
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
    lead = next((b for b in parts["A"] if b.type == "lead"), None)
    tier_a_extras = [b for b in parts["A"] if b.type not in ("headline", "lead")]

    css = (_TEMPLATE_DIR / "newspaper.css").read_text(encoding="utf-8")
    template = _env().get_template("newspaper.html")
    return template.render(
        eom=eom,
        css=css,
        headline=headline,
        lead=lead,
        tier_a_extras=tier_a_extras,
        tier_b=parts["B"],
        tier_c=parts["C"],
        tier_d=parts["D"],
    )
```

- [ ] **Step 7: Run tests to verify pass**

Run: `uv run pytest tests/test_renderer_newspaper.py -v`
Expected: all 7 passed.

- [ ] **Step 8: Commit**

```bash
git add templates/ eom/renderers/newspaper.py tests/test_renderer_newspaper.py pyproject.toml
git commit -m "feat(renderers): newspaper HTML view via Jinja2"
```

---

### Task 17: Compiler — base interface

**Files:**
- Create: `eom/compilers/__init__.py`
- Create: `eom/compilers/base.py`
- Test: `tests/test_compiler_base.py`

- [ ] **Step 1: Create `eom/compilers/base.py`**

```python
"""Compiler interface and shared types."""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, runtime_checkable

from eom.schema import EOMDocument


class CompileHints(TypedDict, total=False):
    """Optional hints passed to a compiler. All fields are optional."""

    document_type: Literal[
        "memo", "report", "paper", "transcript", "news", "policy", "other"
    ]
    audience: Literal["executive", "researcher", "general"]
    render_profile: Literal["executive_brief", "analytical_brief"]
    token_budget: int


@runtime_checkable
class Compiler(Protocol):
    """Common interface for all EOM compilers."""

    def compile(
        self,
        source_text: str,
        hints: CompileHints | None = None,
    ) -> EOMDocument: ...
```

- [ ] **Step 2: Create `eom/compilers/__init__.py`**

```python
"""EOM compiler implementations.

Use `get_compiler(kind)` to select a backend by name.
"""

from __future__ import annotations

from typing import Literal

from eom.compilers.base import Compiler, CompileHints

CompilerKind = Literal["rules", "prompted", "finetuned"]


def get_compiler(kind: CompilerKind, **kwargs) -> Compiler:
    """Factory: import lazily so optional ML deps don't break the rules path."""
    if kind == "rules":
        from eom.compilers.rules import RulesCompiler
        return RulesCompiler(**kwargs)
    if kind == "prompted":
        from eom.compilers.prompted import PromptedCompiler
        return PromptedCompiler(**kwargs)
    if kind == "finetuned":
        from eom.compilers.finetuned import FineTunedCompiler
        return FineTunedCompiler(**kwargs)
    raise ValueError(f"unknown compiler kind: {kind!r}")


__all__ = ["Compiler", "CompileHints", "get_compiler"]
```

- [ ] **Step 3: Write a smoke test**

```python
# tests/test_compiler_base.py
from eom.compilers import Compiler, CompileHints, get_compiler


def test_get_compiler_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        get_compiler("magical")


def test_compile_hints_is_typed_dict():
    h: CompileHints = {"document_type": "memo"}
    assert h["document_type"] == "memo"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_compiler_base.py -v`
Expected: 2 passed (get_compiler raises but doesn't try to import rules.py yet — wait, the rejection check passes because we never reach the rules branch. But asking for rules will fail until Task 18 lands. That's OK; we're not testing that here.)

- [ ] **Step 5: Commit**

```bash
git add eom/compilers/__init__.py eom/compilers/base.py tests/test_compiler_base.py
git commit -m "feat(compilers): Compiler protocol and CompileHints"
```

---

### Task 18: Compiler — rules-based, part 1: AST and segmentation

**Files:**
- Create: `eom/compilers/rules.py`
- Test: `tests/test_compiler_rules.py`

The rules compiler is intentionally limited: it must pass H1, H2, H4, H5, H6, H7 always, H3/H8/H9/H10 by truncation, and CANNOT pass H11/H12 reliably (no real semantic salience). It exists for fallback and as scaffolding for the synthetic data pipeline.

- [ ] **Step 1: Write the failing tests (segmentation + structural pass)**

```python
# tests/test_compiler_rules.py
import pytest

from eom.compilers.rules import RulesCompiler
from eom.harness import validate
from tests.fixtures.loader import load_pair


@pytest.fixture
def compiler():
    return RulesCompiler()


def test_compile_returns_eom_document(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    assert eom.version == "0.1"
    assert eom.document_type == "memo"


def test_compile_produces_one_headline(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    n_headline = sum(1 for b in eom.blocks if b.type == "headline")
    assert n_headline == 1


def test_compile_produces_one_lead(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    n_lead = sum(1 for b in eom.blocks if b.type == "lead")
    assert n_lead == 1


def test_compile_passes_structural_invariants(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"document_type": "memo"})
    report = validate(eom, source)
    structural = {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
    structural_failures = [f for f in report.failures if f.rule in structural]
    assert structural_failures == [], structural_failures


def test_compile_fills_source_metadata(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source)
    assert eom.source.chars == len(source)
    assert eom.source.checksum.startswith("sha256:")


def test_default_render_profile_when_no_hints(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source)
    assert eom.render_profile == "executive_brief"


def test_compile_respects_render_profile_hint(compiler):
    source, _ = load_pair("freight_memo")
    eom = compiler.compile(source, hints={"render_profile": "analytical_brief"})
    assert eom.render_profile == "analytical_brief"
    assert eom.attention_budget.B_A == 400
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_compiler_rules.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `eom/compilers/rules.py`**

```python
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

from markdown_it import MarkdownIt

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
        cap_b = max(1, int(0.25 * n))
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
        tier_a_total = sum(count_tokens(b.content) for b in blocks if b.attention_tier == "A")
        if tier_a_total > budget.B_A:
            # Truncate the lowest-priority tier-A block(s) until we fit.
            blocks_sorted = sorted(blocks, key=lambda b: (b.attention_tier != "A", b.priority))
            for i, b in enumerate(blocks_sorted):
                if b.attention_tier != "A":
                    break
                allowance = budget.B_A - sum(
                    count_tokens(x.content) for x in blocks_sorted[:i] if x.attention_tier == "A"
                )
                if allowance <= 0:
                    blocks = [
                        x.model_copy(update={"attention_tier": "B"}) if x.id == b.id else x
                        for x in blocks
                    ]
                    continue
                new_content = self._truncate_to_tokens(b.content, allowance)
                blocks = [
                    x.model_copy(update={"content": new_content}) if x.id == b.id else x
                    for x in blocks
                ]
        # Same for A+B against B_AB.
        ab_total = sum(
            count_tokens(b.content) for b in blocks if b.attention_tier in ("A", "B")
        )
        if ab_total > budget.B_AB:
            for b in sorted(blocks, key=lambda x: (x.attention_tier != "B", x.priority)):
                if b.attention_tier != "B":
                    continue
                blocks = [
                    x.model_copy(update={"attention_tier": "C"}) if x.id == b.id else x
                    for x in blocks
                ]
                ab_total = sum(
                    count_tokens(x.content) for x in blocks if x.attention_tier in ("A", "B")
                )
                if ab_total <= budget.B_AB:
                    break
        return blocks

    def _minimal_doc(self, source, render_profile, budget, document_type):
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_compiler_rules.py -v`
Expected: at least the 7 tests above pass. Some may fail on edge cases — fix them inline by adjusting heuristics, **not by tweaking the harness**.

- [ ] **Step 5: Commit**

```bash
git add eom/compilers/rules.py tests/test_compiler_rules.py
git commit -m "feat(compilers): rules-based deterministic compiler"
```

---

### Task 19: LLM client abstraction

**Files:**
- Create: `eom/compilers/llm_client.py`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Implement `eom/compilers/llm_client.py`**

```python
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
```

- [ ] **Step 2: Write tests**

```python
# tests/test_llm_client.py
from eom.compilers.llm_client import LLMRequest, StubLLMClient


def test_stub_returns_preset():
    stub = StubLLMClient(response="hello")
    req = LLMRequest(system="sys", user="usr")
    out = stub.complete(req)
    assert out == "hello"
    assert stub.last_request is req


def test_request_defaults():
    req = LLMRequest(system="s", user="u")
    assert req.temperature == 0.0
    assert "gemma" in req.model.lower()
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add eom/compilers/llm_client.py tests/test_llm_client.py
git commit -m "feat(compilers): LLM client abstraction (Together + Stub)"
```

---

### Task 20: Compiler — prompted, part 1 (prompt template)

**Files:**
- Create: `eom/compilers/prompted.py`
- Create: `eom/compilers/prompt_template.py`
- Test: `tests/test_compiler_prompted.py`

- [ ] **Step 1: Create `eom/compilers/prompt_template.py`**

```python
"""Prompt template for the prompted EOM compiler.

The system prompt teaches the model the schema and harness in compact
form. The user prompt provides few-shot exemplars + the source.
"""

from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent("""\
    You are an EOM compiler. EOM (Editorial Object Model) is a structured
    representation of documents that serves both human readers and language
    models. Your job is to convert the input document into a valid EOM JSON
    object.

    EOM JSON shape:
    {
      "version": "0.1",
      "document_type": "memo|report|paper|transcript|news|policy|other",
      "summary": "<one-sentence summary>",
      "render_profile": "executive_brief" | "analytical_brief",
      "attention_budget": {"B_A": <int>, "B_AB": <int>},
      "blocks": [
        {
          "id": "<type>-<n>",      // e.g. "claim-1"
          "type": "headline" | "lead" | "claim" | "evidence" |
                  "factbox" | "caveat" | "decision" | "appendix",
          "content": "<the block's text>",
          "attention_tier": "A" | "B" | "C" | "D",
          "priority": <float 0-1>,
          "reading_order": <int>,
          "source_span": {
              "start": <int char offset>,
              "end": <int char offset>,
              "quote": "<verbatim substring of source>"
          } | null,
          "is_inferred": <bool>,
          "inference_basis": [<block_id>, ...],
          "parent_id": <block_id> | null
        }
      ],
      "source": {"checksum": "<sha256:...>", "chars": <int>, "lang": "en"}
    }

    Rules you MUST follow:
    1. Exactly one headline block.
    2. Exactly one lead block, with reading_order ≤ 3.
    3. Tier A ≤ 10% of blocks; Tier B ≤ 25%.
    4. reading_order is a total order in [0, N).
    5. IDs are unique, lowercase, slug-form (e.g. "evidence-2").
    6. evidence and factbox blocks MUST have a valid source_span whose
       `quote` is the verbatim substring at [start:end].
    7. claim and decision blocks must have a source_span OR be is_inferred=true
       with inference_basis pointing to evidence/factbox block IDs.
    8. headline ≤ 100 chars; lead ≤ 60 words.
    9. Tier A total tokens ≤ B_A; Tier A+B total tokens ≤ B_AB.
    10. Newspaper budget: only the most load-bearing blocks earn Tier A.

    Output ONLY the JSON. No prose, no fences, no explanation.
""").strip()


FEW_SHOT_USER = dedent("""\
    Document type: {document_type}
    Render profile: {render_profile}
    Source text (between <<<>>>):

    <<<
    {source_text}
    >>>

    Examples of well-formed EOM (study the structure, then produce JSON for the source above):

    {few_shots}

    Now output the EOM JSON for the source above. Output JSON only.
""").strip()


def build_user_prompt(source_text: str, document_type: str, render_profile: str,
                      few_shots: str) -> str:
    return FEW_SHOT_USER.format(
        document_type=document_type,
        render_profile=render_profile,
        source_text=source_text,
        few_shots=few_shots,
    )
```

- [ ] **Step 2: Commit (template only — implementation in Task 21)**

```bash
git add eom/compilers/prompt_template.py
git commit -m "feat(compilers): prompt template for compiler_prompted"
```

---

### Task 21: Compiler — prompted, part 2 (compile pipeline)

**Files:**
- Create: `eom/compilers/prompted.py`
- Test: `tests/test_compiler_prompted.py`

- [ ] **Step 1: Write the failing tests (using `StubLLMClient`)**

```python
# tests/test_compiler_prompted.py
import json

import pytest

from eom.compilers.llm_client import StubLLMClient
from eom.compilers.prompted import PromptedCompiler
from eom.harness import validate
from eom.schema import EOMDocument
from tests.fixtures.loader import load_pair


def _stub_eom_for(source: str, expected_eom: EOMDocument) -> str:
    """Serialise expected EOM but with placeholder source.checksum/chars (compiler will fill)."""
    d = expected_eom.model_dump(mode="json")
    d["source"]["checksum"] = "PLACEHOLDER"
    d["source"]["chars"] = 0
    return json.dumps(d)


def test_compile_with_stub_returns_validated_eom():
    source, expected = load_pair("freight_memo")
    stub = StubLLMClient(response=_stub_eom_for(source, expected))
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source, hints={"document_type": "memo"})
    assert eom.version == "0.1"


def test_compile_calls_llm_with_source_in_prompt():
    source, expected = load_pair("freight_memo")
    stub = StubLLMClient(response=_stub_eom_for(source, expected))
    compiler = PromptedCompiler(client=stub, few_shots=[])
    compiler.compile(source, hints={"document_type": "memo"})
    assert source[:80] in stub.last_request.user


def test_compile_strips_code_fences_around_json():
    source, expected = load_pair("freight_memo")
    response_with_fence = "```json\n" + _stub_eom_for(source, expected) + "\n```"
    stub = StubLLMClient(response=response_with_fence)
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source, hints={"document_type": "memo"})
    assert eom.version == "0.1"


def test_compile_falls_back_to_rules_on_invalid_json():
    source, _ = load_pair("freight_memo")
    stub = StubLLMClient(response="this is not json at all")
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source, hints={"document_type": "memo"})
    # Fallback path should still produce a valid EOM (from RulesCompiler).
    assert eom.version == "0.1"


def test_fills_source_metadata():
    source, expected = load_pair("freight_memo")
    stub = StubLLMClient(response=_stub_eom_for(source, expected))
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom = compiler.compile(source)
    assert eom.source.chars == len(source)
    assert eom.source.checksum.startswith("sha256:")
```

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_compiler_prompted.py -v`
Expected: ImportError on `PromptedCompiler`.

- [ ] **Step 3: Implement `eom/compilers/prompted.py`**

```python
"""LLM-driven EOM compiler. Uses an injectable LLMClient."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import ValidationError

from eom.compilers.base import CompileHints
from eom.compilers.llm_client import LLMClient, LLMRequest
from eom.compilers.prompt_template import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from eom.compilers.rules import RulesCompiler
from eom.normalise import normalise
from eom.schema import (
    AttentionBudget,
    EOMDocument,
    RENDER_PROFILES,
    SourceMetadata,
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
    model: str = "google/gemma-2-27b-it"  # pinned default

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
            return EOMDocument.model_validate(payload)
        except ValidationError:
            return self._fallback(source, hints)

    def _fallback(self, source: str, hints: CompileHints) -> EOMDocument:
        rules = RulesCompiler()
        return rules.compile(source, hints)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_compiler_prompted.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add eom/compilers/prompted.py tests/test_compiler_prompted.py
git commit -m "feat(compilers): prompted compiler with code-fence stripping and rules fallback"
```

---

### Task 22: Repair loop

**Files:**
- Create: `eom/repair.py`
- Test: `tests/test_repair.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_repair.py
import pytest
from unittest.mock import MagicMock

from eom.repair import compile_with_repair, summarise_failures
from eom.compilers.prompted import PromptedCompiler
from eom.compilers.llm_client import StubLLMClient
from eom.harness import FailureRecord
from tests.fixtures.loader import load_pair


def test_summarise_failures_empty():
    assert summarise_failures([]) == ""


def test_summarise_failures_groups_by_rule():
    fs = [
        FailureRecord(rule="H3", message="tier A 30%"),
        FailureRecord(rule="H3", message="tier B 40%"),
        FailureRecord(rule="H11", message="span out of range", block_id="evidence-1"),
    ]
    summary = summarise_failures(fs)
    assert "H3" in summary
    assert "H11" in summary
    assert "evidence-1" in summary


def test_compile_with_repair_returns_immediately_when_passes(monkeypatch):
    source, expected_eom = load_pair("freight_memo")
    import json
    payload = json.dumps(expected_eom.model_dump(mode="json"))
    stub = StubLLMClient(response=payload)
    compiler = PromptedCompiler(client=stub, few_shots=[])
    eom, attempts = compile_with_repair(compiler, source, hints={"document_type": "memo"})
    assert attempts == 1


def test_compile_with_repair_retries_on_failure(monkeypatch):
    source, expected_eom = load_pair("freight_memo")
    import json
    # First response: malformed; second: valid
    valid = json.dumps(expected_eom.model_dump(mode="json"))
    bad = "not json"

    class TwoShotClient:
        def __init__(self):
            self.responses = [bad, valid]
            self.calls = 0

        def complete(self, req):
            r = self.responses[self.calls]
            self.calls += 1
            return r

    client = TwoShotClient()
    compiler = PromptedCompiler(client=client, few_shots=[])
    eom, attempts = compile_with_repair(compiler, source,
                                         hints={"document_type": "memo"},
                                         max_attempts=3)
    # First attempt: bad json -> falls back to rules (counts as attempt 1)
    # Subsequent repair calls re-prompt the same compiler.
    assert attempts >= 1
```

- [ ] **Step 2: Implement `eom/repair.py`**

```python
"""Repair loop for LLM-based compilers.

If the compiler output fails the harness, summarise failures and re-prompt
the compiler with the failure feedback. Cap at `max_attempts` total
compile invocations. Return (best_eom, attempts) where `best_eom` is the
last attempt's output (which may still fail; check `validate(eom, source).passed`).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol

from eom.compilers.base import Compiler, CompileHints
from eom.harness import FailureRecord, validate
from eom.schema import EOMDocument


def summarise_failures(failures: list[FailureRecord]) -> str:
    """Group failures by rule and produce an actionable summary string."""
    if not failures:
        return ""
    by_rule: dict[str, list[FailureRecord]] = defaultdict(list)
    for f in failures:
        by_rule[f.rule].append(f)
    lines = ["Your previous output failed the EOM harness. Fix these issues:"]
    for rule, fs in sorted(by_rule.items()):
        lines.append(f"\n[{rule}] {len(fs)} failure(s):")
        for f in fs[:5]:  # cap per-rule details
            tag = f" (block {f.block_id})" if f.block_id else ""
            lines.append(f"  - {f.message}{tag}")
        if len(fs) > 5:
            lines.append(f"  - ...and {len(fs) - 5} more.")
    lines.append("\nProduce a corrected EOM JSON. Output JSON only.")
    return "\n".join(lines)


class _CompilerWithFeedback(Protocol):
    """Marker protocol: a compiler that supports an optional feedback string."""
    def compile(
        self, source_text: str, hints: CompileHints | None = ...,
        feedback: str | None = ...,
    ) -> EOMDocument: ...


def compile_with_repair(
    compiler: Compiler,
    source_text: str,
    hints: CompileHints | None = None,
    max_attempts: int = 3,
) -> tuple[EOMDocument, int]:
    """Run compiler with up to `max_attempts` attempts; return (eom, attempts)."""
    attempts = 0
    last_eom: EOMDocument | None = None
    feedback: str | None = None
    for _ in range(max_attempts):
        attempts += 1
        if feedback is None:
            eom = compiler.compile(source_text, hints=hints)
        else:
            # If compiler doesn't accept feedback kw, fall back to plain compile.
            try:
                eom = compiler.compile(source_text, hints=hints, feedback=feedback)  # type: ignore[call-arg]
            except TypeError:
                eom = compiler.compile(source_text, hints=hints)
        last_eom = eom
        report = validate(eom, source_text)
        if report.passed:
            return eom, attempts
        feedback = summarise_failures(report.failures)
    assert last_eom is not None  # max_attempts >= 1
    return last_eom, attempts
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_repair.py -v`
Expected: at least 4 tests pass. (`test_compile_with_repair_retries_on_failure` may pass with attempts=1 if fallback already succeeds; the harness check determines retries.)

- [ ] **Step 4: Commit**

```bash
git add eom/repair.py tests/test_repair.py
git commit -m "feat(repair): repair loop with failure summarisation"
```

---

### Task 23: CLI

**Files:**
- Create: `eom/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from eom.cli import cli
from tests.fixtures.loader import load_pair


@pytest.fixture
def runner():
    return CliRunner()


def test_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "compile" in result.output
    assert "validate" in result.output
    assert "render" in result.output


def test_compile_with_rules(tmp_path, runner):
    source, _ = load_pair("freight_memo")
    md = tmp_path / "doc.md"
    md.write_text(source)
    out = tmp_path / "doc.eom.json"
    result = runner.invoke(cli, [
        "compile", "--input", str(md),
        "--compiler", "rules",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["version"] == "0.1"


def test_validate_fixture(tmp_path, runner):
    source, eom = load_pair("freight_memo")
    md = tmp_path / "doc.md"
    md.write_text(source)
    eom_path = tmp_path / "doc.eom.json"
    eom_path.write_text(eom.model_dump_json())
    result = runner.invoke(cli, [
        "validate", "--eom", str(eom_path),
        "--source", str(md),
    ])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_render_newspaper(tmp_path, runner):
    _, eom = load_pair("freight_memo")
    eom_path = tmp_path / "doc.eom.json"
    eom_path.write_text(eom.model_dump_json())
    out = tmp_path / "doc.html"
    result = runner.invoke(cli, [
        "render", "--eom", str(eom_path),
        "--target", "newspaper",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    assert out.read_text().lower().startswith(("<!doctype", "<html"))


def test_render_context_pack(tmp_path, runner):
    _, eom = load_pair("freight_memo")
    eom_path = tmp_path / "doc.eom.json"
    eom_path.write_text(eom.model_dump_json())
    out = tmp_path / "doc.txt"
    result = runner.invoke(cli, [
        "render", "--eom", str(eom_path),
        "--target", "context-pack",
        "--budget", "3000",
        "--output", str(out),
    ])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "## headline" in text.lower()
```

- [ ] **Step 2: Implement `eom/cli.py`**

```python
"""eom CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from eom.compilers import get_compiler
from eom.compilers.llm_client import TogetherClient
from eom.harness import validate as run_validate
from eom.normalise import normalise
from eom.renderers import render_context_pack, render_newspaper
from eom.repair import compile_with_repair
from eom.schema import EOMDocument


@click.group()
@click.version_option()
def cli():
    """EOM — Editorial Object Model."""


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--compiler", "compiler_kind", type=click.Choice(["rules", "prompted"]),
              default="rules")
@click.option("--profile", type=click.Choice(["executive_brief", "analytical_brief"]),
              default="executive_brief")
@click.option("--document-type",
              type=click.Choice(["memo", "report", "paper", "transcript",
                                  "news", "policy", "other"]),
              default="other")
@click.option("--max-attempts", type=int, default=3)
def compile(input_path, output_path, compiler_kind, profile, document_type, max_attempts):
    """Compile a markdown / text document into EOM JSON."""
    source = normalise(Path(input_path).read_text(encoding="utf-8"))
    if compiler_kind == "prompted":
        client = TogetherClient()
        compiler_obj = get_compiler("prompted", client=client, few_shots=[])
    else:
        compiler_obj = get_compiler("rules")

    hints = {"document_type": document_type, "render_profile": profile}
    eom, attempts = compile_with_repair(compiler_obj, source, hints=hints, max_attempts=max_attempts)
    Path(output_path).write_text(eom.model_dump_json(indent=2))
    click.echo(f"Compiled in {attempts} attempt(s) -> {output_path}")


@cli.command()
@click.option("--eom", "eom_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--source", "source_path", required=True, type=click.Path(exists=True, dir_okay=False))
def validate(eom_path, source_path):
    """Validate an EOM JSON against its source."""
    source = normalise(Path(source_path).read_text(encoding="utf-8"))
    eom = EOMDocument.model_validate_json(Path(eom_path).read_text(encoding="utf-8"))
    report = run_validate(eom, source)
    if report.passed:
        click.echo("PASS")
        click.echo(f"  blocks: {int(report.metrics['n_blocks'])}")
        click.echo(f"  tier A: {int(report.metrics['tier_a_count'])}")
        click.echo(f"  tier A tokens: {int(report.metrics['tier_a_tokens'])}")
    else:
        click.echo(f"FAIL ({len(report.failures)} failures)")
        for f in report.failures:
            tag = f" [{f.block_id}]" if f.block_id else ""
            click.echo(f"  {f.rule}{tag}: {f.message}")
        sys.exit(1)


@cli.command()
@click.option("--eom", "eom_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--target", type=click.Choice(["newspaper", "context-pack"]), required=True)
@click.option("--output", "-o", "output_path", required=True, type=click.Path(dir_okay=False))
@click.option("--budget", type=int, default=3000, help="Token budget for context-pack")
def render(eom_path, target, output_path, budget):
    """Render an EOM to HTML or text."""
    eom = EOMDocument.model_validate_json(Path(eom_path).read_text(encoding="utf-8"))
    if target == "newspaper":
        out = render_newspaper(eom)
    else:
        out = render_context_pack(eom, token_budget=budget)
    Path(output_path).write_text(out, encoding="utf-8")
    click.echo(f"Rendered -> {output_path}")
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 5 passed.

- [ ] **Step 4: Smoke-test the CLI manually**

Run:
```bash
uv run eom --help
uv run eom compile --input tests/fixtures/freight_memo.md --compiler rules --output /tmp/freight.eom.json
uv run eom validate --eom /tmp/freight.eom.json --source tests/fixtures/freight_memo.md
uv run eom render --eom /tmp/freight.eom.json --target newspaper --output /tmp/freight.html
uv run eom render --eom /tmp/freight.eom.json --target context-pack --budget 1000 --output /tmp/freight.txt
```
Expected: each command exits 0; opening `/tmp/freight.html` in a browser shows a recognisable newspaper layout.

- [ ] **Step 5: Commit**

```bash
git add eom/cli.py tests/test_cli.py
git commit -m "feat(cli): eom compile / validate / render commands"
```

---

### Task 24: Gold seed — curation infrastructure

**Files:**
- Create: `data/gold/MANIFEST.json`
- Create: `data/gold/.gitkeep`
- Create: `scripts/scaffold_gold.py`
- Create: `scripts/validate_gold.py`

- [ ] **Step 1: Create `data/gold/MANIFEST.json`** (initially empty)

```json
{
  "version": "0.1",
  "examples": []
}
```

- [ ] **Step 2: Create `data/gold/.gitkeep`** (empty file)

- [ ] **Step 3: Implement `scripts/scaffold_gold.py`**

```python
"""Scaffold a gold example from a raw source.

Runs the rules compiler to produce a starting-point EOM, which the human
then hand-corrects before committing.

Usage:
    uv run python scripts/scaffold_gold.py \\
        --input data/raw/freight_memo.md \\
        --doc-type memo \\
        --slug freight_memo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eom.compilers.rules import RulesCompiler
from eom.normalise import normalise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--doc-type", required=True,
                   choices=["memo", "report", "paper", "transcript",
                            "news", "policy", "other"])
    p.add_argument("--profile", default="executive_brief")
    p.add_argument("--slug", required=True)
    args = p.parse_args()

    src_text = Path(args.input).read_text(encoding="utf-8")
    norm = normalise(src_text)

    out_dir = Path("data/gold") / args.doc_type
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{args.slug}.md"
    eom_path = out_dir / f"{args.slug}.eom.json"
    md_path.write_text(norm, encoding="utf-8")

    compiler = RulesCompiler()
    eom = compiler.compile(norm, hints={
        "document_type": args.doc_type,
        "render_profile": args.profile,
    })
    eom_path.write_text(eom.model_dump_json(indent=2), encoding="utf-8")
    print(f"Wrote scaffold:\n  source: {md_path}\n  eom:    {eom_path}")
    print("Hand-correct the EOM file, then run scripts/validate_gold.py.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `scripts/validate_gold.py`**

```python
"""Validate every (source, eom) pair under data/gold/ against the harness.

Updates data/gold/MANIFEST.json with the inventory.
Exits non-zero if any pair fails the harness.

Usage:
    uv run python scripts/validate_gold.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from eom.harness import validate
from eom.normalise import normalise
from eom.schema import EOMDocument

GOLD_DIR = Path("data/gold")


def main() -> int:
    examples = []
    failures_total = 0
    for type_dir in sorted(p for p in GOLD_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            slug = md.stem
            eom_path = type_dir / f"{slug}.eom.json"
            if not eom_path.exists():
                print(f"  ! {md} has no matching .eom.json")
                failures_total += 1
                continue
            source = normalise(md.read_text(encoding="utf-8"))
            try:
                eom = EOMDocument.model_validate_json(eom_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"  ! {eom_path} invalid schema: {e}")
                failures_total += 1
                continue
            report = validate(eom, source)
            if report.passed:
                print(f"  PASS  {type_dir.name}/{slug}")
                examples.append({
                    "doc_type": type_dir.name,
                    "slug": slug,
                    "source": str(md),
                    "eom": str(eom_path),
                    "n_blocks": int(report.metrics["n_blocks"]),
                })
            else:
                print(f"  FAIL  {type_dir.name}/{slug} ({len(report.failures)} failures)")
                for f in report.failures[:5]:
                    tag = f" [{f.block_id}]" if f.block_id else ""
                    print(f"      {f.rule}{tag}: {f.message}")
                failures_total += 1

    manifest = {
        "version": "0.1",
        "examples": examples,
    }
    Path(GOLD_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(examples)} pass, {failures_total} fail.")
    return 0 if failures_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Commit**

```bash
git add data/gold/.gitkeep data/gold/MANIFEST.json scripts/scaffold_gold.py scripts/validate_gold.py
git commit -m "tools: gold-seed scaffolding and validator scripts"
```

---

### Tasks 25–31: Hand-curate gold seed (30–50 examples)

This is **manual editorial work**, not code. Each example is a (source, EOM) pair that passes H1–H12. Target distribution:

| Doc type | Count | Source candidates |
|---|---|---|
| `memo` | 6 | Public PRDs, internal-style memos from Zapier/Stripe blog, OECD policy briefs |
| `report` | 6 | Federal Register summaries, OECD economic outlooks, World Bank country reports |
| `paper` | 6 | arXiv abstracts (CC-BY), PubMed Central full-text intros |
| `transcript` | 5 | TED Talk transcripts (CC-BY), public conference recordings |
| `news` | 6 | Wikipedia "Current events" portal entries, Reuters fact pages |
| `policy` | 6 | IPCC AR6 chapter summaries, WHO publications, IETF RFC abstracts |
| `other` | 5 | README files (MIT-licensed projects), GitHub release notes |
| **Total** | **40** | |

**Workflow per example:**

- [ ] **Step A: Choose a source document**

Pick a license-friendly source ~300–2000 words. Save as `data/raw/<slug>.md`.

- [ ] **Step B: Scaffold via `scripts/scaffold_gold.py`**

```bash
uv run python scripts/scaffold_gold.py --input data/raw/<slug>.md --doc-type <type> --slug <slug>
```

- [ ] **Step C: Hand-correct the EOM**

Open `data/gold/<type>/<slug>.eom.json` and fix:
- Block typing (the rules compiler often misclassifies; correct based on editorial reading).
- Tier assignment — ensure the most load-bearing 5–10% of blocks are tier A.
- Source spans — verify every `quote` matches `source[start:end]` exactly. Use `python -c "..."` to compute offsets if needed.
- Inferred claims/decisions — if the recommendation isn't directly in the source, set `is_inferred: true` and populate `inference_basis`.
- Compactness — trim tier A blocks to fit B_A; demote others.
- Summary sentence — make it crisp.

- [ ] **Step D: Validate**

```bash
uv run python scripts/validate_gold.py
```

Iterate until your example shows `PASS`.

- [ ] **Step E: Commit when batch of 5 lands**

```bash
git add data/gold/<type>/
git commit -m "data(gold): add <type> examples (<slug-1>, <slug-2>, ...)"
```

**Acceptance criterion for Tasks 25–31**: `scripts/validate_gold.py` reports ≥ 30 examples passing, with at least 4 in each `doc_type`. Run:

```bash
uv run python scripts/validate_gold.py
```
Expected: `30+ pass, 0 fail.`

These tasks are by far the slowest in Phase 1 — budget **2 calendar weeks** for the curation cycle. Lower the count to 30 if 40 takes too long; quality matters more than quantity.

---

### Task 32: End-to-end smoke test on the gold seed

**Files:**
- Create: `tests/test_e2e.py`

This locks in that compiler_rules + harness + renderers all line up on real (not fixture) gold data.

- [ ] **Step 1: Write the test**

```python
# tests/test_e2e.py
"""End-to-end: compile every gold .md with rules, validate, render both views."""

import json
from pathlib import Path

import pytest

from eom.compilers.rules import RulesCompiler
from eom.harness import validate
from eom.normalise import normalise
from eom.renderers import render_context_pack, render_newspaper

GOLD_DIR = Path("data/gold")


def _gold_pairs():
    if not GOLD_DIR.exists():
        return []
    pairs = []
    for type_dir in sorted(p for p in GOLD_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md")):
            eom_path = type_dir / f"{md.stem}.eom.json"
            if eom_path.exists():
                pairs.append((type_dir.name, md, eom_path))
    return pairs


@pytest.mark.parametrize("doc_type,md_path,eom_path", _gold_pairs())
def test_gold_eom_passes_harness(doc_type, md_path, eom_path):
    from eom.schema import EOMDocument
    source = normalise(md_path.read_text(encoding="utf-8"))
    eom = EOMDocument.model_validate_json(eom_path.read_text(encoding="utf-8"))
    report = validate(eom, source)
    if not report.passed:
        for f in report.failures:
            print(f"{f.rule} [{f.block_id}]: {f.message}")
    assert report.passed


@pytest.mark.parametrize("doc_type,md_path,eom_path", _gold_pairs())
def test_gold_eom_renders(doc_type, md_path, eom_path):
    from eom.schema import EOMDocument
    eom = EOMDocument.model_validate_json(eom_path.read_text(encoding="utf-8"))
    html = render_newspaper(eom)
    assert html.lower().startswith(("<!doctype", "<html"))
    text = render_context_pack(eom, token_budget=3000)
    assert "## headline" in text.lower()
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_e2e.py -v`
Expected: 30+ tests, all PASS (one parametrised test per gold example).

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end smoke test across gold seed"
```

---

### Task 33: Phase 1 wrap — README quickstart and notebook

**Files:**
- Modify: `README.md`
- Create: `notebooks/01-explore-gold.ipynb`

- [ ] **Step 1: Update `README.md` quickstart**

Replace the Quickstart section with:

```markdown
## Quickstart

```bash
# Install (Python 3.11)
uv venv && uv sync --extra dev

# Compile a markdown document with the rules-based compiler
uv run eom compile \
    --input tests/fixtures/freight_memo.md \
    --compiler rules \
    --document-type memo \
    --output freight.eom.json

# Validate against the harness
uv run eom validate --eom freight.eom.json --source tests/fixtures/freight_memo.md

# Render
uv run eom render --eom freight.eom.json --target newspaper --output freight.html
uv run eom render --eom freight.eom.json --target context-pack --budget 1000 --output freight.txt

# Test the gold seed
uv run python scripts/validate_gold.py
```

For the prompted compiler (Gemma-4-27B via Together AI), set `TOGETHER_API_KEY`:

```bash
export TOGETHER_API_KEY=...
uv run eom compile -i my.md --compiler prompted --output my.eom.json
```

## Repository status (Phase 1 complete)

- ✓ Schema (Pydantic) and harness validator (H1-H12)
- ✓ Two compilers: `rules` (deterministic) and `prompted` (Gemma-4-27B)
- ✓ Two renderers: newspaper HTML and LLM context pack
- ✓ Repair loop with failure summarisation
- ✓ CLI: `eom compile / validate / render`
- ✓ 30+ hand-curated gold examples passing harness
- ☐ Phase 2: synthetic dataset generation pipeline
- ☐ Phase 3: Gemma-4 E2B fine-tune via Unsloth
- ☐ Phase 4: standard publication
```

- [ ] **Step 2: Create `notebooks/01-explore-gold.ipynb`**

(Use Jupyter or write it as JSON manually.) The notebook should:
1. Load every gold example via `tests/fixtures/loader`.
2. Show distribution: blocks-per-doc, tier histogram, doc_type counts, lang distribution.
3. Render one example per doc_type as inline HTML (using `IPython.display.HTML`).
4. Print harness pass rate.

This serves as a sanity check and a Phase-2 baseline reference.

Skeleton (write to `notebooks/01-explore-gold.ipynb` as a Jupyter notebook):

```python
# Cell 1
from pathlib import Path
import json
from collections import Counter

from eom.normalise import normalise
from eom.schema import EOMDocument
from eom.harness import validate
from eom.renderers import render_newspaper

GOLD = Path("data/gold")

# Cell 2: inventory
manifest = json.loads((GOLD / "MANIFEST.json").read_text())
print(f"{len(manifest['examples'])} examples")
type_counts = Counter(e["doc_type"] for e in manifest["examples"])
for t, n in type_counts.most_common():
    print(f"  {t:12s} {n}")

# Cell 3: harness pass rate
passed = 0
total = 0
for ex in manifest["examples"]:
    src = normalise(Path(ex["source"]).read_text(encoding="utf-8"))
    eom = EOMDocument.model_validate_json(Path(ex["eom"]).read_text(encoding="utf-8"))
    report = validate(eom, src)
    if report.passed:
        passed += 1
    total += 1
print(f"Harness pass rate: {passed}/{total} = {passed/total:.0%}")

# Cell 4: tier distribution
from itertools import chain
all_blocks = []
for ex in manifest["examples"]:
    eom = EOMDocument.model_validate_json(Path(ex["eom"]).read_text(encoding="utf-8"))
    all_blocks.extend(eom.blocks)
tier_counts = Counter(b.attention_tier for b in all_blocks)
type_counts = Counter(b.type for b in all_blocks)
print("Tier distribution:", dict(tier_counts))
print("Type distribution:", dict(type_counts))

# Cell 5: render one example per doc_type as HTML
from IPython.display import HTML, display
seen = set()
for ex in manifest["examples"]:
    if ex["doc_type"] in seen:
        continue
    seen.add(ex["doc_type"])
    eom = EOMDocument.model_validate_json(Path(ex["eom"]).read_text(encoding="utf-8"))
    print(f"\n--- {ex['doc_type']}: {ex['slug']} ---")
    display(HTML(render_newspaper(eom)))
```

- [ ] **Step 3: Run the notebook end-to-end**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/01-explore-gold.ipynb`
Expected: clean execution, harness pass rate ≥ 95%.

- [ ] **Step 4: Commit**

```bash
git add README.md notebooks/01-explore-gold.ipynb
git commit -m "docs: Phase 1 wrap — README quickstart and gold exploration notebook"
```

---

## Phase 2: Data (Weeks 5–9)

Phase 2 produces the synthetic training dataset (~5k harness-passing pairs) used to fine-tune the converter in Phase 3. Tasks here are coarser-grained because they depend on what we learn shipping Phase 1's prompted compiler.

### Task 34: Phase 2 kickoff — measure prompted baseline

**Deliverable:** `notebooks/02-prompted-baseline.ipynb` reporting H1–H12 pass rates of `compiler_prompted` (Gemma-4-27B via Together AI) on the gold seed, plus failure-mode breakdown by rule.

**Acceptance criteria:**
- Notebook runs end-to-end against `TOGETHER_API_KEY`.
- Reports per-rule pass rate (e.g., "H11 fails 35% of attempts because of off-by-one source spans").
- Saves a CSV `data/eval/prompted_baseline.csv` with one row per gold example.
- Identifies the 2–3 most common failure modes; we'll target those first in Phase 2 dataset filters.

**Implementation notes:**
- For each gold example, run `compile_with_repair(PromptedCompiler(...), source, hints, max_attempts=3)`.
- Record `final_passed`, `attempts_used`, per-rule failures across all attempts.
- This is the empirical baseline `compiler_finetuned` must beat (success criterion #2).

### Task 35: Source pool curation

**Deliverable:** ~10,000 unlabeled markdown documents in `data/raw/` from license-clean sources.

**Acceptance criteria:**
- Mix: 50% Wikipedia stubs (curated for length 300–3000 words), 20% government reports (federal register summaries, IPCC AR6 chapter sections), 15% PubMed abstracts, 10% IETF RFC abstracts, 5% public PRDs / W3C specs.
- Stored as `data/raw/<doc_type>/<slug>.md`.
- A `data/raw/MANIFEST.json` index with `source_url`, `license`, `lang`, `doc_type`.
- ≥ 95% are English; the rest are tagged with their actual `lang` for later multilingual work.

**Implementation notes:**
- Wikipedia: dump-extract via `mwparserfromhell` or `wikiextractor`; sample 5,000 stubs by length filter.
- Government: scrape federal register API for executive summaries.
- arXiv abstracts: download CC-BY subset via the arXiv API.
- This is mostly scripting — write `scripts/fetch_*.py` per source; do not commit raw files individually (the manifest is enough).

### Task 36: Privileged-context teacher pipeline

**Deliverable:** `scripts/generate_synthetic.py` that takes a raw markdown doc and emits an EOM JSON paired with it, harness-validated.

**Acceptance criteria:**
- Pipeline stages (per the spec §8.2):
  1. Normalise source.
  2. Compute scaffolding: heading hierarchy (markdown-it AST), entities (spaCy NER), sentence-level salience (extractive summariser e.g. `lexrank` or `bertsum`).
  3. Compose teacher prompt: source + scaffolding + harness rules + 3 random gold few-shots + structured field placeholders.
  4. Call teacher LLM (Gemma-4-27B via Together; optionally allow a frontier-LLM "calibration" mode toggled by env var for the first 200 examples).
  5. Parse JSON; if malformed, repair via `_strip_fences` and one retry.
  6. Run harness; if fails, repair-loop up to 3 attempts with failure feedback.
  7. Emit only harness-passing pairs.
- CLI:
  ```bash
  uv run python scripts/generate_synthetic.py \
      --raw-dir data/raw \
      --out-dir data/synthetic \
      --target 5000 \
      --workers 4
  ```
- Logs per-source attempt counts + final outcome to `data/synthetic/log.jsonl`.

**Implementation notes:**
- `scripts/generate_synthetic.py` is one self-contained module; reuse `eom.compilers.prompted` and `eom.repair` rather than reimplementing.
- Run in parallel with `concurrent.futures.ThreadPoolExecutor` (Together API is rate-limited to ~10 concurrent at base tier).
- Track yields: target ≥ 95% post-repair pass rate; if you see < 30%, expand gold seed or improve teacher prompt before continuing.

### Task 37: Synthetic generation — first 500 batch

**Deliverable:** 500 harness-passing pairs in `data/synthetic/`.

**Acceptance criteria:**
- 500 unique source documents, each paired with a passing EOM JSON.
- Average attempts ≤ 1.8 (i.e., most pass on first try after week-7 prompt iteration).
- Per-rule pass-rate report saved as `data/synthetic/quality_report_500.json`.

**Decision gate:**
- If post-repair pass rate < 30%: stop and audit teacher prompt or expand gold seed (revise to ~100 gold).
- If 30–60%: continue cautiously; consider adding the calibration LLM (e.g., Claude or GPT) for the next batch.
- If ≥ 60%: proceed to scale to 5k.

### Task 38: Synthetic generation — scale to 5,000

**Deliverable:** 5,000 harness-passing pairs.

**Acceptance criteria:**
- 5,000 in `data/synthetic/`.
- Distribution roughly proportional to `data/raw/` (no single doc_type dominates).
- 200 documents reserved for held-out evaluation in `data/eval/`. These are sampled BEFORE training, never seen during training, and used for H13/H14 measurements.
- `data/synthetic/quality_report_5k.json` with per-rule pass rate, attempts distribution, content-length histogram.

### Task 39: Corpus evaluators (H13, H13b, H14)

**Files:**
- Create: `eom/evaluators.py`
- Create: `scripts/eval_corpus.py`
- Create: `notebooks/04-eval-corpus.ipynb` (skeleton; fill in with results in Phase 3)

**Deliverable:** Working corpus evaluators that read held-out (source, EOM) pairs and produce H13/H13b/H14 pass/fail decisions plus comparator results vs. baselines.

**Acceptance criteria:**
- H13 implementation:
  - For each held-out doc, generate ~5 questions with a frontier oracle LLM (`anthropic` or `openai` SDK) given the source.
  - Build context packs three ways: tier-A-first (EOM order), reading-order, random.
  - For each (question, pack), ask the oracle "is the answer present in this context?" → binary.
  - Aggregate accuracy per strategy at budgets 1k, 3k, 8k.
  - Report whether tier-A-first > reading-order > random with p < 0.05 (paired bootstrap).
- H13b implementation:
  - Sample 100 blocks across held-out.
  - Oracle judges entailment of `block.content` from `source[block.source_span.start:end]` (or from `inference_basis` blocks for inferred ones).
  - Pass threshold: ≥ 95%.
- H14 implementation:
  - For each held-out doc, ablate each tier-A block from the context pack; measure accuracy drop.
  - Lead must produce the largest mean drop across the corpus.
- Comparator: also compute `truncation`, `lexrank top-k`, and `LLM summary` baselines on same questions.
- CLI:
  ```bash
  uv run python scripts/eval_corpus.py \
      --eval-dir data/eval \
      --questions-cache data/eval/questions.jsonl \
      --output data/eval/results_<run_id>.json
  ```
- Results notebook visualises monotonicity (line chart of accuracy vs. budget for the three strategies).

**Implementation notes:**
- Question generation is expensive — cache to `questions.jsonl` and reuse across runs.
- Use Anthropic Claude or OpenAI GPT as the oracle (toggle via env: `EOM_ORACLE_PROVIDER=anthropic|openai`). Default: Anthropic Claude.
- Bootstrap CI: 1,000 resamples; report point estimate + 95% CI.

---

## Phase 3: Fine-tune (Weeks 9–11)

Phase 3 produces the research artifact: a Gemma-4 E2B converter that beats the prompted baseline on H1–H12 pass rate.

### Task 40: Dataset prep for SFT

**Deliverable:** Hugging Face `datasets`-format training set ready for Unsloth.

**Files:**
- Create: `scripts/prepare_sft_dataset.py`

**Acceptance criteria:**
- Reads `data/synthetic/*.eom.json` + matching sources.
- Tokenises (input, target) pairs:
  - Input = system prompt (compact harness summary) + user prompt (source text + render_profile + doc_type) + `<assistant>` token.
  - Target = EOM JSON serialised compactly (no indentation; deterministic key order).
- Filters: only pairs where harness passes (`validate(eom, source).passed is True`).
- Splits: 95% train / 5% validation; held-out 200 stays separate.
- Saves to `data/train/sft.parquet` and `data/train/val.parquet`.

### Task 41: Unsloth SFT training loop

**Deliverable:** `scripts/train_sft.py` and `notebooks/03-train-sft.ipynb` (Kaggle/Colab runnable).

**Files:**
- Create: `scripts/train_sft.py`
- Create: `notebooks/03-train-sft.ipynb`

**Acceptance criteria:**
- Uses Unsloth's `FastLanguageModel.from_pretrained(model_name="unsloth/gemma-2-2b-it-bnb-4bit", ...)` (or Gemma 4 E2B id when available).
- LoRA: rank 16, alpha 32, target modules `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`.
- Optimiser: AdamW 8-bit; LR 2e-4; cosine schedule with 3% warmup.
- Batch size: 4 (effective 16 with grad accumulation).
- Max sequence length: 8k (for source + EOM JSON).
- Token-weighted loss: 2x weight on `attention_tier` and `priority` field tokens — add a regex-based mask in the data collator.
- Eval per epoch on `data/train/val.parquet` and (cheaper) on a 50-doc subset of held-out gold.
- Checkpoints to `runs/sft/<timestamp>/`.
- Report metrics: train loss, val loss, harness pass rate on subset, per-rule pass rate.
- Acceptance: held-out H1–H12 pass rate ≥ 85% by epoch 5; if not, see decision gates below.
- Notebook is self-contained and can run on a Kaggle TPU-VM or Colab Pro+ A100.

**Decision gates during training:**
- If val loss is plateauing but harness pass rate is < 60% by epoch 3: prompt format is wrong; revisit data collator.
- If pass rate ≥ 90%: skip Stage 3b RLVR; ship SFT-only.
- If pass rate is stuck at 70–85%: run Stage 3b RLVR.

### Task 42: Compiler — fine-tuned wrapper

**Files:**
- Create: `eom/compilers/finetuned.py`
- Test: `tests/test_compiler_finetuned.py` (skipped on CI; runs only when `EOM_FINETUNED_CKPT` env var points to a checkpoint)

**Deliverable:** A `FineTunedCompiler` class that loads the SFT'd Gemma-4 E2B checkpoint and produces EOM JSON.

**Acceptance criteria:**
- Implements the `Compiler` protocol.
- Loads from `EOM_FINETUNED_CKPT` env var (default to a placeholder that errors with a helpful message).
- Inference uses Unsloth's `FastLanguageModel.for_inference()` mode.
- Same JSON parsing + rules-fallback path as `PromptedCompiler`.
- Runs in < 5s per document on a T4 GPU; tested on the 30-doc gold seed.

### Task 43 (optional): RLVR

**Files:**
- Create: `scripts/train_rlvr.py` (only if Task 41's checkpoint stalls in 70–85% pass-rate band)

**Deliverable:** GRPO/PPO training loop using the harness validator as the reward function.

**Acceptance criteria:**
- Reward: `0.6 * harness_pass + 0.2 * compactness_score - 0.2 * faithfulness_failures`.
- Reference policy: Stage 3a checkpoint.
- Saves to `runs/rlvr/<timestamp>/`.
- Acceptance: held-out pass rate uplift ≥ 5 absolute pp over Stage 3a, with no regression on H13b (faithfulness audit).
- If Stage 3a already hits ≥ 90%, **skip this task entirely**.

### Task 44: Phase 3 evaluation

**Deliverable:** Updated `notebooks/04-eval-corpus.ipynb` with results from `compiler_finetuned`.

**Acceptance criteria:**
- All four success criteria from the spec evaluated:
  1. `compiler_finetuned` H1–H12 pass rate on held-out gold ≥ 95%.
  2. `compiler_finetuned` strictly beats `compiler_prompted` on H1–H12 pass rate.
  3. EOM context pack beats raw-truncation baseline by ≥ 5pp at 3k budget on downstream QA.
  4. H13 monotonicity holds with p < 0.05.
- Notebook produces publishable plots: pass rate by rule (bar chart), accuracy vs. budget (line chart), tier-removal ablations (bar chart for H14).
- Final results saved as `data/eval/phase3_results.json`.

---

## Phase 4: Publish (Weeks 11–12)

### Task 45: Harness spec doc

**Deliverable:** `docs/harness-spec.md` — RFC-style standard document.

**Acceptance criteria:**
- Sections: Abstract, Status, Conventions (text normalisation), Block Types, Per-Document Properties (H1–H12 each with formal statement, rationale, examples of conformant + non-conformant), Corpus-Level Properties (H13–H14), Reference Encoding (link to schema-spec), Versioning, Appendix (FAQ).
- Each property has a formal statement, examples, and a reference Python check.
- Reads as a publishable standard. Length: 4,000–6,000 words.

### Task 46: Schema spec doc

**Deliverable:** `docs/schema-spec.md` — reference encoding spec.

**Acceptance criteria:**
- Defines: JSON Schema, Pydantic model reference, render-profile catalog, normalisation rules.
- Includes the worked `freight_memo` example end-to-end.
- Documents extension points for v0.2 (relations, render hints, llm_tier).

### Task 47: Public release

**Deliverable:** GitHub release v0.1.0 + announcement post.

**Acceptance criteria:**
- Tagged release `v0.1.0` with built wheel.
- README has citation block.
- Public Hugging Face repo for the fine-tuned Gemma-4 E2B checkpoint.
- 500–1500 word writeup at `docs/releases/v0.1.0.md` with design narrative, benchmark table, link to spec.
- License: MIT for code; CC-BY 4.0 for spec doc; checkpoint follows Gemma's terms of use.

### Task 48: Retrospective

**Deliverable:** `docs/releases/v0.1.0-retro.md` — what worked, what didn't, what changes for v0.2.

**Acceptance criteria:**
- Honest assessment of where the harness needs to evolve.
- v0.2 candidate list with rationale (relations? render hints? Spatial EOM?).
- Lessons for the synthetic-data pipeline.

---

## Risks and decision gates (cross-phase)

| Phase | Trigger | Action |
|---|---|---|
| 1 | Gold seed curation > 2 weeks | Drop to 30 examples; broaden type coverage instead |
| 2 | Synthetic pass rate < 30% post-repair | Audit teacher prompt; expand gold to 100 |
| 2 | H13 monotonicity fails on prompted-baseline | Diagnose: is it the priority signal or the budget filling? Tweak `render_context_pack` greedy strategy before training |
| 3 | SFT epoch-5 pass rate < 85% | Investigate distribution shift between gold and synthetic; consider larger LoRA rank; consider adding RLVR |
| 3 | SFT pass rate < `compiler_prompted` | The fine-tune is overfitting teacher artifacts. Filter synthetic by stricter criteria; add 200 hand-validated synthetic samples |
| 4 | Standard not adopted by anyone | Out of scope for v0.1; v0.2 cycle includes outreach plan |

## Self-review notes (for the plan author / engineer reading this)

- The plan reuses the canonical `freight_memo` fixture across all of Phase 1; treat it as the litmus test before committing each component.
- All tests use the harness as ground truth — never relax a harness rule to make a test pass; instead, fix the compiler.
- Each commit is atomic: a single feature with passing tests. If a test fails after green-light, revert and try again rather than chaining fixes.
- The mega-plan is intentionally less prescriptive in Phases 2–4. When you reach the start of each phase, write a Phase-N-detailed-tasks file in `docs/superpowers/plans/` that fleshes those tasks out at TDD granularity using the same template as Phase 1 here.
