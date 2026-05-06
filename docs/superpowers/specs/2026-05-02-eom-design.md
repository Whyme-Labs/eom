# EOM (Editorial Object Model) — Design Spec v0.1

**Status**: Draft for implementation
**Date**: 2026-05-02
**Author**: soh (with collaborative brainstorm)

---

## 1. Problem statement

Current document formats are inefficient for both humans and language models. They optimise one of three things — visual layout (PDF), linear text (markdown, plaintext), or machine-readable shape (JSON, XML) — at the expense of the others.

For human-AI collaboration, none of these is sufficient. Humans need scannable hierarchy and visual priority. Models need explicit salience, source-grounded provenance, and compact context payloads under attention budgets. Today both are forced through a flattened markdown pipeline that loses the structure carrying meaning.

EOM proposes a downstream representation, sitting between any document source (or any LLM output) and any consumer (human-rendered or model-ingested). It is built around one principle: **layout is information, and attention budgets are part of the protocol.**

---

## 2. Goals and non-goals

### Goals

- **G1 (standard-first)**: Publish v0.1 of EOM as an open standard. Ship a versioned spec, a property-based harness, a reference schema, and a reference implementation.
- **G2 (research-first)**: Train a small specialised converter (Gemma 4 E2B via Unsloth) that turns plain markdown / text into harness-conformant EOM. Demonstrate that the converter beats a strong prompted baseline on harness pass rate, and that EOM context packs beat raw-truncation baselines on downstream task accuracy at fixed token budgets.
- **G3 (deferred / future)**: A domain-specialised education submission to the Gemma 4 Good Hackathon Unsloth track. Not in v0.1 scope unless the May 18 2026 deadline is reachable without compromising G1 and G2; under the 12-week plan below it is not, and we accept that.

### Non-goals (v0.1)

- A new authoring syntax (no "Editorial Markdown" tags). Authors keep writing markdown / prose; the converter does the structure inference.
- Modifying or fine-tuning the main reasoning LLM. EOM is downstream of any LLM and does not change how that LLM is prompted or trained.
- Multimodal input. Images, figures, and tables are treated as text descriptions inside `evidence` blocks. Image/PDF spatial source spans are v0.2.
- Spatial / 3D rendering ("Spatial EOM"). Promising future direction, out of v0.1.
- A typed inter-block relations layer (`supports`, `contradicts`, `qualifies`). Considered for v0.2 once we have evidence it changes downstream task accuracy.
- Per-block render-slot hints. Renderers derive layout from `attention_tier + priority + reading_order` deterministically.

---

## 3. Architecture

The compiler accepts markdown / plain text from two upstream paths:

```
       [existing markdown doc]              [user prompt]
                  │                              │
                  │                              ▼
                  │                       ┌──────────────┐
                  │                       │  Main LLM    │  (any model: GPT-5, Claude,
                  │                       │  EOM-blind   │   Gemma-4-27B, etc.)
                  │                       └──────┬───────┘
                  │                              │ markdown / prose
                  └──────────────┬───────────────┘
                                 ▼
                ┌─────────────────────────────────────┐
                │  EOM Compiler                       │  one interface, three impls:
                │                                     │  - rules (deterministic, fallback)
                │                                     │  - prompted (frontier LLM)
                │                                     │  - finetuned (Gemma-4 E2B + Unsloth)
                └──────────────┬──────────────────────┘
                               │ EOM JSON
                               ▼
                ┌─────────────────────────────────────┐
                │  Harness Validator (H1-H12)         │  the standard
                │  + repair pass (LLM compilers only) │
                └──────────────┬──────────────────────┘
                               │ validated EOM JSON
                               ▼
                ┌─────────────┬───────────────────────┐
                │ Newspaper   │ LLM context pack      │  deterministic renderers
                │ HTML view   │ (text payload)        │  consume the same EOM
                └─────────────┴───────────────────────┘
```

The compiler is symmetric across the two upstream paths: it does not know or care whether the markdown came from a human-authored document or from another LLM's output.

### Components

1. **Harness** — versioned set of testable properties; the source of truth for "is this EOM-conformant?". Pure-Python validator library plus an RFC-style spec doc.
2. **Reference schema** — Pydantic + JSON Schema; the simplest encoding that satisfies the harness.
3. **Compiler** — three implementations behind one `Compiler` interface (rules / prompted / finetuned).
4. **Renderers** — `render_newspaper(eom) -> HTML` (Jinja2) and `render_context_pack(eom, budget) -> text`. Deterministic; no model in the rendering loop.
5. **Dataset pipeline** — gold seed (30–50 hand-curated) + synthetic generator (privileged-context teacher → harness-filtered pairs) targeting ~5k pairs.
6. **Eval harness** — per-doc validators, corpus-level evaluators (H13–H14), and comparator evals against truncation/summary/extractive baselines.

### Architectural invariants (v0.1 will not violate)

- The main LLM is never modified, fine-tuned, or instructed about EOM. EOM lives entirely downstream.
- Only the harness defines conformance. No model sits inside the validator.
- Renderers are deterministic. No ML in human-facing or LLM-facing rendering.
- The compiler is replaceable. Swapping rules/prompted/finetuned must not change the contract.

---

## 4. The harness

The harness is the standard. It is a versioned set of testable properties partitioned into per-document validators (cheap, run on every output) and corpus-level evaluators (slow, run at milestones).

### 4.1 Per-document validators (H1–H12)

Pure functions of `(eom_json, source_text)`. Sub-millisecond.

**Structural** (syntactic):
- **H1**: Exactly one block of `type=headline`.
- **H2**: Exactly one block of `type=lead`, with `reading_order ≤ 3`.
- **H3**: Tier distribution caps: `|A| / N ≤ 0.10`, `|B| / N ≤ 0.25`, `|A| + |B| + |C| ≤ N`. Hard error if violated.
- **H4**: `reading_order` is a total order: unique integers in `[0, N)`, no gaps.
- **H5**: Block IDs unique within document.
- **H6**: Every block has non-empty `content`.
- **H7**: `block.type` is one of the eight canonical types: `headline`, `lead`, `claim`, `evidence`, `factbox`, `caveat`, `decision`, `appendix`.

**Compactness** (newspaper budget):
- **H8**: `headline ≤ 100 chars`; `lead ≤ 60 words` (English; non-English bounds are language-specific and tracked in §14).
- **H9**: Sum of tokens across all tier A blocks ≤ `attention_budget.B_A` (default 200). Tokenisation reference: `tiktoken` `cl100k_base`; the validator pins this to keep budgets reproducible.
- **H10**: Sum of tokens across all tier A and B blocks ≤ `attention_budget.B_AB` (default 800).

**Faithfulness** (provenance):
- **H11**: Every `evidence` and `factbox` block has a `source_span` whose `(start, end)` offsets are valid in `source_text` and whose `quote` field exactly matches `source_text[start:end]` after normalisation.
- **H12**: Every `claim` and `decision` block has either a `source_span` or `is_inferred = true`. If `is_inferred`, `inference_basis` must list at least one block ID, and every listed ID must reference a block of type `evidence` or `factbox` within the same document. Inferred claims/decisions are still subject to entailment requirements (see §4.2 H13b).

### 4.2 Corpus-level evaluators (H13–H14)

Cannot be checked on a single document. Measured over a held-out 200-document evaluation set.

- **H13 — Salience monotonicity**: Build LLM context packs at fixed token budgets (1k, 3k, 8k) using three strategies — tier-A-first (EOM order), reading-order, random. Run a downstream QA suite (≥5 questions per source, generated by a frontier oracle LLM). Tier-A-first must score strictly higher than reading-order and random at p < 0.05.
- **H13b — Faithfulness audit (sub-property of H11/H12 at corpus scale)**: Sample 100 EOM blocks across the held-out set; an oracle LLM judges entailment of `content` from the cited `source_span` (or `inference_basis` for inferred blocks). Pass threshold: ≥95% entailed.
- **H14 — Lead centrality**: For each held-out doc, ablate the `lead` block from the context pack; measure downstream-accuracy drop. Compare to ablating any other single block. The lead must produce the largest mean drop across the held-out set.

### 4.3 Validator return shape

```python
ValidationReport(
    passed: bool,                      # True iff all H1-H12 pass
    failures: list[FailureRecord],     # which Hxx, why, where (block ID, span, etc.)
    warnings: list[WarningRecord],     # H13/H14 are not checkable here
    metrics: dict[str, float],         # tier counts, token counts, span coverage
)
```

### 4.4 Text normalisation

All offset comparisons assume:
- NFC unicode normalisation.
- Newlines normalised to `\n` (no `\r\n`, no `\r`).
- Trailing whitespace stripped per line.
- No leading/trailing BOM.

The normaliser is a pure function exposed as `eom.harness.normalise(text) -> str`. Compilers must apply it before computing offsets; validators apply it before checking.

### 4.5 What the harness does not claim

- A passing document is **EOM-conformant**, not necessarily **good**.
- The harness does not specify the highest-quality EOM; H13/H14 evaluate quality at corpus scale.
- The harness does not specify authoring style or aesthetic.

---

## 5. Reference schema

```python
class EOMDocument:
    version: Literal["0.1"]
    document_type: Literal["memo", "report", "paper", "transcript", "news", "policy", "other"]
    summary: str                          # one-sentence; for catalog/retrieval
    render_profile: Literal["executive_brief", "analytical_brief"]
    attention_budget: AttentionBudget     # B_A, B_AB token budgets
    blocks: list[Block]
    source: SourceMetadata                # checksum, chars, lang

class Block:
    id: str                               # slug, type-prefixed, unique in doc (e.g. "claim-3")
    type: Literal["headline", "lead", "claim", "evidence",
                  "factbox", "caveat", "decision", "appendix"]
    content: str
    attention_tier: Literal["A", "B", "C", "D"]
    priority: float                       # 0.0–1.0, intra-tier ranking only
    reading_order: int                    # 0..N-1
    source_span: SourceSpan | None        # required for evidence/factbox; required
                                          # for claim/decision unless is_inferred
    is_inferred: bool = False             # only valid for claim/decision
    inference_basis: list[str] = []       # block IDs (evidence/factbox)
    parent_id: str | None = None          # optional hierarchy

class SourceSpan:
    start: int                            # char offset in normalised source_text
    end: int                              # exclusive
    quote: str                            # exact substring; self-validates offsets

class AttentionBudget:
    B_A: int                              # max tokens at tier A
    B_AB: int                             # max tokens at tier A+B

class SourceMetadata:
    checksum: str                         # sha256 of normalised source_text
    chars: int                            # length of normalised source_text
    lang: str                             # ISO-639-1
```

### 5.1 Render profiles (v0.1)

| Profile | B_A | B_AB | Use case |
|---|---|---|---|
| `executive_brief` | 200 | 800 | One-page decision memo; phone reading |
| `analytical_brief` | 400 | 2000 | Two-page analyst report; deep review |

A document declares its profile; the validator looks up budgets unless the doc supplies an explicit `attention_budget` override.

### 5.2 Worked example

Source: a 600-word memo about a freight cost increase after a port closure (paraphrased here for brevity).

```json
{
  "version": "0.1",
  "document_type": "memo",
  "summary": "Q1 freight cost rose 9% after a port closure; recommend temporary rerouting.",
  "render_profile": "executive_brief",
  "attention_budget": {"B_A": 200, "B_AB": 800},
  "blocks": [
    {
      "id": "headline-1", "type": "headline",
      "content": "Q1 Freight Cost Up 9% After Port Closure",
      "attention_tier": "A", "priority": 1.0, "reading_order": 0,
      "source_span": {"start": 0, "end": 42, "quote": "Q1 freight cost rose 9% after a port closure"}
    },
    {
      "id": "lead-1", "type": "lead",
      "content": "On-time delivery dropped from 96% to 91%; recovery is expected within six weeks.",
      "attention_tier": "A", "priority": 0.95, "reading_order": 1,
      "source_span": {"start": 102, "end": 198, "quote": "<...>"}
    },
    {
      "id": "factbox-1", "type": "factbox",
      "content": "Freight: +9% • On-time: 96%→91% • Backlog: +14% • Recovery est.: 6 weeks",
      "attention_tier": "A", "priority": 0.9, "reading_order": 2,
      "source_span": {"start": 312, "end": 540, "quote": "<...>"}
    },
    {
      "id": "decision-1", "type": "decision",
      "content": "Approve temporary rerouting and a 2-week buffer-stock build.",
      "attention_tier": "A", "priority": 0.85, "reading_order": 3,
      "is_inferred": true,
      "inference_basis": ["factbox-1", "evidence-1"]
    }
  ],
  "source": {"checksum": "sha256:<…>", "chars": 612, "lang": "en"}
}
```

The full canonical example (with `evidence`, `caveat`, and `appendix` blocks) ships in `data/gold/freight_memo.eom.json` as part of the gold seed (Stage 1, §8.1).

### 5.3 What is deliberately out of the schema (and why)

- **`relations`** (cross-block links): doubles annotation cost and unnecessary to prove the dual-audience thesis. v0.2 candidate.
- **`render_hint`** (per-block desktop_slot, mobile_slot): renderer derives layout from `attention_tier + priority + reading_order`; revisit only if that proves insufficient.
- **`confidence`**: overlapping with `priority`; v0 uses `priority` only.
- **`audience`**: subsumed by `render_profile`.
- **`embedding`** / retrieval anchors: out of scope.
- **`llm_tier`** (separate compression tier for LLM consumption): collapsed into `attention_tier` for v0; if H13 evaluation shows the renderer is dropping critical context, add in v0.2.

---

## 6. Compiler

Three implementations behind one interface:

```python
class Compiler(Protocol):
    def compile(
        self,
        source_text: str,
        hints: CompileHints | None = None,
    ) -> EOMDocument: ...

class CompileHints(TypedDict, total=False):
    document_type: DocumentType
    audience: Literal["executive", "researcher", "general"]
    render_profile: Literal["executive_brief", "analytical_brief"]
    token_budget: int
```

### 6.1 `compiler_rules`

Pure Python, deterministic.

- Markdown AST parse via `markdown-it-py` → heuristic block segmentation.
- Rules: first H1 → `headline`; first paragraph after headline → `lead` candidate; numeric clusters / bullet lists with metrics → `factbox`; sentences with hedging language ("however", "limited", "may not", "depends on") → `caveat`; everything else → `claim` (early-position) or `appendix` (late-position) by length and position.
- Tier assignment: position-based + length-based heuristic; truncates content to compactness budgets.
- Passes structural and compactness invariants; cannot pass faithfulness or salience.
- Use cases: fallback, training-data scaffolding for the teacher, low-resource environments.

### 6.2 `compiler_prompted`

LLM-driven. Default backbone: Gemma-4-27B; configurable.

- Prompt = harness summary + schema reference + 3 few-shot exemplars from gold + the source.
- Output parsed into EOM JSON; if JSON is malformed, one re-attempt with explicit error message; then fall through to `compiler_rules` output annotated with `validation_status="rules_fallback"`.
- Used as: the week-6 baseline that `compiler_finetuned` must beat.

### 6.3 `compiler_finetuned`

Gemma-4 E2B fine-tuned via Unsloth.

- Same `Compiler` interface as the prompted variant.
- Smaller model, no few-shots needed at inference.
- The research artifact (G2). Trained on the synthetic dataset (§7).

### 6.4 Repair loop (LLM compilers only)

```
compile → validate → 
  if pass: return
  if fail: 
    summarise failures into actionable feedback,
    re-prompt model with feedback in context,
    try again, up to N=3 attempts
  if still fail after N: 
    return best-effort EOM with validation_status="partial" + failures attached
```

The repair loop is a runtime convenience, not part of the standard. A document is EOM-conformant iff the final output passes the harness, regardless of the number of attempts.

---

## 7. Renderers

Both deterministic; no model in the rendering loop.

### 7.1 `render_newspaper(eom) -> str`

HTML/CSS string, generated via Jinja2 templates.

- Layout grid: hero region (top), main column (left/centre), rail (right), archive (collapsed at bottom).
- Tier A → hero + above-fold; Tier B → main column; Tier C → rail / collapsed; Tier D → omitted from primary view, linked at bottom as "Archive (n blocks)" expandable.
- Within tier, blocks are ordered by `priority` desc, then `reading_order`.
- Print-ready CSS; can be exported as PDF without changes.
- Source spans rendered as superscript footnote markers; clicking reveals original text.

### 7.2 `render_context_pack(eom, token_budget) -> str`

Text payload optimised for LLM ingestion.

- Greedy-fills budget by tier: include all tier A; then tier B until budget tight; then tier C as one-line summaries; tier D omitted.
- Format: section-headed plaintext (`## headline`, `## lead`, `## decisions`, `## evidence`, `## caveats`, `## appendix`).
- Inline source citations: `[src:b3]` after each sentence pointing to block ID. Downstream LLM can request the verbatim quote via the EOM JSON's `source_span.quote` if needed.
- Always-included headers: `headline`, `lead`, summary, source token count, compression ratio.

---

## 8. Dataset and training pipeline

### 8.1 Stage 1 — Gold seed (weeks 1–2)

- 30–50 hand-curated `(source_text, ideal_eom)` pairs across all seven `document_type` values.
- Source pool: Wikipedia (CC-BY-SA), IETF RFCs, IPCC reports, PubMed abstracts, US federal register, public PRDs.
- Workflow: run `compiler_rules` for scaffold → hand-correct → hand-validate harness → commit.
- Estimated 30–60 minutes per example.

### 8.2 Stage 2 — Synthetic via privileged-context teacher (weeks 7–9)

The privileged-context teacher pattern:

```
Teacher input:                       Student input:
  source_text                          source_text
  + heading hierarchy
  + entity scaffolding
  + sentence salience scores
  + harness rules in prompt
  + validator feedback (repair loop)
```

Teacher: Gemma-4-27B (with optional frontier-LLM assistance on a small slice for quality calibration).
Student (training target): Gemma-4 E2B.

Pipeline (`scripts/generate_synthetic.py`):
1. Sample a source from the unlabeled pool (~10k raw documents).
2. Compute scaffolding (deterministic: AST parse, NER via spaCy, sentence salience via a small extractive summariser).
3. Teacher produces EOM JSON given source + scaffolding + harness rules + 3 gold few-shots.
4. Validator runs H1–H12. Initial pass rate target: 60–80%.
5. Repair loop, up to 3 attempts. Post-repair pass rate target: ≥95%.
6. Discard all failures. Only conformant pairs enter the training set.
7. Targets: 500 by end of week 7, 5,000 by end of week 9.

License-clean source pool:
- English Wikipedia (50k diverse stubs).
- US federal register / GovInfo (public domain).
- IPCC AR6 reports + WHO publications (open access).
- arXiv abstracts (CC-BY subset).
- Public PRDs / IETF RFCs / W3C specs.

A held-out set of 200 documents never enters training. Used for H13/H14 corpus evaluation.

### 8.3 Stage 3 — Fine-tune (weeks 9–11)

**3a — SFT** via Unsloth on Gemma-4 E2B.
- LoRA, rank 16–32, alpha 32. Standard Unsloth recipe.
- Loss: cross-entropy on the EOM JSON output sequence; token weighting biases the loss toward `attention_tier` and `priority` decisions.
- Hardware: single A100 (80GB) or Kaggle TPU-VM equivalent.
- Eval each epoch: H1–H12 pass rate on the 200-doc held-out.
- Stop when held-out pass rate plateaus (~3–5 epochs expected).

**3b — RLVR on harness pass rate** (week 11, optional).
- The validator is the verifier — no learned reward model.
- Reward = `Σ_i w_i · 1[H_i passed] + λ · compactness_score - μ · faithfulness_failures`. Faithfulness failures (H11, H12, H13b) are penalised hardest.
- GRPO via Unsloth RL extensions.
- Skip if SFT alone hits ≥85% held-out pass rate.

---

## 9. Evaluation

Three layers, in order of cost:

1. **Per-document validator** (H1–H12). Fast. Runs on every compile and every synthetic pair.
2. **Corpus-level evaluators** (H13, H13b, H14). Slow. Runs at milestones (after baseline, after SFT, after RLVR).
3. **Comparator evals** (the thesis test). Runs at end of weeks 6, 10, 12. Compares EOM context pack to baselines: raw truncation, top-k extractive (LexRank), LLM summary, RAG chunks. Metric: downstream QA accuracy at fixed token budgets.

### 9.1 Success criteria for v0.1

| # | Criterion | Threshold |
|---|---|---|
| 1 | `compiler_finetuned` H1–H12 pass rate on held-out gold | ≥ 95% |
| 2 | `compiler_finetuned` beats `compiler_prompted` on H1–H12 pass rate | strictly greater |
| 3 | EOM context pack beats raw-truncation baseline on downstream QA | ≥ 5pp absolute, at 3k budget |
| 4 | H13 monotonicity holds (tier-A-first > random > reading-order) | statistically significant on 200-doc held-out |

Outcome interpretation:
- **Hit all 4** → ship the standard publicly; if a Gemma hackathon window aligns, ship a domain-specialised fine-tune.
- **Hit 2–3** → ship the standard with explicit caveats; defer hackathon submission.
- **Hit fewer than 2** → extend the timeline; revisit dataset and harness design before publication.

---

## 10. Public API

### 10.1 CLI

```bash
eom compile      --input doc.md --output doc.eom.json [--profile executive_brief] [--compiler prompted]
eom validate     --eom doc.eom.json --source doc.md
eom render       --eom doc.eom.json --target newspaper --output doc.html
eom render       --eom doc.eom.json --target context-pack --budget 3000 --output doc.txt
```

### 10.2 Python

```python
from eom import compile, validate, render_newspaper, render_context_pack

eom = compile(source_text, hints={"document_type": "memo", "render_profile": "executive_brief"})
report = validate(eom, source_text)
html = render_newspaper(eom)
text = render_context_pack(eom, token_budget=3000)
```

---

## 11. Repository structure

```
eom/
├── README.md
├── pyproject.toml                    # Python 3.11, uv-managed
├── docs/
│   ├── superpowers/specs/            # design docs (this file lives here)
│   ├── harness-spec.md               # the standard, RFC-style
│   └── schema-spec.md                # reference encoding
├── eom/                              # python package
│   ├── schema.py                     # Pydantic models
│   ├── harness.py                    # H1-H12 validators + normalise()
│   ├── evaluators.py                 # H13, H13b, H14 corpus evals
│   ├── compilers/{rules,prompted,finetuned}.py
│   ├── renderers/{newspaper,context_pack}.py
│   ├── repair.py
│   └── cli.py
├── scripts/
│   ├── generate_synthetic.py
│   ├── train_sft.py                  # Unsloth + LoRA
│   ├── train_rlvr.py                 # optional GRPO
│   └── eval_corpus.py
├── data/
│   ├── gold/                         # 30-50 hand-curated, committed
│   ├── synthetic/                    # generated; gitignored
│   ├── eval/                         # held-out 200; committed
│   └── domain/education/             # future: hackathon gold set
├── templates/newspaper.html          # Jinja2
├── tests/{test_harness,test_schema,test_compilers,test_renderers}.py
└── notebooks/
    ├── 01-explore-gold.ipynb
    ├── 02-prompted-baseline.ipynb
    ├── 03-train-sft.ipynb            # Kaggle / Colab runnable
    └── 04-eval-corpus.ipynb
```

---

## 12. Timeline (12 weeks, starting 2026-05-02)

| Phase | Weeks | Calendar | Deliverable |
|---|---|---|---|
| Standard | 1–4 | May 2 – May 30 | Harness spec doc, schema, validator (H1–H12), gold seed (30–50), `compiler_rules`, both renderers, `compiler_prompted` + repair loop |
| Research | 5–9 | May 31 – Jul 4 | Prompted baseline measurements, synthetic generation pipeline, 5k training pairs |
| Fine-tune | 9–11 | Jul 5 – Jul 25 | Gemma-4 E2B SFT via Unsloth, optional RLVR, eval pass |
| Publish | 11–12 | Jul 26 – Aug 8 | RFC-style harness doc, GitHub release, blog post / writeup |

### Note on the Gemma 4 Good Hackathon (May 18 2026)

The hackathon deadline falls in week 3 of this plan. Under the 12-week schedule, no fine-tuned converter exists by then. We accept this trade-off: the standard is the durable contribution; chasing a hackathon submission with a half-formed schema would compromise G1.

If a future Gemma hackathon window aligns with weeks 11–12 (or the v0.2 cycle), the education-domain specialisation can be added as an additional Stage-3c fine-tune (~200 domain gold examples + 1–2 epoch fine-tune). The current plan keeps the dataset structure compatible (`data/domain/education/` reserved).

---

## 13. Risks and mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Gold seed quality bottleneck — 40 examples take >2 weeks | Medium | High | Drop to 30 examples with broader domain coverage; expand later if needed |
| R2 | Synthetic pipeline yields <30% post-repair pass rate | Medium | High | Invest more in teacher prompt engineering at week 7; expand gold seed to ~100 if quality stays low |
| R3 | Compute unavailable for SFT (no A100 access) | Low | Medium | Fall back to Kaggle TPU-VM or Colab Pro+; E2B is small enough to fit |
| R4 | H13 monotonicity fails empirically (the salience invariant doesn't hold) | Medium | High | Diagnose: is it the tier labels or the budget filling? Adjust `priority` weighting in the renderer; if invariant cannot be made to hold, the standard's central claim is wrong and we revise the harness |
| R5 | Held-out pass rate gap between `compiler_prompted` and `compiler_finetuned` is small or negative | Medium | Medium | Investigate distribution shift between gold and synthetic; check overfitting to teacher artefacts; consider increasing gold seed |
| R6 | Teacher (Gemma-4-27B) produces low-quality scaffolding-grounded EOM | Low | Medium | Use a frontier LLM (e.g., Claude / GPT) for the early gold-seed scaffolding to bootstrap quality before scaling synthetic with Gemma-4 |

---

## 14. Open questions deferred to v0.2

- Whether to add a typed `relations` layer (`supports / contradicts / qualifies`).
- Whether to add per-block `render_hint` slots, or whether tier+priority+reading-order is sufficient.
- Whether to add a separate `llm_tier` distinct from `attention_tier`.
- Multimodal source spans (image bounding boxes, table cells).
- Spatial / 3D rendering ("Spatial EOM scene graph").
- Domain specialisation as a first-class concept (currently handled by render_profile + dataset).
- Multilingual harness invariants (current spec assumes English; H8's "60 words" needs language-specific tuning).

---

## 15. Out of scope clarifications

- **EOM is not a chunking strategy for RAG.** Block boundaries are editorial, not retrieval-optimal. RAG anchors are a separate layer that may consume EOM but is not specified here.
- **EOM is not a summarisation system.** A summariser produces shorter prose; EOM produces a structured graph of editorial blocks with provenance and salience. The two are complementary.
- **EOM is not opinionated about authoring.** Users continue writing markdown / prose / chat output. The compiler does the work.
