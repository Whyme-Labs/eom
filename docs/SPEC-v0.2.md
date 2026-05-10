# EOM v0.2 — Specification (draft)

> **EOM is a two-way wire format between humans and models, with attention budgets and source grounding built in.**

Markdown is fine for storage. It is not for dialogue. Both directions of the human–AI exchange today are flat byte streams that force the receiver to re-derive salience, grounding, and structure. EOM carries those signals on the wire.

## 1. Architectural shape: asymmetric

EOM has one shared **core IR** and two asymmetric **dialects**. Each dialect has its own primitives and lowering targets; both share the same validator.

```
                       [ Core IR ]
                            |
           +----------------+----------------+
           |                                 |
    Outbound dialect                  Inbound dialect
    (AI -> Human)                     (Human -> AI)
           |                                 |
   Lowerings:                        Lowerings:
   - HTML newspaper                  - LLM context-pack
   - mobile cards                    - retrieval payload
   - slide deck                      - tool-call payload
   - voice script                    - cross-doc graph
```

Both dialects are valid EOM. Both pass H1–H12. They differ in which optional fields they populate and which projections they target.

## 2. Core IR (shared, additive over v0.1)

The v0.1 schema is the foundation. v0.2 is **strictly additive** — every v0.1 document is a valid v0.2 document. New fields are optional with sensible defaults.

### 2.1 Block (existing, v0.1)

```
Block:
  id: slug
  type: headline | lead | claim | evidence | factbox | caveat | decision | appendix
  content: string
  attention_tier: A | B | C | D
  priority: 0.0..1.0
  reading_order: int
  source_span: { start, end, quote } | null
  is_inferred: bool
  inference_basis: [block_id, ...]
  parent_id: block_id | null
```

### 2.2 Block (new in v0.2, optional)

```
  relations: [Relation, ...]    # typed edges to other blocks (see 2.3)
  role_tag: ground_truth | claim | speculation | citation   # inbound dialect signal
  evidence_layer: surface | drill | archive                 # collapse policy
```

### 2.3 Relation (new)

Typed directed edge between blocks. Replaces v0.1's untyped `inference_basis: [str]` with a typed algebra. v0.1 docs have `inference_basis` only; v0.2 docs may have either or both during the migration window.

```
Relation:
  type: supports | qualifies | contradicts | derived_from | cites | refines
  target: block_id          # the *other* block this edge points to
  confidence: 0.0..1.0      # optional, default 1.0
```

### 2.4 EOMDocument (new top-level fields, optional)

```
  dialect: outbound | inbound   # default: outbound (matches v0.1 behaviour)
  schema_version: "0.2"         # version sentinel
  retrieval_meta: { source_uri, retrieved_at, doc_kind, jurisdiction } | null
  system_intent: question | summarize | extract | compare | decide | null
  cross_doc_refs: [eom_doc_id, ...]
```

`attention_budget` (v0.1) gains an optional `token_budget: int` field for hard caps when the consumer is an LLM.

## 3. Outbound dialect (AI -> Human)

**Use when** an AI produces an artefact for a human reader. This is the v0.1 default.

**Required:**
- Every block has `attention_tier`, `priority`, `reading_order`.
- Every non-inferred block has a `source_span`.
- `render_profile` is set on the document (`executive_brief` | `analytical_brief`).

**Lowering targets:**

| Target | Module | Notes |
|---|---|---|
| HTML newspaper | `eom.renderers.newspaper` | hero/lede/body/archive layout, source-span hover |
| Mobile cards | (planned) | one card per Tier-A/B block, swipe for archive |
| Slide deck | (planned) | one slide per decision/headline, evidence as speaker notes |
| Voice script | (planned) | linear by `reading_order`, drop Tier-C/D |

**Reader contract:** humans should be able to act on the output without reading the source. Source spans exist so they can audit selectively.

## 4. Inbound dialect (Human -> AI)

**Use when** a human (or upstream pipeline) prepares a document for an AI consumer.

**Required:**
- `dialect = "inbound"`.
- `attention_budget.token_budget` is set.
- Every block has a `role_tag`.
- `system_intent` is set on the document.

**Lowering targets:**

| Target | Module | Notes |
|---|---|---|
| LLM context-pack | `eom.renderers.context_pack` | token-ordered, role-tagged, evidence collapsed by `evidence_layer` |
| Retrieval payload | (planned) | block-level chunks for vector index, with relations preserved |
| Tool-call payload | (planned) | per-tool slices keyed by `system_intent` |
| Cross-doc graph | (planned) | merge multiple inbound EOMs by `cross_doc_refs` |

**Receiver contract:** the LLM should be able to answer the `system_intent` from the context-pack alone, with citations resolving via `source_span`. No re-derivation of salience needed.

## 5. Validator (shared)

H1–H12 from v0.1 apply unchanged. Two new rules cover the new dialect-specific contracts.

| Rule | Direction | Check |
|---|---|---|
| H13 (inbound) | inbound | every block has `role_tag`; total context-pack tokens ≤ `token_budget`; per-block tokens ≤ `token_budget // 4` |
| H14 (outbound) | outbound | every required `visual_role` slot for the active `render_profile` has at least one block; if `len(source_chars) > B_AB`, archive section is non-empty |

H13/H14 are reported but not blocking in v0.2. They go gating once the benchmark calibrates the thresholds.

## 6. v0.1 -> v0.2 migration

Strictly additive. Migration is a no-op for existing documents:

1. v0.1 docs without `dialect` are read as `outbound` (the v0.1 default behaviour).
2. v0.1 docs without `schema_version` are tagged `0.1`. The validator runs the v0.1 ruleset only.
3. v0.2 docs use `schema_version: "0.2"` explicitly.
4. `inference_basis` (v0.1) and `relations` (v0.2) coexist during the migration window. New tooling reads both; new compilers emit `relations` and leave `inference_basis` empty.

## 7. What this lets us claim

Two demoable claims, both grounded in the same IR:

**Claim 1 (inbound — bandwidth efficiency).** Given a long document, the EOM context-pack at budget `B` produces strictly more answerable Q&A coverage per token than the raw document at budget `B`, with strictly higher citation rate. Measured on the benchmark in §8.

**Claim 2 (outbound — reading-time efficiency).** Given the same EOM document, the newspaper-brief lowering produces a target-reader comprehension score equivalent to reading the full source, in less than one-fifth of the reading time. Measured by H1–H12 pass-rate plus a reading-time stopwatch.

## 8. Benchmark (Day 4)

Side-by-side, same downstream model:

```
Doc D, Q-set Q with N questions.

Cell 1: raw_text(D) -> LLM -> answers
Cell 2: eom_inbound(D, budget=B).context_pack -> LLM -> answers

Score per cell:
  - input_tokens
  - answers_correct (LLM-as-judge or rubric)
  - citations_resolved   # answer cites a source_span that survives in source
  - hallucinations       # answer asserts something not in source
```

Targets D:
- 3 from existing gold corpus (RFC, paper, README)
- 1 long doc (~30k tokens; full RFC) so the token-savings claim is load-bearing

Downstream LLM: Anthropic Claude (Sonnet 4.6) and OpenRouter Gemma-4-31B, run on both cells, average results.

## 9. Implementation status

| Component | Status |
|---|---|
| Core IR (v0.1) | ✅ shipped, `eom/schema.py` |
| H1–H12 validator | ✅ shipped, `eom/harness.py` |
| Outbound HTML lowering | ✅ shipped, `eom/renderers/newspaper.py` |
| Outbound context-pack lowering | ✅ shipped, `eom/renderers/context_pack.py` (used here for inbound) |
| Outbound prompted compiler | ✅ shipped, `eom/compilers/prompted.py` |
| Outbound fine-tuned compiler | ✅ Stage-2 G3-1B (5/5 schema-valid); v5 G4-E4B (Unsloth track) |
| v0.2 schema additions (§2) | 🚧 Day 3 |
| H13 / H14 | 🚧 Day 3 |
| Inbound benchmark (§8) | 🚧 Day 4 |
| Bidirectional Streamlit | 🚧 Day 5 |
| Mobile / slides / voice / retrieval / tool-call lowerings | 📅 post-hackathon |

## 10. What v0.2 deliberately does not solve

- **Semantic correctness of the source.** EOM does not claim a document is true. It claims the document is structurally well-formed, attention-budgeted, and source-grounded.
- **Universal natural-language to formal-spec translation.** The compiler frontends are scoped to a fixed set of document kinds (memo, report, paper, transcript, news, policy, other).
- **Multi-document reasoning beyond cross-doc references.** `cross_doc_refs` is a pointer, not a graph algebra. v0.3 territory.
- **Equivalence and canonicalisation.** Two EOM documents may convey the same content with different block IDs and orderings. Diffing across runs is not yet defined.
