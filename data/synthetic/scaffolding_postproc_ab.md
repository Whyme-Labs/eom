# Post-processor A/B — 2026-05-07

## Hypothesis

Sentence-level scaffolding eliminated the H11 dominant failure mode but
exposed H3 (tier distribution caps) as the new bottleneck (3/6 = 50% on
hardest arXiv papers, all 3 surviving failures were H3).

H3 is deterministic — given priorities, demoting the lowest-priority
excess A/B blocks until under cap is mechanical. Doing this *after* the
LLM call (rather than re-prompting) should turn H3 from a hard failure
into a clean fixup.

## Change

Extracted `enforce_tier_caps` and `enforce_token_budget` from
`RulesCompiler` into a shared `eom/compilers/post_process.py`. Both
RulesCompiler and PromptedCompiler now call them. PromptedCompiler
applies them between schema validation and harness validation.

## A/B test

Same 6 arXiv papers as the sentence-scaffold A/B:

| Slug | Result | Attempts | Latency |
|---|---|---|---|
| arxiv-260505206v1 | **PASS** | 1 | 82s |
| arxiv-260504759v1 | **PASS** | 1 | 15s |
| arxiv-260501591v1 | **PASS** | 1 | 83s |
| arxiv-260504952v1 | **PASS** | 1 | 278s |
| arxiv-260504913v1 | **PASS** | 1 | 97s |
| arxiv-260505138v1 | **PASS** | 1 | 617s |

**Yield: 6/6 = 100%, all on first attempt.**

## Progression on the same 6 hardest papers

| Stage | Yield |
|---|---|
| No scaffolding | 0/6 = 0% |
| Paragraph scaffolding | ~0% (these were excluded from earlier 88% A/B) |
| Sentence scaffolding only | 3/6 = 50% |
| Sentence scaffolding + post-processor | **6/6 = 100%** |

## Implication

The combination eliminates both dominant failure modes (H11, H3) on the
hardest documents. Surviving failures in the broader 87-doc pool should
be limited to:
- H8 size-limit on rare extreme cases (long titles, lead overflow)
- True teacher errors on weird sources (e.g., disambiguation pages)

Re-running the full 87-doc batch with both improvements should produce
≥80%+ pool yield.
