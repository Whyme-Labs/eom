# Sentence-level scaffolding A/B — 2026-05-07

## Hypothesis

Paragraph-only scaffolding lifted overall yield from 33% → 68% but left a 48%
yield on arXiv papers (vs 77% on policy/RFCs). Diagnosis: arXiv abstracts are
typically one big paragraph containing 5-10 distinct claims/results.
Paragraph-level reference spans force the teacher to either cite the whole
abstract (oversized) or invent sub-paragraph offsets (H11 mismatch).

## Change

Extended `eom/compilers/scaffolding.py` so paragraphs ≥200 chars with multiple
sentences also emit individual `sentence` reference spans alongside the
`paragraph` span. The teacher can now cite a single result statement rather
than the whole abstract.

## A/B test

Re-ran 8 of the 21 previously-failed arXiv papers with sentence-level
scaffolding. The 7th call hung on a slow OpenRouter response and was
interrupted before the 8th started, so this is **6/8 attempted, 1 lost
to network**.

| Slug | Result | Attempts | Latency | Failed rule |
|---|---|---|---|---|
| arxiv-260505206v1 | FAIL | 3 | 314s | H3 |
| arxiv-260504759v1 | **PASS** | 2 | 388s | — |
| arxiv-260501591v1 | **PASS** | 1 | 47s | — |
| arxiv-260504952v1 | FAIL | 3 | 91s | H3 |
| arxiv-260504913v1 | **PASS** | 1 | 35s | — |
| arxiv-260505138v1 | FAIL | 3 | 141s | H3 |

**Yield: 3/6 = 50%** with sentence-level scaffolding on these previously-
impossible papers. Was 0/6 without any scaffolding.

## Key finding: failure mode shifted H11 → H3

All 3 surviving failures are now H3 (tier distribution caps), not H11
(source-span mismatch). The teacher is no longer inventing offsets — but
with more granular spans available, it tags more blocks as load-bearing
than the H3 cap permits.

This is a *different* problem and arguably an easier one: H3 violations
are arithmetic over block tiers, fixable by post-processing
(re-distribute tiers by priority) or by stronger prompt guidance about
the tier caps.

## Conclusion

Sentence-level scaffolding **eliminates the H11 dominant failure mode**
on the hardest documents. The remaining 50% paper failures have a
clean, addressable root cause (over-tagging tier A/B) rather than the
opaque "model can't compute char offsets" issue.

Recommended next step: post-process the teacher's output to enforce H3
tier caps (demote lowest-priority A/B blocks until under cap) — turn
H3 from a hard failure into a deterministic fix-up, similar to what
RulesCompiler._enforce_tier_caps does.
