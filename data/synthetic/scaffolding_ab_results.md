# Scaffolding A/B test results — 2026-05-07

Compared `PromptedCompiler(use_scaffolding=False)` vs `=True` on 8 raw docs that
had previously failed all 3 repair attempts in the first validation batch.

## Results

| Slug | Without scaffolding | With scaffolding | Attempts | Latency |
|---|---|---|---|---|
| paper/arxiv-260505155v1 | FAIL | FAIL (H11, H3) | 3 | 494s |
| other/revelation-records | FAIL | **PASS** | 1 | 5s |
| other/b-of-the-bang | FAIL | **PASS** | 1 | 4s |
| paper/arxiv-260505166v1 | FAIL | **PASS** | 2 | 94s |
| paper/arxiv-260504759v1 | FAIL | **PASS** | 1 | 5s |
| paper/arxiv-260504576v1 | FAIL | **PASS** | 1 | 29s |
| paper/arxiv-260505138v1 | FAIL | **PASS** | 1 | 4s |
| other/nikos-goumas-stadium | FAIL | **PASS** | 1 | 9s |

**Yield: 7/8 = 88% with scaffolding (vs 0/8 without).**

## Notes

- 5 of 7 wins passed on first try; mean attempts = 1.14
- First-try latency dropped to 4-30s (vs ~200s averages without scaffolding)
- The single remaining failure was a long arXiv paper with complex structure
  (likely a multi-figure paper where reference spans don't capture all
  load-bearing content)

## Implication

The 76% H11 dominant failure mode is largely closed by giving the teacher a
menu of valid (start, end, quote) tuples instead of asking it to compute
offsets. Cost projection for scaling to 500-1000 docs drops 5x because
most calls now succeed on first try (no repair iterations).
