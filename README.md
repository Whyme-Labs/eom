# EOM — Editorial Object Model

> **EOM is a two-way wire format between humans and models, with attention budgets and source grounding built in.**

Markdown is fine for storage. It is not for dialogue. EOM carries
salience, grounding, and structure on the wire so neither side has to
re-derive them every turn.

```
                  [ Core IR ]
                       |
       +---------------+---------------+
       |                               |
 Outbound dialect              Inbound dialect
 (AI -> human)                 (human -> AI)
       |                               |
 HTML newspaper                LLM context-pack
 mobile cards                  retrieval payload
 slide deck                    tool-call payload
```

One core IR, two asymmetric dialects, one shared validator.
[Spec at `docs/SPEC-v0.2.md`](docs/SPEC-v0.2.md).

This repo is the submission for the
**[Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon)**,
Unsloth track. Writeup at [`docs/KAGGLE-WRITEUP.md`](docs/KAGGLE-WRITEUP.md);
3-minute video script at [`docs/VIDEO-SCRIPT.md`](docs/VIDEO-SCRIPT.md).

**Live demo (Cloudflare Workers Suite):**
[https://eom-demo.swmengappdev.workers.dev](https://eom-demo.swmengappdev.workers.dev)
— Pages + Workers + R2 (gold corpus) + D1 (qsets + benchmark results) + KV
(pack cache) + Workers AI binding + OpenRouter (inbound LLM).

## Quickstart

```bash
# Python 3.11 + uv
uv venv && uv sync --extra dev

# Run the bidirectional Streamlit demo
export OPENROUTER_API_KEY=sk-or-...   # for the prompted compiler + Ask AI tab
uv run streamlit run demo/app.py
# open http://localhost:8501
```

The demo's tabs cover both directions on the same compiled IR:

- **📰 Newspaper** — outbound HTML brief, hero/lede/body/archive
- **🤖 Context pack** — inbound LLM payload, token-budgeted
- **📋 JSON** — the IR
- **✓ Harness** — H1-H12 pass/fail with metrics
- **🔄 Ask AI** — live raw-vs-pack comparison: same model, same question, two contexts

## Inbound benchmark

Three documents, fifteen questions, two cells per question. Same
downstream model on both cells; Claude Sonnet 4.5 as judge.

| Doc | Raw -> Pack tokens | Compression | Score (raw / pack, /2) |
|---|---|---|---|
| GDPR | 5,159 -> 3,239 | 0.63x | 2.00 / 1.20 |
| Paris 2024 | 7,488 -> 2,348 | **0.31x** | 2.00 / 1.20 |
| RFC 9293 (TCP)* | 3,485 -> 2,090 | 0.60x | 0.40 / 0.40 |
| **Total** | **16,132 -> 7,677** | **0.48x** | 1.47 / 0.93 |

\* the TCP gold doc is itself a summary that doesn't contain the
questioned details — both modes score the same low number, isolating the
EOM-vs-raw effect to the other two rows.

EOM cuts input by **52%** across the benchmark. The pack is editorially
lossy by design: Tier-A always survives, Tier-C compresses to one-line
summaries, Tier-D is dropped. On documents whose source contains the
answer (GDPR, Paris-2024), the pack preserves 3/5 high-salience
questions and drops 2/5 tail-detail.

```bash
uv run python -m bench.inbound          # full benchmark
uv run python -m bench.inbound --no-judge --docs gdpr   # cheap smoke
```

Raw rows in `data/bench/results/<run-id>.json`; summary table in
`<run-id>.md`.

## Unsloth-track fine-tune

`scripts/modal_train_gemma4_v[2-5].py` — five iterations of fine-tuning
Gemma-4-E4B via Unsloth on Modal. v5 is the canonical recipe:
`FastModel`, `bf16`, `r=32/alpha=32`, 30 epochs, `lr=1e-4`. Adapter
saved to Modal volume `eom-sft-out:/output/eom-sft-adapter-gemma4-v5`.

The adapter is the IR frontend — raw text in, EOM out — for offline
inbound compilation. The schema doesn't fully crystallise on v5 (the
training prompt leaks input metadata fields as output keys); the
writeup documents the failure mode honestly. Stage-2 Gemma-3-1B (a
smaller model with a weaker instruction prior) crystallises the schema
cleanly on the same data.

```bash
modal run scripts/modal_train_gemma4_v5.py    # 60-90 min on A100-80GB
modal run scripts/modal_eval_gemma4_v5.py     # post-train salvage / val gens
```

## CLI

```bash
# Compile a markdown document with the deterministic rules compiler
uv run eom compile -i my.md --compiler rules --document-type memo -o my.eom.json

# Or via Gemma-4-31B over OpenRouter
uv run eom compile -i my.md --compiler prompted -o my.eom.json

# Validate against H1-H12
uv run eom validate --eom my.eom.json --source my.md

# Render outbound (HTML newspaper) or inbound (LLM context-pack)
uv run eom render --eom my.eom.json --target newspaper --output my.html
uv run eom render --eom my.eom.json --target context-pack --budget 1000 --output my.txt
```

## Repository map

| Path | What |
|---|---|
| `docs/SPEC-v0.2.md` | Spec — abstract syntax, dialects, lowerings, H-rules, migration |
| `docs/KAGGLE-WRITEUP.md` | Hackathon writeup, ≤1500 words |
| `docs/VIDEO-SCRIPT.md` | 3-min video shot list |
| `eom/schema.py` | Pydantic schema (v0.1 + v0.2 additive) |
| `eom/harness.py` | H1-H12 validator |
| `eom/compilers/{rules,prompted,finetuned}.py` | Three compiler frontends |
| `eom/renderers/{newspaper,context_pack}.py` | Outbound + inbound lowerings |
| `eom/repair.py` | Compile-with-repair loop |
| `bench/inbound.py` + `data/bench/qsets.json` | Inbound benchmark |
| `demo/app.py` | Bidirectional Streamlit demo |
| `scripts/modal_train_gemma4_v[2-5].py` | Unsloth-track fine-tune iterations |
| `data/gold/` | 30+ hand-curated EOM examples passing H1-H12 |

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Schema, harness, rules + prompted compilers, two renderers, CLI, gold corpus | ✓ |
| 2 | Synthetic data + Stage-2 SFT (Gemma-3-1B, 5/5 schema-valid) | ✓ |
| 3 | Unsloth-track Gemma-4-E4B fine-tune (v5 adapter, schema partial) | ✓ |
| 4 | Bidirectional protocol (v0.2 spec + dialects + benchmark + demo) | ✓ |
| 5 | Post-hackathon: equivalence/canonicalisation, multi-doc graphs, more lowerings, RLVR loop | 📅 |

## License

MIT. See [`LICENSE`](LICENSE).
