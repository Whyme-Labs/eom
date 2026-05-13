# EOM — a two-way wire format between humans and models

*Submission for the Gemma 4 Good Hackathon, Unsloth track.*

## The problem

Markdown is fine for storage. It is not for dialogue.

Today, every human/AI exchange flows as a flat byte stream: walls of
Markdown going one way, walls of model output coming back. Both sides
re-derive what matters every turn — which sentence is the decision,
which is caveat, which is evidence. The receiver pays in attention. The
sender paid in tokens. Nobody encodes what they actually meant.

Two recent observations of the same pain. Thariq Shihipar [argues HTML
is winning over Markdown for agent output][thariq] because layout, density,
and interaction matter when the artefact is non-trivial. The OpenAI
discussion linked from this submission’s thread argues for [SpecLang][spec],
a formal intent layer above natural language for AI-agent tasks.

EOM is a third leg of the same stool. **EOM is a two-way wire format
between humans and models, with attention budgets and source grounding
built in.** Same IR carries the signal in both directions; lowerings
project it for the receiver in front of you.

## The architecture

One core IR, two asymmetric dialects, shared validator.

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

The core IR is a list of typed editorial blocks
(`headline`, `lead`, `claim`, `evidence`, `factbox`, `caveat`, `decision`,
`appendix`) plus an attention budget and source-span grounding for every
non-inferred block. Twelve formal validators (H1–H12) check structural
correctness, source-grounding integrity, and attention-budget
conformance.

The **outbound dialect** carries visual primitives — render profile,
visual roles, section hints — and lowers to HTML newspaper briefs,
mobile cards, and slide decks.

The **inbound dialect** carries model-context primitives — system intent,
role tags (`ground_truth` / `claim` / `speculation` / `citation`),
evidence layers, hard token budgets — and lowers to LLM context-packs,
retrieval payloads, and tool-call slices.

Same blocks underneath. Different projections out. v0.2 adds the dialect
layer and typed relations strictly additively over v0.1; existing
documents validate unchanged. Full spec at `docs/SPEC-v0.2.md`.

## Inbound demo: bandwidth numbers, not vibes

The headline result. Three documents, fifteen questions, two cells per
question — raw markdown vs EOM context-pack — same downstream model
(`google/gemma-4-31b-it` via OpenRouter), graded by Claude Sonnet 4.5
as judge.

| Doc | Raw -> Pack tokens | Compression | Score (raw / pack, /2) |
|---|---|---|---|
| GDPR | 5,159 -> 3,239 | 0.63x | 2.00 / 1.20 |
| Paris 2024 | 7,488 -> 2,348 | **0.31x** | 2.00 / 1.20 |
| RFC 9293 (TCP) | 3,485 -> 2,090 | 0.60x | 0.40 / 0.40 |
| **Total** | **16,132 -> 7,677** | **0.48x (52% reduction)** | **1.47 / 0.93** |

Two things to read from this table.

**One. EOM cuts input by 52% across the benchmark.** On the Paris-2024
document specifically, the pack is 31% the size of the raw — almost
70% reduction — because the source has long enumerative sections and
the EOM compiler correctly demotes them.

**Two. The pack is editorially lossy by design.** On GDPR and Paris-2024,
where the source actually contains the answers, the pack preserves three
of five high-salience questions perfectly and drops two of five
tail-detail questions (e.g., the GDPR breach-notification window). The
compiler made an editorial choice: those facts fell below the priority
threshold for the budget. That is the trade EOM is built around — Tier-A
content always survives, Tier-C is one-line summaries, Tier-D is dropped.
The benchmark surfaces the cost of that trade honestly.

The TCP row is the control. Both raw and pack score the same low number
because the gold-corpus source is itself a summary that doesn’t contain
the questioned details — neither cell can answer what the document
doesn’t say. That isolates the EOM-vs-raw effect to the other two rows.

Code at `bench/inbound.py`. Question sets at `data/bench/qsets.json`.
Results at `data/bench/results/`.

## Outbound demo: same IR, different lowering

The live demo at
[eom-demo.swmengappdev.workers.dev](https://eom-demo.swmengappdev.workers.dev)
shows both directions on the same compiled IR. Outbound lowering produces
a newspaper-style HTML brief with hero, lede, body, archive, and
source-span hover. The harness report next to it shows pass/fail per
H-rule and the metrics each contributed. All 32 documents in the gold
corpus pass H1–H12.

The outbound side is what most readers expect from “document AI today” —
a render. The inbound side is the wedge. Most existing tools treat the
artefact as the goal; EOM treats the IR as the goal and the artefact as
a projection.

The demo is deployed on the Cloudflare Workers suite: **Pages + Workers**
(static + edge functions), **R2** (gold corpus), **D1** (qsets +
benchmark history), **KV** (pack memoisation, hit/miss surfaced in
`x-eom-cache` headers), and **Workers AI** (binding reserved for an
inbound-model toggle). All public except the **🔄 Ask AI** tab, which is
BYO OpenRouter key — judges paste their own `sk-or-…` and it lives only
in browser localStorage. That keeps the demo zero-cost regardless of
traffic and the inbound benchmark numbers reproducible without any
shared budget asymmetry.

## Unsloth track: Gemma-4-E4B as the IR frontend

The compiler frontend is the part that takes raw text and emits EOM IR.
For the Unsloth track we fine-tuned Gemma-4-E4B on Modal.

Five iterations, one of which (v3) cracked a real puzzle: Gemma-4 uses
`<|turn>user/model` chat-template markers, not Gemma-3’s
`<start_of_turn>...`. The first version masked the wrong tokens and
produced flat-loss zero gradients; the second version mode-collapsed to
`{"eom": true}` because of fp16 underflow on the residual stream. The
canonical Unsloth recipe — `FastModel`, `bf16`, `r=8/alpha=8`, then
`r=16/alpha=16`, then the final v5 at `r=32/alpha=32` for thirty epochs
with `lr=1e-4` — trained cleanly with no collapse.

The schema didn’t fully crystallise on the v5 adapter: the model echoes
input metadata fields (`document_type`, `render_profile`) as output keys
rather than producing the EOM `attention_budget` + `blocks` envelope.
The root cause is prompt-format leakage, not model capacity — the
training prompt itself contained those fields in JSON-key shape, and
the strong instruction-tuned prior reflects them back. Stage-2
Gemma-3-1B (a smaller model with weaker prior, included for comparison)
crystallises the schema cleanly at twenty epochs on the same data.

Adapter saved to Modal volume `eom-sft-out:/output/eom-sft-adapter-gemma4-v5`.
Training scripts at `scripts/modal_train_gemma4_v[2-5].py`. We document
the failure mode honestly because the lesson — *prompt formats compete
with output schemas in fine-tuning, and a 4B-class instruction-tuned
prior beats a 100-example LoRA at template mimicry* — is exactly the
kind of finding the Unsloth community benefits from.

## Roadmap (post-hackathon)

- **v0.3 schema:** equivalence and canonicalisation, so two EOMs that
  convey the same content can be diffed reliably.
- **Multi-doc graphs:** `cross_doc_refs` (already in v0.2 schema) becomes
  a real graph algebra so working memory across documents is first-class.
- **More lowerings:** mobile cards, slides, voice scripts on the outbound
  side; retrieval payloads and tool-call slices on the inbound side.
- **Fine-tune frontend retry:** rebuild the training prompt so input
  metadata is not in JSON-key shape, then re-run v6 of the Gemma-4-E4B
  adapter. Estimated $3 and half a day.
- **RLVR loop:** H1–H12 failures already produce structured error
  messages used by `eom.repair`; the next step is to use them as a
  training signal, not just a gate.

## Repo

- Spec: `docs/SPEC-v0.2.md`
- Live demo: [`eom-demo.swmengappdev.workers.dev`](https://eom-demo.swmengappdev.workers.dev) (Cloudflare Workers Suite — Pages + Workers + R2 + D1 + KV + AI; BYO OpenRouter key for the inbound Ask AI tab)
- Local demo: `web/` (Wrangler) or `demo/app.py` (Streamlit)
- Benchmark: `bench/inbound.py` + `data/bench/qsets.json`
- Validator: `eom/harness.py` (H1–H12)
- Compilers: `eom/compilers/{rules,prompted,finetuned}.py`
- Renderers: `eom/renderers/{newspaper,context_pack}.py`
- Fine-tune scripts: `scripts/modal_train_gemma4_v[2-5].py`
- Adapter: Modal volume `eom-sft-out:/output/eom-sft-adapter-gemma4-v5`

Markdown is for storage. EOM is for dialogue.

[thariq]: https://x.com/trq212/status/2017024445244924382
[spec]: https://chatgpt.com/share/69ff6798-9f5c-83ec-8a7b-f26463d4abb8
