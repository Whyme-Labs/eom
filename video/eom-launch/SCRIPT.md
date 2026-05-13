# EOM — script

3:00 hackathon submission video. **Captions only, no voice-over.** Background
music + sound design. Every line below renders as one on-screen caption frame
synced to the visuals. Timing assumes the *reading* pace (~3 wps for captions)
rather than the spoken pace (~2.5 wps) — captions can be terser.

Word budget: ~360 words across 36 lines.

---

## Beat 1 — Hook (0:00 – 0:15)

Markdown was built for storage.

Not for dialogue.

Every chat with an AI is a flat byte stream — both sides re-derive what
matters every turn.

So we built a wire format for it.

## Beat 2 — Outbound (0:15 – 0:50)

EOM is a two-way wire format between humans and models, with attention
budgets and source grounding built in.

One IR. Two lowerings.

**Outbound** — AI to human.

The same EOM document projects to a newspaper brief.

Headline. Lede. Body. Archive. Every claim grounded to a span in the
source.

Twelve formal validators. H1 through H12. This document passes all of them.

## Beat 3 — Inbound (0:50 – 1:55)

**Inbound** — human to AI.

Same IR. Different lowering.

We ask the same model the same question — with two different contexts.

Left: raw markdown.

Right: the EOM context-pack.

Same answer. Smaller payload.

**0.31× the tokens.**

**69% reduction.**

Across our benchmark — three documents, fifteen questions — EOM cuts input
tokens by **52%**.

The pack is editorially lossy *by design*.

Tier-A always survives. Tier-C compresses. Tier-D drops.

High-salience answers preserved. Tail detail demoted by priority, not by
accident.

## Beat 4 — Architecture (1:55 – 2:30)

One core IR. Two asymmetric dialects. Shared validator.

Outbound carries visual primitives — lede, hero, factbox, archive.

Inbound carries model-context primitives — system intent, role tags,
evidence layers, token budgets.

Same blocks underneath. Different projections out.

Deployed on the Cloudflare suite — Pages, Workers, R2, D1, KV, Workers AI.

## Beat 5 — Fine-tune (2:30 – 2:50)

For the Unsloth track, we fine-tuned Gemma-4-E4B on Modal.

bf16. Rank-32 LoRA. Thirty epochs. Canonical Unsloth recipe.

The adapter is the IR frontend — raw text in, EOM out, offline.

## Beat 6 — Close (2:50 – 3:00)

Markdown is for storage.

EOM is for dialogue.

Spec → docs/SPEC-v0.2.md.

eom-demo.swmengappdev.workers.dev
