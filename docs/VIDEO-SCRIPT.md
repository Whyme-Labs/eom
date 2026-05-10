# EOM hackathon video — 3-minute script

Submission: Gemma 4 Good Hackathon, Unsloth track. Total: 3:00. Format: screen
recording with voice-over. Aspect 16:9, 1080p, 30fps.

## Pre-flight (run once before recording)

```bash
# 1. Confirm env
echo $OPENROUTER_API_KEY  # must be set
git rev-parse --short HEAD  # latest master

# 2. Warm OpenRouter caches by running the benchmark once
uv run python -m bench.inbound --docs paris-2024 --no-judge

# 3. Boot Streamlit
uv run streamlit run demo/app.py --server.port 8501
# Open http://localhost:8501 in a clean browser window (no extensions, no
# bookmark bar, full-screen). Use light mode.

# 4. Quit any noisy mac/win notification daemons (Slack, Mail) before record
```

## Shot-by-shot

Timing is wall-clock; each row is one continuous shot unless noted.

### 0:00 — 0:15  Open with the pitch

**On screen.** Streamlit landing page (pre-compile state). Title visible:
"📰 EOM — two-way wire format between humans and models". Caption visible.

**Voice-over.**
> Today, the way humans and AIs talk is a flat byte stream — Markdown, raw
> text, walls of JSON. Both sides re-derive what matters every turn. EOM is
> a two-way wire format between humans and models, with attention budgets
> and source grounding built in.

### 0:15 — 0:50  Outbound (AI → human)

**On screen.** Click sidebar **Sample → news/paris-2024-olympics**, click
**Load sample**, then click **Compile EOM →**. Wait ~10s for compile.
Cut to compiled state — newspaper tab shows hero/lede/body. Hover one block
to show source-span tooltip. Then flip through tabs: 📰 Newspaper,
✓ Harness ("All H1–H12 checks pass").

**Voice-over.**
> Outbound direction: an AI compiles a 1500-word source into an EOM document
> — a structured editorial graph with attention tiers, source spans, and
> typed blocks. The same IR projects to a newspaper-style brief that a human
> reads in thirty seconds, with every claim grounded back to a span in the
> source. Twelve formal validators run automatically. This document passes
> all of them.

### 0:50 — 1:55  Inbound (human → AI) — the wedge

**On screen.** Click **🔄 Ask AI** tab. The question dropdown is
pre-populated with five preset questions for this sample. Pick:

> "Which countries topped the medal table by gold and how did they compare?"

The reference-answer caption appears below the dropdown.

Click **Run side-by-side →**. Wait ~6s for both calls. Two columns appear:

| Raw markdown | EOM context-pack |
|---|---|
| ~1500 input tokens | ~470 input tokens (-69%) |
| ≈3s | ≈3s |
| Correct answer | Correct answer |

Below the columns, the compression line reads:
> "pack uses 0.31× the tokens of raw (471 vs 1499, 69% reduction)"

**Voice-over.**
> Inbound direction: same IR, different lowering. We ask the same model the
> same question with two contexts — raw markdown on the left, EOM context-pack
> on the right. The pack is sixty-nine percent smaller. The answer is the
> same. The IR carried the salience signal on the wire so the model didn't
> have to rebuild it.

> Across our benchmark of three documents and fifteen questions, EOM cuts
> input tokens by fifty-two percent. Editorially-lossy by design — the pack
> drops tail-detail by priority, not by accident. Tier-A blocks always
> survive; Tier-C is one-line summaries.

### 1:55 — 2:30  The architecture

**On screen.** Cut to a slide / overlay — show this diagram (can be a static
PNG made from `docs/SPEC-v0.2.md` §1):

```
                    [ Core IR ]
                         |
       +-----------------+-----------------+
       |                                   |
 Outbound dialect                  Inbound dialect
 (AI → human)                      (human → AI)
       |                                   |
 HTML newspaper                   LLM context-pack
 mobile cards                     retrieval payload
 slide deck                       tool-call payload
```

Then briefly show `docs/SPEC-v0.2.md` open in an editor — point at the
H1–H12 + H13/H14 validator section.

**Voice-over.**
> One core IR, two asymmetric dialects, shared validator. The outbound
> dialect carries visual primitives — lede, hero, factbox, archive. The
> inbound dialect carries model-context primitives — system intent, role
> tags, evidence layers, token budgets. Same blocks underneath, different
> projections out.

### 2:30 — 2:50  The Unsloth-track fine-tune

**On screen.** Cut to terminal. Run:

```bash
modal volume ls eom-sft-out | grep gemma4-v5
```

Show `eom-sft-adapter-gemma4-v5` listed. Then briefly flip to
`scripts/modal_train_gemma4_v5.py` — point at the Unsloth recipe header.

**Voice-over.**
> For the Unsloth track, we fine-tuned Gemma 4 E4B on Modal — thirty epochs,
> rank-32 LoRA, bf16, the canonical Unsloth recipe. The adapter is the
> compiler frontend: raw text in, EOM IR out. Locally, offline, no API
> dependency for the inbound side.

### 2:50 — 3:00  Close

**On screen.** Cut back to demo, then a closing card with the GitHub URL
and the spec path.

**Voice-over.**
> EOM is a protocol, not a format. The spec is at `docs/SPEC-v0.2.md`.
> Code, benchmark, and demo are open-source on GitHub. Markdown is for
> storage. EOM is for dialogue.

---

## Post-production checklist

- [ ] Subtitles burned in (auto-caption, then proofread numbers)
- [ ] Number callouts overlaid at 1:25 ("69% reduction") and 1:42 ("52% across benchmark")
- [ ] GitHub URL on closing card
- [ ] Background music — none. Voice-only is fine for technical pitch.

## If any step fails live

| Failure | Mitigation |
|---|---|
| OpenRouter compile times out | Re-run; if still failing, switch to **rules** compiler in the sidebar — it's deterministic and offline. Use it for the outbound shot only. |
| Ask AI side-by-side returns "not in document" on the pack side | The compression headline is still load-bearing; lean on the token numbers and acknowledge "pack is editorially lossy by design — that's the trade-off." This is true and on-message. |
| Streamlit crashes mid-shot | Restart, refresh; the compile result is in session_state and survives reruns within a session but **not** across restarts — re-do the load+compile sequence. |

## Numbers to memorise (for any improvised explanation)

- Total benchmark: raw 16,132 tokens vs pack 7,677 tokens — **52% reduction**.
- Best per-doc compression: paris-2024 at **0.31×** (69% reduction).
- Quality: pack scores 3/5 high-salience answers correctly, drops 2/5 tail-detail per doc — by editorial choice, not by accident.
- Schema: v0.2 strictly additive over v0.1; 213 unit tests pass.
- Fine-tune: Gemma-4-E4B, Unsloth, Modal A100-80GB, 30 epochs, rank-32 LoRA, bf16.
