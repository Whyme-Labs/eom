# EOM live demo

Streamlit app for the **Editorial Object Model** — a two-way wire format
between humans and models. Paste any markdown source, watch Gemma-4
compile it into the EOM IR, then drive both directions live: the
outbound newspaper brief on one side, and the inbound LLM context-pack
benchmarked side-by-side against raw markdown on the other.

## Local run

```bash
# From repo root
uv sync --extra demo
export OPENROUTER_API_KEY=sk-or-...
uv run streamlit run demo/app.py
# Open http://localhost:8501
```

## Streamlit Cloud deployment

1. Push this repo to GitHub (public).
2. Sign in to [streamlit.io/cloud](https://streamlit.io/cloud) with your GitHub account.
3. New app → repo `<your-handle>/eom` → branch `main` → main file `demo/app.py`.
4. **Advanced settings → Secrets**: paste
   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   ```
5. Deploy. Cold-start takes ~2 minutes the first time (installs deps from `demo/requirements.txt`); subsequent loads are <5s.

## What it shows

| Tab | Direction | Content |
|---|---|---|
| 📰 Newspaper | outbound (AI → human) | Print-style HTML brief — hero / grid / archive |
| 🤖 Context pack | inbound (human → AI) | Token-budgeted text payload an LLM ingests in one shot |
| 📋 JSON | core IR | The raw EOM document |
| ✓ Harness | shared validator | Pass/fail per H1–H12, with metrics + warnings |
| 🔄 Ask AI | inbound (live) | Side-by-side raw vs context-pack on the same model — shows token cost, latency, and answer per column |

## What's running

- **Compiler**: `eom.compilers.prompted.PromptedCompiler` calling Gemma-4-31B via OpenRouter, plus a deterministic fall-back to the rules-based compiler.
- **Scaffolding**: `eom.compilers.scaffolding` extracts heading + paragraph + sentence reference spans from the source so the model picks valid offsets instead of inventing them.
- **Repair loop**: `eom.repair.compile_with_repair` re-prompts the model with structured failure messages on harness errors.
- **Post-processing**: `eom.compilers.post_process` enforces H3 tier caps and H9/H10 token budgets deterministically.
