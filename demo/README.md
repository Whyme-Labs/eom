# EOM live demo

Streamlit app for the **Editorial Object Model** standard. Paste any markdown source and watch a Gemma-4 model compile it into newspaper-style EOM JSON in real time, then render it three ways: as an HTML newspaper, as an LLM-friendly context pack, and as the raw schema.

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

| Tab | Content |
|---|---|
| Newspaper | Print-style HTML render with hero / grid / archive layout |
| Context pack | Token-budgeted text payload an LLM ingests in one shot |
| JSON | The raw EOM v0.1 document |
| Harness | Pass/fail per H1–H12, with metrics + warnings |

## What's running

- **Compiler**: `eom.compilers.prompted.PromptedCompiler` calling Gemma-4-31B via OpenRouter, plus a deterministic fall-back to the rules-based compiler.
- **Scaffolding**: `eom.compilers.scaffolding` extracts heading + paragraph + sentence reference spans from the source so the model picks valid offsets instead of inventing them.
- **Repair loop**: `eom.repair.compile_with_repair` re-prompts the model with structured failure messages on harness errors.
- **Post-processing**: `eom.compilers.post_process` enforces H3 tier caps and H9/H10 token budgets deterministically.
