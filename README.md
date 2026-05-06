# EOM — Editorial Object Model

A standard for representing documents that serves both human readers and language models.

## Quickstart

```bash
# Install (Python 3.11)
uv venv && uv sync --extra dev

# Compile a markdown document with the rules-based compiler
uv run eom compile \
    --input tests/fixtures/freight_memo.md \
    --compiler rules \
    --document-type memo \
    --output freight.eom.json

# Validate against the harness
uv run eom validate --eom freight.eom.json --source tests/fixtures/freight_memo.md

# Render
uv run eom render --eom freight.eom.json --target newspaper --output freight.html
uv run eom render --eom freight.eom.json --target context-pack --budget 1000 --output freight.txt

# Test the gold seed
uv run python scripts/validate_gold.py
```

For the prompted compiler (Gemma-4-27B via Together AI), set `TOGETHER_API_KEY`:

```bash
export TOGETHER_API_KEY=...
uv run eom compile -i my.md --compiler prompted --output my.eom.json
```

## Repository status (Phase 1 complete)

- ✓ Schema (Pydantic) and harness validator (H1-H12)
- ✓ Two compilers: `rules` (deterministic) and `prompted` (Gemma-4-27B)
- ✓ Two renderers: newspaper HTML and LLM context pack
- ✓ Repair loop with failure summarisation
- ✓ CLI: `eom compile / validate / render`
- ✓ 30+ hand-curated gold examples passing harness
- ☐ Phase 2: synthetic dataset generation pipeline
- ☐ Phase 3: Gemma-4 E2B fine-tune via Unsloth
- ☐ Phase 4: standard publication
