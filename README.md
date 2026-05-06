# EOM — Editorial Object Model

An attention-architecture standard for documents that serve both humans and AI.

EOM is a downstream representation: any markdown / prose source — whether human-authored or LLM-generated — compiles into a structured graph of editorial blocks (headline, lead, claim, evidence, factbox, caveat, decision, appendix) with explicit salience, source provenance, and a hard attention budget.

The standard is defined as a **harness** (versioned testable properties); the JSON shape is just one encoding that satisfies the harness.

**Status:** v0.1 in development. See [the design spec](docs/superpowers/specs/2026-05-02-eom-design.md) for full details.

## Quickstart (when Phase 1 ships)

```bash
uv sync
uv run eom compile --input examples/memo.md --compiler rules --output memo.eom.json
uv run eom render --eom memo.eom.json --target newspaper --output memo.html
uv run eom render --eom memo.eom.json --target context-pack --budget 3000 --output memo.txt
```

## Components

| Layer | Responsibility |
|---|---|
| Harness | Defines what makes a document EOM-conformant. Source of truth. |
| Schema | The simplest JSON encoding that satisfies the harness. |
| Compiler | Three implementations: rules-based, prompted-LLM, fine-tuned (Gemma-4 E2B). |
| Renderers | Newspaper HTML view + LLM context pack — both deterministic. |

## License

MIT.
