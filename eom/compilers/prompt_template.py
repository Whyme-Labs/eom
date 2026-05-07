"""Prompt template for the prompted EOM compiler.

The system prompt teaches the model the schema and harness in compact
form. The user prompt provides few-shot exemplars + the source.
"""

from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent("""\
    You are an EOM compiler. EOM (Editorial Object Model) is a structured
    representation of documents that serves both human readers and language
    models. Your job is to convert the input document into a valid EOM JSON
    object.

    EOM JSON shape:
    {
      "version": "0.1",
      "document_type": "memo|report|paper|transcript|news|policy|other",
      "summary": "<one-sentence summary>",
      "render_profile": "executive_brief" | "analytical_brief",
      "attention_budget": {"B_A": <int>, "B_AB": <int>},
      "blocks": [
        {
          "id": "<type>-<n>",      // e.g. "claim-1"
          "type": "headline" | "lead" | "claim" | "evidence" |
                  "factbox" | "caveat" | "decision" | "appendix",
          "content": "<the block's text>",
          "attention_tier": "A" | "B" | "C" | "D",
          "priority": <float 0-1>,
          "reading_order": <int>,
          "source_span": {
              "start": <int char offset>,
              "end": <int char offset>,
              "quote": "<verbatim substring of source>"
          } | null,
          "is_inferred": <bool>,
          "inference_basis": [<block_id>, ...],
          "parent_id": <block_id> | null
        }
      ],
      "source": {"checksum": "<sha256:...>", "chars": <int>, "lang": "en"}
    }

    Rules you MUST follow:
    1. Exactly one headline block.
    2. Exactly one lead block, with reading_order ≤ 3.
    3. Tier A ≤ 10% of blocks; Tier B ≤ 25%.
    4. reading_order is a total order in [0, N).
    5. IDs are unique, lowercase, slug-form (e.g. "evidence-2").
    6. evidence and factbox blocks MUST have a valid source_span whose
       `quote` is the verbatim substring at [start:end].
    7. claim and decision blocks must have a source_span OR be is_inferred=true
       with inference_basis pointing to evidence/factbox block IDs.
    8. headline ≤ 100 chars; lead ≤ 60 words.
    9. Tier A total tokens ≤ B_A; Tier A+B total tokens ≤ B_AB.
    10. Newspaper budget: only the most load-bearing blocks earn Tier A.

    Output ONLY the JSON. No prose, no fences, no explanation.
""").strip()


FEW_SHOT_USER = dedent("""\
    Document type: {document_type}
    Render profile: {render_profile}
    Source text (between <<<>>>):

    <<<
    {source_text}
    >>>

    Examples of well-formed EOM (study the structure, then produce JSON for the source above):

    {few_shots}

    Now output the EOM JSON for the source above. Output JSON only.
""").strip()


def build_user_prompt(source_text: str, document_type: str, render_profile: str,
                      few_shots: str) -> str:
    return FEW_SHOT_USER.format(
        document_type=document_type,
        render_profile=render_profile,
        source_text=source_text,
        few_shots=few_shots,
    )


WITH_SPANS_USER = dedent("""\
    Document type: {document_type}
    Render profile: {render_profile}
    Source text (between <<<>>>):

    <<<
    {source_text}
    >>>

    REFERENCE SPANS — every block's `source_span` field MUST be drawn
    verbatim from this list. Do NOT invent offsets or quotes; pick a span
    by copying its `start`, `end`, and `quote` exactly.

    {spans_menu}

    Examples of well-formed EOM (study the structure, then produce JSON):

    {few_shots}

    Now output the EOM JSON for the source above. Output JSON only.
""").strip()


def build_user_prompt_with_spans(source_text: str, document_type: str,
                                  render_profile: str, few_shots: str,
                                  spans_menu: str) -> str:
    return WITH_SPANS_USER.format(
        document_type=document_type,
        render_profile=render_profile,
        source_text=source_text,
        few_shots=few_shots,
        spans_menu=spans_menu,
    )
