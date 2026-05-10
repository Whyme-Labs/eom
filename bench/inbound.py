"""Inbound benchmark: raw text vs EOM context-pack as LLM input.

For each (doc, question, mode), feed the chosen context + question to an LLM
and record input tokens, output, latency, and citation marker count. Then run
LLM-as-judge to score answer correctness against the human-authored reference.

Outputs:
    data/bench/results/<run-id>.json   (per-question rows)
    data/bench/results/<run-id>.md     (summary table for the writeup)

Run with:
    uv run python -m bench.inbound

Env required:
    OPENROUTER_API_KEY  (already in .env)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from eom.compilers.llm_client import LLMRequest, OpenRouterClient
from eom.renderers.context_pack import render_context_pack
from eom.schema import EOMDocument
from eom.tokens import count_tokens

ROOT = Path(__file__).resolve().parent.parent
QSETS = ROOT / "data" / "bench" / "qsets.json"
RESULTS = ROOT / "data" / "bench" / "results"

ANSWERER_MODEL = "google/gemma-4-31b-it"     # downstream consumer
JUDGE_MODEL = "anthropic/claude-sonnet-4.5"  # reliable judge
TOKEN_BUDGET = 1500                           # context-pack budget per doc

# Citation marker emitted by render_context_pack: " [src:<block-id>]"
CITATION_RE = re.compile(r"\[src:([a-z][a-z0-9-]*)\]")


@dataclass
class Row:
    run_id: str
    doc_id: str
    question_id: str
    question: str
    mode: str                         # "raw" | "pack"
    model: str
    input_tokens: int
    output_text: str = ""
    output_tokens: int = 0
    citations: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    judge_score: int | None = None    # 0 wrong, 1 partial, 2 correct
    judge_rationale: str = ""
    error: str | None = None


_ANSWER_SYSTEM = (
    "You answer questions about a document the user provides. "
    "Quote the relevant text where helpful. If the answer is not in the "
    "document, say 'not in document' and stop. Be concise."
)

_JUDGE_SYSTEM = (
    "You are a strict grader. Given a question, a reference answer, and a "
    "candidate answer, return JSON: "
    '{"score": 0|1|2, "rationale": "..."} where 2=fully correct, '
    "1=partially correct (missing or imprecise), 0=wrong or 'not in document'. "
    "Output ONLY the JSON object, no preamble."
)


def _answer_prompt(context_kind: str, context: str, question: str) -> str:
    return (
        f"### Document ({context_kind})\n\n"
        f"{context}\n\n"
        f"### Question\n{question}\n\n"
        "Answer the question using only the document above."
    )


def _judge_prompt(question: str, reference: str, candidate: str) -> str:
    return (
        f"### Question\n{question}\n\n"
        f"### Reference answer\n{reference}\n\n"
        f"### Candidate answer\n{candidate}\n\n"
        "Grade the candidate."
    )


def _load_qsets() -> dict:
    return json.loads(QSETS.read_text())


def _load_doc(doc_path: str) -> tuple[str, EOMDocument]:
    base = ROOT / doc_path
    raw = (base.with_suffix(".md")).read_text()
    eom = EOMDocument.model_validate_json(
        (base.with_suffix(".eom.json")).read_text()
    )
    return raw, eom


def _call_llm(client: OpenRouterClient, system: str, user: str, model: str,
              max_tokens: int = 1024) -> tuple[str, float]:
    t0 = time.time()
    out = client.complete(LLMRequest(
        system=system, user=user, model=model, max_tokens=max_tokens,
    ))
    return out, time.time() - t0


def _judge(client: OpenRouterClient, question: str, reference: str,
           candidate: str) -> tuple[int | None, str]:
    user = _judge_prompt(question, reference, candidate)
    try:
        raw, _ = _call_llm(client, _JUDGE_SYSTEM, user, JUDGE_MODEL, max_tokens=300)
    except Exception as e:
        return None, f"judge_error: {e}"
    # Tolerate ```json fences.
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
        score = int(obj.get("score"))
        if score not in (0, 1, 2):
            raise ValueError(f"score out of range: {score}")
        return score, str(obj.get("rationale", ""))[:500]
    except Exception as e:
        return None, f"judge_parse_error: {e!r} on {raw[:120]!r}"


def run(qset_doc_ids: list[str] | None, model: str, judge: bool,
        run_id: str | None = None) -> Path:
    client = OpenRouterClient()
    data = _load_qsets()
    rid = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rows: list[Row] = []

    for qset in data["qsets"]:
        if qset_doc_ids and qset["doc_id"] not in qset_doc_ids:
            continue
        raw_text, eom = _load_doc(qset["doc_path"])
        pack_text = render_context_pack(eom, token_budget=TOKEN_BUDGET)
        contexts = {
            "raw":  ("raw markdown",     raw_text),
            "pack": ("EOM context-pack", pack_text),
        }
        for q in qset["questions"]:
            for mode, (kind, ctx) in contexts.items():
                user_prompt = _answer_prompt(kind, ctx, q["q"])
                in_tokens = count_tokens(user_prompt)
                row = Row(
                    run_id=rid, doc_id=qset["doc_id"], question_id=q["id"],
                    question=q["q"], mode=mode, model=model,
                    input_tokens=in_tokens,
                )
                try:
                    out, dt = _call_llm(client, _ANSWER_SYSTEM, user_prompt, model)
                    row.output_text = out
                    row.output_tokens = count_tokens(out)
                    row.latency_s = round(dt, 2)
                    row.citations = CITATION_RE.findall(out)
                except Exception as e:
                    row.error = repr(e)
                if judge and row.error is None:
                    score, rat = _judge(client, q["q"], q["ref"], row.output_text)
                    row.judge_score = score
                    row.judge_rationale = rat
                rows.append(row)
                print(f"  [{qset['doc_id']:14}] {q['id']:7} {mode:4} "
                      f"in={in_tokens:5} out={row.output_tokens:4} "
                      f"cite={len(row.citations)} score={row.judge_score} "
                      f"{'OK' if not row.error else 'ERR'}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"{rid}.json"
    json_path.write_text(json.dumps(
        [r.__dict__ for r in rows], indent=2, sort_keys=True,
    ))
    md_path = RESULTS / f"{rid}.md"
    md_path.write_text(_summary_md(rows, rid, model, judge))
    print(f"\nwrote {json_path}\nwrote {md_path}")
    return md_path


def _summary_md(rows: list[Row], run_id: str, model: str, judge: bool) -> str:
    by_mode_doc: dict[tuple[str, str], list[Row]] = {}
    for r in rows:
        by_mode_doc.setdefault((r.doc_id, r.mode), []).append(r)

    lines = [
        f"# Inbound benchmark — {run_id}",
        "",
        f"answerer: `{model}`  |  judge: `{JUDGE_MODEL if judge else 'off'}`  "
        f"|  budget: {TOKEN_BUDGET} tokens",
        "",
        "## Per-doc summary",
        "",
        "| Doc | Mode | Q | Σ input tok | mean output tok | mean citations | mean score |",
        "|---|---|---|---|---|---|---|",
    ]
    docs = sorted({r.doc_id for r in rows})
    for doc in docs:
        for mode in ("raw", "pack"):
            rs = by_mode_doc.get((doc, mode), [])
            if not rs:
                continue
            n = len(rs)
            sum_in = sum(r.input_tokens for r in rs)
            mean_out = sum(r.output_tokens for r in rs) / n
            mean_cites = sum(len(r.citations) for r in rs) / n
            scored = [r.judge_score for r in rs if r.judge_score is not None]
            mean_score = (sum(scored) / len(scored)) if scored else None
            score_str = f"{mean_score:.2f}" if mean_score is not None else "—"
            lines.append(
                f"| {doc} | {mode} | {n} | {sum_in} | {mean_out:.0f} | "
                f"{mean_cites:.1f} | {score_str} |"
            )
    lines.append("")
    lines.append("## Headline numbers")
    raw_in = sum(r.input_tokens for r in rows if r.mode == "raw")
    pack_in = sum(r.input_tokens for r in rows if r.mode == "pack")
    raw_scored = [r.judge_score for r in rows if r.mode == "raw" and r.judge_score is not None]
    pack_scored = [r.judge_score for r in rows if r.mode == "pack" and r.judge_score is not None]
    raw_mean = (sum(raw_scored) / len(raw_scored)) if raw_scored else None
    pack_mean = (sum(pack_scored) / len(pack_scored)) if pack_scored else None
    pack_cites = sum(len(r.citations) for r in rows if r.mode == "pack")
    raw_cites = sum(len(r.citations) for r in rows if r.mode == "raw")
    lines += [
        "",
        f"- Total input tokens: raw={raw_in:,} | pack={pack_in:,} | "
        f"compression={(pack_in/raw_in if raw_in else 0):.2f}x",
        f"- Citations resolved: raw={raw_cites} | pack={pack_cites}",
    ]
    if raw_mean is not None and pack_mean is not None:
        lines.append(
            f"- Mean judge score (0/1/2): raw={raw_mean:.2f} | pack={pack_mean:.2f}"
        )
    return "\n".join(lines) + "\n"


def _load_dotenv() -> None:
    """Trivial .env loader so `OPENROUTER_API_KEY` resolves."""
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="*", default=None,
                    help="doc_ids to run (default: all)")
    ap.add_argument("--model", default=ANSWERER_MODEL)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM-as-judge phase (cheaper)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    run(qset_doc_ids=args.docs, model=args.model, judge=not args.no_judge,
        run_id=args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
