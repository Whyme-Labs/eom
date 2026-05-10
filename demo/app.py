"""EOM live demo — paste any markdown source, compile via Gemma-4 (OpenRouter),
visualise the newspaper render + harness pass status + context pack + ask-AI Q&A.

EOM is a two-way wire format between humans and models, with attention budgets
and source grounding built in. This demo shows both directions on the same IR.

Run locally:
    OPENROUTER_API_KEY=... uv run streamlit run demo/app.py

Deploy to Streamlit Cloud:
    set OPENROUTER_API_KEY in app secrets; everything else is in this repo.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import streamlit as st

from eom.compilers.llm_client import LLMRequest, OpenRouterClient
from eom.compilers.prompted import PromptedCompiler
from eom.compilers.rules import RulesCompiler
from eom.harness import validate
from eom.normalise import normalise
from eom.renderers import render_context_pack, render_newspaper
from eom.repair import compile_with_repair
from eom.schema import EOMDocument
from eom.tokens import count_tokens

REPO_ROOT = Path(__file__).resolve().parent.parent
QSETS_PATH = REPO_ROOT / "data" / "bench" / "qsets.json"
ANSWERER_MODEL = "google/gemma-4-31b-it"
ANSWER_SYSTEM = (
    "You answer questions about a document the user provides. "
    "Quote the relevant text where helpful. If the answer is not in the "
    "document, say 'not in document' and stop. Be concise."
)


# --- Streamlit config ---
st.set_page_config(
    page_title="EOM — two-way wire format",
    page_icon="📰",
    layout="wide",
)


# --- Hero ---
st.title("📰 EOM — two-way wire format between humans and models")
st.caption(
    "Attention budgets and source grounding built in. "
    "Same IR drives outbound (AI → human, newspaper brief) and inbound "
    "(human → AI, token-budgeted context pack)."
)


# --- Sidebar: settings + sample loader ---
st.sidebar.title("EOM")
st.sidebar.markdown(
    "**Editorial Object Model** v0.2 — formal IR for human/AI documents."
)
st.sidebar.divider()

compiler_kind = st.sidebar.radio(
    "Compiler",
    ("prompted (Gemma-4-31B)", "rules (deterministic)"),
    help="Prompted uses Gemma-4 via OpenRouter; rules uses regex heuristics offline.",
)
doc_type = st.sidebar.selectbox(
    "Document type", ["memo", "report", "paper", "transcript", "news", "policy", "other"]
)
render_profile = st.sidebar.selectbox("Render profile", ["executive_brief", "analytical_brief"])
max_attempts = st.sidebar.slider("Repair attempts", 1, 5, 3)

st.sidebar.divider()
st.sidebar.markdown("### Try a sample")
SAMPLES_DIR = REPO_ROOT / "data" / "gold"
sample_options: dict[str, Path] = {}
if SAMPLES_DIR.exists():
    for type_dir in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md"))[:3]:
            sample_options[f"{type_dir.name}/{md.stem}"] = md
sample_choice = st.sidebar.selectbox("Sample", ["(none)"] + list(sample_options.keys()))


# --- Question presets keyed by sample slug ---
@st.cache_data
def _load_qsets() -> dict[str, list[dict]]:
    if not QSETS_PATH.exists():
        return {}
    data = json.loads(QSETS_PATH.read_text())
    return {q["doc_id"]: q["questions"] for q in data.get("qsets", [])}


QSETS = _load_qsets()


def _qset_for_choice(choice: str) -> list[dict]:
    """Match a sidebar sample slug (e.g. 'policy/gdpr') to a qset doc_id."""
    if not choice or choice == "(none)":
        return []
    leaf = choice.split("/", 1)[-1]
    return QSETS.get(leaf, [])


# --- Source input ---
if "source_text" not in st.session_state:
    st.session_state.source_text = ""

if sample_choice != "(none)":
    if st.sidebar.button("Load sample"):
        st.session_state.source_text = sample_options[sample_choice].read_text(encoding="utf-8")

source_text = st.text_area(
    "Source markdown",
    value=st.session_state.source_text,
    height=200,
    placeholder="# Title\n\nFirst paragraph...\n\n## Section\n\n...",
)

col_compile, col_status = st.columns([1, 5])
with col_compile:
    go = st.button("Compile EOM →", type="primary", use_container_width=True)


def _ensure_openrouter() -> OpenRouterClient | None:
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("OPENROUTER_API_KEY not set.")
        return None
    return OpenRouterClient()


def _ask(client: OpenRouterClient, context: str, kind: str, question: str) -> tuple[str, float]:
    user = (
        f"### Document ({kind})\n\n{context}\n\n"
        f"### Question\n{question}\n\n"
        "Answer the question using only the document above."
    )
    t0 = time.time()
    out = client.complete(LLMRequest(
        system=ANSWER_SYSTEM, user=user, model=ANSWERER_MODEL, max_tokens=512,
    ))
    return out, time.time() - t0


# --- Compile path ---
if go:
    if not source_text.strip():
        st.error("Paste some markdown first.")
        st.stop()

    norm = normalise(source_text)
    t0 = time.time()
    with col_status:
        with st.spinner(f"Compiling via {compiler_kind}…"):
            try:
                if compiler_kind.startswith("prompted"):
                    client = _ensure_openrouter()
                    if client is None:
                        st.stop()
                    compiler = PromptedCompiler(client=client, few_shots=[])
                else:
                    compiler = RulesCompiler()
                eom, attempts = compile_with_repair(
                    compiler, norm,
                    hints={"document_type": doc_type, "render_profile": render_profile},
                    max_attempts=max_attempts,
                )
            except Exception as e:
                st.error(f"Compile failed: {e}")
                st.stop()
    elapsed = time.time() - t0
    report = validate(eom, norm)
    st.session_state.compiled_eom = eom
    st.session_state.compiled_norm = norm
    st.session_state.compiled_report = report
    st.session_state.compiled_choice = sample_choice
    col_status.success(
        f"✓ {len(eom.blocks)} blocks · {attempts} attempt(s) · "
        f"{elapsed:.1f}s · harness: "
        f"{'PASS' if report.passed else f'{len(report.failures)} fails'}"
    )


# --- Output (persisted across reruns from session_state) ---
eom = st.session_state.get("compiled_eom")
norm = st.session_state.get("compiled_norm")
report = st.session_state.get("compiled_report")
compiled_choice = st.session_state.get("compiled_choice", "(none)")

if eom is None:
    st.info(
        "Paste a markdown document and click **Compile EOM →**, "
        "or pick a sample from the sidebar. After compiling, the **Ask AI** tab "
        "shows the inbound (Human → AI) bandwidth thesis live."
    )
else:
    st.divider()
    tab_news, tab_pack, tab_json, tab_harness, tab_ask = st.tabs([
        "📰 Newspaper", "🤖 Context pack", "📋 JSON", "✓ Harness", "🔄 Ask AI",
    ])

    with tab_news:
        html = render_newspaper(eom)
        st.components.v1.html(html, height=900, scrolling=True)
        st.download_button("Download HTML", html, "eom.html", "text/html")

    with tab_pack:
        budget = st.slider("Token budget", 200, 5000, 1500, key="pack_budget")
        pack = render_context_pack(eom, token_budget=budget)
        st.code(pack, language="markdown")
        st.download_button("Download text", pack, "eom-pack.txt", "text/plain")

    with tab_json:
        eom_json = eom.model_dump_json(indent=2)
        st.code(eom_json, language="json")
        st.download_button("Download JSON", eom_json, "eom.json", "application/json")

    with tab_harness:
        if report.passed:
            st.success("All H1–H12 checks pass.")
        else:
            st.error(f"{len(report.failures)} failures:")
            for f in report.failures:
                tag = f"[{f.block_id}] " if f.block_id else ""
                st.markdown(f"- **{f.rule}** {tag}{f.message}")
        st.markdown("### Metrics")
        st.json(report.metrics)
        st.markdown("### Warnings (corpus-level, not checked here)")
        for w in report.warnings:
            st.markdown(f"- **{w.rule}**: {w.message}")

    with tab_ask:
        st.markdown(
            "**Inbound demo (Human → AI).** Send the same question to the same "
            "model with two different contexts: the raw markdown and the EOM "
            "context-pack. Compare input cost, latency, and answer quality."
        )

        presets = _qset_for_choice(compiled_choice)
        question_options = ["(custom)"] + [p["q"] for p in presets]
        chosen_q_idx = st.selectbox(
            "Question", range(len(question_options)),
            format_func=lambda i: question_options[i],
        )
        if chosen_q_idx == 0:
            question = st.text_input("Custom question", "")
        else:
            question = question_options[chosen_q_idx]
            preset = presets[chosen_q_idx - 1]
            st.caption(f"Reference answer: {preset['ref']}")

        ask_budget = st.slider(
            "Pack budget (tokens)", 200, 4000, 1500, key="ask_budget",
            help="Token budget used to lower the EOM IR into a context-pack."
        )
        run_ask = st.button("Run side-by-side →", type="primary")

        if run_ask:
            if not question.strip():
                st.error("Pick a preset or enter a custom question.")
            else:
                client = _ensure_openrouter()
                if client is None:
                    st.stop()
                pack_text = render_context_pack(eom, token_budget=ask_budget)
                raw_text = norm  # normalised source string the EOM was compiled from
                raw_in_tok = count_tokens(raw_text)
                pack_in_tok = count_tokens(pack_text)

                col_raw, col_pack = st.columns(2)
                with st.spinner(f"Asking {ANSWERER_MODEL} twice…"):
                    raw_ans, raw_dt = _ask(client, raw_text, "raw markdown", question)
                    pack_ans, pack_dt = _ask(client, pack_text, "EOM context-pack", question)

                with col_raw:
                    st.markdown("#### Raw markdown")
                    st.metric("Input tokens", f"{raw_in_tok:,}")
                    st.metric("Latency", f"{raw_dt:.1f}s")
                    st.markdown("**Answer:**")
                    st.write(raw_ans)
                    with st.expander("Context sent (truncated)"):
                        preview = raw_text[:1200] + ("…" if len(raw_text) > 1200 else "")
                        st.code(preview, language="markdown")

                with col_pack:
                    st.markdown("#### EOM context-pack")
                    delta = (pack_in_tok - raw_in_tok) / max(raw_in_tok, 1) * 100
                    st.metric("Input tokens", f"{pack_in_tok:,}", f"{delta:+.0f}% vs raw")
                    st.metric("Latency", f"{pack_dt:.1f}s")
                    st.markdown("**Answer:**")
                    st.write(pack_ans)
                    with st.expander("Context sent"):
                        st.code(pack_text, language="markdown")

                st.divider()
                compression = pack_in_tok / max(raw_in_tok, 1)
                st.markdown(
                    f"**Compression**: pack uses **{compression:.2f}×** the tokens "
                    f"of raw ({pack_in_tok:,} vs {raw_in_tok:,}, "
                    f"{(1 - compression) * 100:.0f}% reduction). "
                    "Same model, same prompt skeleton, different context."
                )
                if presets and chosen_q_idx > 0:
                    st.caption(
                        "Reference: " + presets[chosen_q_idx - 1]["ref"]
                    )


# --- Footer ---
st.divider()
st.markdown(
    "*EOM v0.2 (draft) · MIT license · "
    "[GitHub](https://github.com/Whyme-Labs/eom) · spec at `docs/SPEC-v0.2.md`*"
)
