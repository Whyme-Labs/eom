"""EOM live demo — paste any markdown source, compile via Gemma-4 (OpenRouter),
visualise the newspaper render + harness pass status + context pack.

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

from eom.compilers.llm_client import OpenRouterClient
from eom.compilers.prompted import PromptedCompiler
from eom.compilers.rules import RulesCompiler
from eom.harness import validate
from eom.normalise import normalise
from eom.renderers import render_context_pack, render_newspaper
from eom.repair import compile_with_repair
from eom.schema import EOMDocument

# --- Streamlit config ---
st.set_page_config(
    page_title="EOM — Editorial Object Model",
    page_icon="📰",
    layout="wide",
)

# --- Sidebar: settings + sample loader ---
st.sidebar.title("EOM")
st.sidebar.markdown(
    "**Editorial Object Model** — turn any document into newspaper-style "
    "editorial JSON that LLMs can read efficiently."
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
SAMPLES_DIR = Path(__file__).parent.parent / "data" / "gold"
sample_options: dict[str, Path] = {}
if SAMPLES_DIR.exists():
    for type_dir in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        for md in sorted(type_dir.glob("*.md"))[:3]:
            sample_options[f"{type_dir.name}/{md.stem}"] = md
sample_choice = st.sidebar.selectbox("Sample", ["(none)"] + list(sample_options.keys()))

# --- Main: source input + compile ---
st.title("📰 EOM compiler")

if "source_text" not in st.session_state:
    st.session_state.source_text = ""

if sample_choice != "(none)":
    if st.sidebar.button("Load sample"):
        st.session_state.source_text = sample_options[sample_choice].read_text(encoding="utf-8")

source_text = st.text_area(
    "Source markdown",
    value=st.session_state.source_text,
    height=240,
    placeholder="# Title\n\nFirst paragraph...\n\n## Section\n\n...",
)

col_compile, col_status = st.columns([1, 5])
with col_compile:
    go = st.button("Compile EOM →", type="primary", use_container_width=True)

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
                    if not os.getenv("OPENROUTER_API_KEY"):
                        st.error(
                            "OPENROUTER_API_KEY not set. Switch to 'rules' compiler "
                            "or set the env var."
                        )
                        st.stop()
                    client = OpenRouterClient()
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

    col_status.success(
        f"✓ {len(eom.blocks)} blocks · {attempts} attempt(s) · "
        f"{elapsed:.1f}s · harness: "
        f"{'PASS' if report.passed else f'{len(report.failures)} fails'}"
    )

    st.divider()

    # --- Three-pane output ---
    tab_news, tab_pack, tab_json, tab_harness = st.tabs(
        ["📰 Newspaper", "🤖 Context pack", "📋 JSON", "✓ Harness"]
    )

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

else:
    st.info(
        "Paste a markdown document on the left and click **Compile EOM →**. "
        "Or pick a sample from the sidebar."
    )

# --- Footer ---
st.divider()
st.markdown(
    "*EOM v0.1 · MIT license · "
    "[GitHub](https://github.com/soh/eom) · Built with Gemma 4 via OpenRouter*"
)
