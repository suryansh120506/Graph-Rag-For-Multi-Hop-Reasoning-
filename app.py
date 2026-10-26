"""
app.py
======

Streamlit dashboard for the Multi-Hop Graph RAG Analyzer.

Layout
------
* Sidebar  - traversal horizon, example questions, source documents with their
             extracted triples, and a form to ingest new documents live.
* Top half - interactive pyvis knowledge graph (drag nodes, hover for
             provenance). The edges used by the latest answer are highlighted.
* Bottom   - chat interface. Every answer shows its retrieval mode; multi-hop
             answers print the exact traversal path *before* the synthesised
             insight is streamed.

Run:  streamlit run app.py
"""

from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path
from typing import Iterator

import streamlit as st
import streamlit.components.v1 as components

from extractor import SEED_DOCUMENTS, KnowledgeExtractor, bootstrap_knowledge_graph
from graph_engine import NODE_STYLE, EntityType, KnowledgeGraph
from multi_hop import DEFAULT_MAX_HOPS, GraphPath, GraphRAGPipeline, QueryMode, RAGResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

GRAPH_HTML = Path(__file__).with_name("graph.html")
DEFAULT_GRAPH_HEIGHT_PX = 660
EXAMPLE_QUESTIONS = [
    "Who supplies batteries to TechNova?",
    "How does the European strike impact the NovaPhone?",
    "What are the downstream impacts of the dockworker strike?",
    "Who competes with TechNova?",
    "How is OmegaCorp related to the NovaPhone?",
]

st.set_page_config(page_title="Multi-Hop Graph RAG Analyzer", page_icon="🕸️", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .path-row {display:flex; flex-wrap:wrap; align-items:center; gap:.35rem; margin:.25rem 0 .5rem;}
      .chip {padding:.22rem .6rem; border-radius:999px; color:#fff; font-weight:600; font-size:.86rem;}
      .rel {font-family:ui-monospace,Consolas,monospace; font-size:.78rem; opacity:.85;}
      .legend {display:flex; flex-wrap:wrap; gap:.9rem; font-size:.85rem; margin-bottom:.3rem;}
      .dot {display:inline-block; width:.75rem; height:.75rem; border-radius:50%; margin-right:.3rem; vertical-align:-1px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def init_state() -> None:
    if "kg" in st.session_state:
        return
    kg, extractor = bootstrap_knowledge_graph(SEED_DOCUMENTS)
    st.session_state.kg = kg
    st.session_state.extractor = extractor
    st.session_state.pipeline = GraphRAGPipeline(kg)
    st.session_state.messages = []          # list[dict]: {"role", "content", "response"?}
    st.session_state.highlight = set()      # edge keys to highlight in the graph


init_state()
kg: KnowledgeGraph = st.session_state.kg
extractor: KnowledgeExtractor = st.session_state.extractor
pipeline: GraphRAGPipeline = st.session_state.pipeline


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def chip(node: str) -> str:
    color = NODE_STYLE[kg.entity_type(node)]["color"] if node in kg else "#6B7280"
    return f'<span class="chip" style="background:{color}">{html.escape(node)}</span>'


def render_path(path: GraphPath) -> None:
    parts = [chip(path.start)]
    for hop in path.hops:
        rel = html.escape(hop.edge.relation.value)
        arrow = f"&larr;[{rel}]&mdash;" if hop.reversed else f"&mdash;[{rel}]&rarr;"
        parts += [f'<span class="rel">{arrow}</span>', chip(hop.to_node)]
    st.markdown(f'<div class="path-row">{"".join(parts)}</div>', unsafe_allow_html=True)
    st.code(path.render(), language=None)


def stream_text(text: str, delay: float = 0.006) -> Iterator[str]:
    for token in re.split(r"(\s+)", text):
        yield token
        time.sleep(delay)


def render_assistant(resp: RAGResponse, live: bool = False) -> None:
    linked = ", ".join(m.node_id for m in resp.mentions) or "none"
    st.caption(f"Mode: **{resp.mode.value}** · linked entities: {linked} · {resp.latency_ms:.1f} ms")

    if live:
        with st.status("Reasoning over the knowledge graph...", expanded=True) as status:
            for step in resp.trace:
                st.write(step)
                time.sleep(0.12)
            status.update(label="Graph retrieval complete", state="complete", expanded=False)

    if resp.paths:
        label = "Graph traversal path" if resp.mode is QueryMode.MULTI_HOP else "Structural path"
        st.markdown(f"**{label}** ({resp.paths[0].length} hops, score {resp.paths[0].score:.3f})")
        for p in resp.paths:
            render_path(p)
        with st.expander("Hop-by-hop evidence"):
            st.dataframe(resp.paths[0].hop_table(), hide_index=True)
    elif resp.mode is QueryMode.ONE_HOP and resp.context.facts:
        st.markdown("**Retrieved relation (1 hop)**")
        st.code("\n".join(f"{f.source} -[{f.relation.value}]-> {f.target}" for f in resp.context.facts), language=None)

    st.markdown("**Answer**")
    if live:
        st.write_stream(stream_text(resp.answer))
    else:
        st.markdown(resp.answer)

    with st.expander("Reasoning trace & grounded prompt"):
        for step in resp.trace:
            st.markdown(f"- {step}")
        st.code(resp.prompt, language=None)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🕸️ Graph RAG Analyzer")
    st.caption("NetworkX knowledge graph · pyvis visualisation · mock LLM (fully offline)")

    max_hops = st.slider("Traversal horizon (max hops)", 1, DEFAULT_MAX_HOPS, DEFAULT_MAX_HOPS)
    graph_height = st.slider("Graph height (px)", 420, 900, DEFAULT_GRAPH_HEIGHT_PX, step=20)

    st.subheader("Try a question")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.queued_question = q

    st.subheader("Source documents")
    for doc_id, text in extractor.documents.items():
        with st.expander(doc_id):
            st.write(text)
            st.dataframe(extractor.results[doc_id].triples(), hide_index=True)

    with st.form("ingest", clear_on_submit=True):
        st.markdown("**Ingest a new document**")
        new_text = st.text_area(
            "Text",
            placeholder="e.g. VoltCell supplies Graphite Anodes to ApexCorp.",
            label_visibility="collapsed",
        )
        if st.form_submit_button("Extract & add to graph") and new_text.strip():
            doc_id = extractor.next_doc_id()
            result = extractor.ingest(doc_id, new_text)
            if result.relations:
                st.success(f"{doc_id}: extracted {len(result.relations)} relations.")
            else:
                st.warning(
                    "The mock extractor found no supported patterns. Try phrasings like "
                    "'X supplies Y to Z', 'X manages shipping for Y', 'X competes with Y', "
                    "or 'A flood in Asia has severely disrupted X operations'."
                )

    c1, c2 = st.columns(2)
    if c1.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.highlight = set()
        st.rerun()
    if c2.button("Reset graph", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# --------------------------------------------------------------------------- #
# Answer the incoming question *before* drawing the graph, so the traversal
# path is highlighted in the same run.
# --------------------------------------------------------------------------- #
prompt = st.chat_input("Ask about suppliers, competitors, or how a risk propagates...")
question = prompt or st.session_state.pop("queued_question", None)
new_exchange: tuple[str, RAGResponse] | None = None
if question:
    response = pipeline.answer(question, max_hops=max_hops)
    st.session_state.highlight = response.highlight_edges
    new_exchange = (question, response)


# --------------------------------------------------------------------------- #
# Top half: knowledge graph
# --------------------------------------------------------------------------- #
st.title("Multi-Hop Graph RAG Analyzer")
st.caption(
    "Supply-chain knowledge graph built from unstructured documents. Drag nodes to rearrange them (your "
    "layout is remembered), hover nodes/edges for provenance, use the toolbar to zoom or fit. "
    "Red = edges used by the latest answer."
)

stats = kg.stats()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Entities", stats["nodes"])
m2.metric("Relationships", stats["edges"])
m3.metric("Documents", stats["documents"])
m4.metric("Graph density", stats["density"])

legend = "".join(
    f'<span><span class="dot" style="background:{NODE_STYLE[t]["color"]}"></span>{t.value}</span>'
    for t in EntityType
    if t is not EntityType.UNKNOWN
)
st.markdown(
    f'<div class="legend">{legend}<span style="opacity:.7">- - - dashed = contextual (no impact flow)</span></div>',
    unsafe_allow_html=True,
)

graph_file = kg.export_html(GRAPH_HTML, highlight_edges=st.session_state.highlight, height=f"{graph_height}px")
components.html(graph_file.read_text(encoding="utf-8"), height=graph_height, scrolling=False)

with st.expander("All extracted triples"):
    st.dataframe(kg.to_triples(), hide_index=True)


# --------------------------------------------------------------------------- #
# Bottom half: chat
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("💬 Ask the graph")

if not st.session_state.messages and not new_exchange:
    st.info(
        "Try a **1-hop** question like *Who supplies batteries to TechNova?* or a **multi-hop** question "
        "like *How does the European strike impact the NovaPhone?* - or pick one from the sidebar."
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg["response"])

if new_exchange:
    q, resp = new_exchange
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        render_assistant(resp, live=True)
    st.session_state.messages += [
        {"role": "user", "content": q},
        {"role": "assistant", "content": resp.answer, "response": resp},
    ]
