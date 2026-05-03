"""
pipeline_monitor.py
────────────────────
Live pipeline monitor — reacts automatically to queries typed in chat_ui.py.
Reads query_bridge.json every second and updates the display.

Run with: py -m streamlit run pipeline_monitor.py --server.port 8502
Open in browser: http://localhost:8502

Keep chat_ui.py open on Monitor 1, this on Monitor 2.
"""

import json
import time
import os
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import chromadb

BRIDGE_FILE = "./query_bridge.json"
POLL_SECS   = 1   # how often to check bridge file

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pipeline Monitor",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}
.stage-idle {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    opacity: 0.45;
}
.stage-active {
    background: #0d2137;
    border: 2px solid #1f6feb;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: 0 0 18px rgba(31,111,235,0.35);
    animation: pulse 1.2s infinite;
}
.stage-done {
    background: #0d2818;
    border: 2px solid #238636;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 8px  rgba(31,111,235,0.3); }
    50%  { box-shadow: 0 0 22px rgba(31,111,235,0.7); }
    100% { box-shadow: 0 0 8px  rgba(31,111,235,0.3); }
}
.stage-title { font-weight: 700; font-size: 0.95em; }
.stage-body  { font-size: 0.82em; color: #8b949e; margin-top: 3px; }
.stage-value { font-size: 0.78em; color: #58a6ff; font-family: monospace;
               margin-top: 4px; word-break: break-all; }
.chunk-card {
    background: #161b22;
    border: 1px solid #238636;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82em;
}
.score-badge { color: #3fb950; font-weight: bold; }
.answer-box {
    background: #0d2137;
    border-left: 4px solid #1f6feb;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.92em;
    line-height: 1.65;
    color: #e6edf3;
}
.waiting-badge {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    color: #8b949e;
    font-size: 0.9em;
}
.query-banner {
    background: #0d2137;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 1em;
    color: #58a6ff;
    margin-bottom: 12px;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Load vector data (for charts) ─────────────────────────────────────────────
@st.cache_resource
def load_vectors():
    chroma   = chromadb.PersistentClient(path="./chroma_db")
    col      = chroma.get_or_create_collection("course_docs")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    r        = col.get(include=["documents", "embeddings"])
    docs     = r["documents"]
    vecs     = np.array(r["embeddings"])
    return docs, vecs, embedder

all_docs, all_vecs, embedder = load_vectors()
n_chunks = len(all_docs)


# ── Bridge reader ─────────────────────────────────────────────────────────────
def read_bridge():
    if not os.path.exists(BRIDGE_FILE):
        return {"status": "idle", "query": "", "timestamp": 0,
                "chunks": [], "scores": [], "answer": "",
                "tokens_in": 0, "tokens_out": 0,
                "embed_ms": 0, "search_ms": 0, "llm_ms": 0}
    with open(BRIDGE_FILE, "r") as f:
        return json.load(f)


# ── Stage renderer ────────────────────────────────────────────────────────────
def stage(icon, title, body, value="", state="idle"):
    css = f"stage-{state}"
    val = f'<div class="stage-value">{value}</div>' if value else ""
    return f"""
    <div class="{css}">
      <div class="stage-title">{icon} {title}</div>
      <div class="stage-body">{body}</div>
      {val}
    </div>
    """

STATUS_ORDER = ["idle", "embedding", "searching",
                "retrieving", "generating", "done"]

def stage_state(current_status, this_stage):
    """Returns idle / active / done for a given stage given current pipeline status."""
    order = STATUS_ORDER
    if current_status == "idle":
        return "idle"
    ci = order.index(current_status) if current_status in order else 0
    si = order.index(this_stage)     if this_stage    in order else 0
    if ci == si:
        return "active"
    elif ci > si:
        return "done"
    else:
        return "idle"


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Pipeline Monitor")
st.caption("Auto-updates when you type a query in chat_ui.py · refreshes every second")

# ── Initialise session state ──────────────────────────────────────────────────
if "last_ts" not in st.session_state:
    st.session_state.last_ts   = 0
if "last_status" not in st.session_state:
    st.session_state.last_status = "idle"

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("### Live Pipeline Stages")
    query_banner  = st.empty()
    stages_ph     = st.empty()
    stats_ph      = st.empty()

with right:
    tab_vec, tab_search, tab_chunks, tab_answer = st.tabs([
        "📡 Query Vector",
        "📊 Similarity Scores",
        "📄 Retrieved Chunks",
        "💬 Answer"
    ])
    with tab_vec:
        vec_ph  = st.empty()
        pca_ph  = st.empty()
    with tab_search:
        bar_ph  = st.empty()
    with tab_chunks:
        chunk_ph = st.empty()
    with tab_answer:
        ans_ph   = st.empty()
        meta_ph  = st.empty()


# ── Render idle state ─────────────────────────────────────────────────────────
def render_idle():
    query_banner.markdown(
        '<div class="waiting-badge">⏳ Waiting for a query from chat_ui.py…<br>'
        '<small>Type a question in the chat window on your other monitor</small></div>',
        unsafe_allow_html=True
    )
    html = (
        stage("1️⃣", "Query Received",      "Waiting for input…",                    state="idle") +
        stage("2️⃣", "Embedding Query",     "Convert text → 384-dim vector",         state="idle") +
        stage("3️⃣", "Vector Search",       "Cosine similarity vs all chunks",       state="idle") +
        stage("4️⃣", "Retrieve Top Chunks", "Pull matching text from ChromaDB",      state="idle") +
        stage("5️⃣", "Build Prompt",        "Assemble system + context + query",     state="idle") +
        stage("6️⃣", "Claude Generates",    "LLM reads context and responds",        state="idle")
    )
    stages_ph.markdown(html, unsafe_allow_html=True)
    vec_ph.info("Waiting for query…")
    pca_ph.empty()
    bar_ph.info("Waiting for query…")
    chunk_ph.info("Waiting for query…")
    ans_ph.info("Waiting for query…")
    meta_ph.empty()
    stats_ph.empty()


# ── Render active / done state ────────────────────────────────────────────────
def render_state(data):
    status     = data["status"]
    query      = data["query"]
    chunks     = data.get("chunks", [])
    scores     = data.get("scores", [])
    answer     = data.get("answer", "")
    embed_ms   = data.get("embed_ms", 0)
    search_ms  = data.get("search_ms", 0)
    llm_ms     = data.get("llm_ms", 0)
    tokens_in  = data.get("tokens_in", 0)
    tokens_out = data.get("tokens_out", 0)

    # Query banner
    query_banner.markdown(
        f'<div class="query-banner">❝ {query} ❞</div>',
        unsafe_allow_html=True
    )

    # Pipeline stages
    html = (
        stage("1️⃣", "Query Received",
              "Text captured from chat_ui.py",
              value=f'"{query[:60]}…"' if len(query) > 60 else f'"{query}"',
              state=stage_state(status, "embedding") if status != "idle" else "done") +

        stage("2️⃣", "Embedding Query",
              f"all-MiniLM-L6-v2 → 384 floats · {embed_ms} ms" if embed_ms else "Running…",
              state=stage_state(status, "embedding")) +

        stage("3️⃣", "Vector Search",
              f"Cosine sim vs {n_chunks} chunks · {search_ms} ms" if search_ms else "Waiting…",
              state=stage_state(status, "searching")) +

        stage("4️⃣", "Retrieve Top Chunks",
              f"{len(chunks)} chunks retrieved" if chunks else "Waiting…",
              value=f"Scores: {', '.join(str(round(s,4)) for s in scores)}" if scores else "",
              state=stage_state(status, "retrieving")) +

        stage("5️⃣", "Build Prompt",
              "System + context + user query assembled" if status in ["generating","done"] else "Waiting…",
              state=stage_state(status, "generating")) +

        stage("6️⃣", "Claude Generates",
              f"Response in {llm_ms} ms · {tokens_out} tokens" if llm_ms else "Waiting…",
              state=stage_state(status, "done"))
    )
    stages_ph.markdown(html, unsafe_allow_html=True)

    # Stats (bottom of left panel)
    if status == "done":
        cost = (tokens_in * 0.25 + tokens_out * 1.25) / 1_000_000
        total_ms = embed_ms + search_ms + llm_ms
        stats_ph.markdown(f"""
| Metric | Value |
|---|---|
| Total time | `{total_ms} ms` |
| Embed | `{embed_ms} ms` |
| Search | `{search_ms} ms` |
| LLM | `{llm_ms} ms` |
| Tokens in | `{tokens_in}` |
| Tokens out | `{tokens_out}` |
| Cost | `${cost:.6f}` |
""")

    # ── Vector tab ────────────────────────────────────────────────────────────
    if status not in ["idle"] and query:
        q_vec = embedder.encode([query])[0]

        fig_vec = px.bar(
            x=[f"D{i}" for i in range(60)],
            y=q_vec[:60],
            color=q_vec[:60],
            color_continuous_scale="RdBu",
            color_continuous_midpoint=0,
            labels={"x": "Dimension", "y": "Value"},
            title="Query vector — first 60 of 384 dimensions"
        )
        fig_vec.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=260, showlegend=False,
            xaxis=dict(tickangle=45, tickfont=dict(size=7), gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d")
        )
        vec_ph.plotly_chart(fig_vec, use_container_width=True)

        # PCA map
        q_vec_np = q_vec.reshape(1, -1)
        sims_all = cosine_similarity(q_vec_np, all_vecs)[0]
        top_idx  = np.argsort(sims_all)[::-1][:3]

        combined = np.vstack([all_vecs, q_vec_np])
        coords   = PCA(n_components=2).fit_transform(combined)
        cxy, qxy = coords[:-1], coords[-1]

        colours = ["#f0883e" if i in top_idx else "#1f6feb" for i in range(n_chunks)]

        fig_pca = go.Figure()
        fig_pca.add_scatter(
            x=cxy[:,0], y=cxy[:,1], mode="markers+text",
            text=[f"C{i+1}" for i in range(n_chunks)],
            textposition="top center",
            marker=dict(size=10, color=colours, line=dict(width=1, color="#fff")),
            hovertext=[d[:80]+"…" for d in all_docs],
            hovertemplate="%{hovertext}<extra></extra>",
            name="Chunks"
        )
        for idx in top_idx:
            fig_pca.add_shape(type="line",
                x0=qxy[0], y0=qxy[1], x1=cxy[idx,0], y1=cxy[idx,1],
                line=dict(color="#f0883e", width=1.5, dash="dot")
            )
        fig_pca.add_scatter(
            x=[qxy[0]], y=[qxy[1]], mode="markers+text",
            text=["QUERY"], textposition="bottom center",
            marker=dict(size=20, color="#ff4f4f", symbol="star",
                        line=dict(width=2, color="#fff")),
            name="Query"
        )
        fig_pca.update_layout(
            title="Query in vector space — orange = retrieved chunks",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=360,
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            legend=dict(bgcolor="#21262d")
        )
        pca_ph.plotly_chart(fig_pca, use_container_width=True)

    # ── Search tab ────────────────────────────────────────────────────────────
    if status in ["searching", "retrieving", "generating", "done"] and query:
        q_vec = embedder.encode([query])[0]
        sims  = cosine_similarity(q_vec.reshape(1,-1), all_vecs)[0]
        top3  = set(np.argsort(sims)[::-1][:3])

        sorted_data = sorted(
            [{"Chunk": f"C{i+1}", "Sim": round(float(sims[i]),4),
              "Preview": all_docs[i][:55]+"…", "top": i in top3}
             for i in range(n_chunks)],
            key=lambda x: x["Sim"], reverse=True
        )
        colours_bar = ["#f0883e" if d["top"] else "#1f6feb" for d in sorted_data]
        threshold   = sorted_data[2]["Sim"] if len(sorted_data) >= 3 else 0

        fig_bar = go.Figure(go.Bar(
            x=[d["Chunk"] for d in sorted_data],
            y=[d["Sim"]   for d in sorted_data],
            marker_color=colours_bar,
            hovertext=[d["Preview"] for d in sorted_data],
            hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<br>%{hovertext}<extra></extra>"
        ))
        fig_bar.add_hline(
            y=threshold, line_dash="dash", line_color="#3fb950",
            annotation_text=f"retrieval cutoff ({threshold:.4f})",
            annotation_font_color="#3fb950"
        )
        fig_bar.update_layout(
            title="Cosine similarity scores — orange = retrieved",
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font_color="#e6edf3", height=380,
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d", title="Cosine Similarity", range=[0,1])
        )
        bar_ph.plotly_chart(fig_bar, use_container_width=True)

    # ── Chunks tab ────────────────────────────────────────────────────────────
    if chunks:
        html_chunks = ""
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            html_chunks += f"""
            <div class="chunk-card">
              <span class="score-badge">#{i+1} · score: {score:.4f}</span><br>
              <span style="color:#8b949e">{chunk}</span>
            </div>
            """
        chunk_ph.markdown(html_chunks, unsafe_allow_html=True)

    # ── Answer tab ────────────────────────────────────────────────────────────
    if answer:
        ans_ph.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True
        )
        cost = (tokens_in * 0.25 + tokens_out * 1.25) / 1_000_000
        meta_ph.markdown(f"""
| | |
|---|---|
| Model | `claude-haiku-4-5-20251001` |
| Response time | `{llm_ms} ms` |
| Input tokens | `{tokens_in}` |
| Output tokens | `{tokens_out}` |
| Cost | `${cost:.6f} USD` |
""")


# ── Main polling loop ─────────────────────────────────────────────────────────
render_idle()

# Auto-refresh placeholder
refresh_ph = st.empty()

# Poll the bridge file
data = read_bridge()

if data["status"] == "idle" or data["timestamp"] == st.session_state.last_ts:
    render_idle()
else:
    st.session_state.last_ts     = data["timestamp"]
    st.session_state.last_status = data["status"]
    render_state(data)

# Auto rerun every POLL_SECS seconds
time.sleep(POLL_SECS)
st.rerun()
