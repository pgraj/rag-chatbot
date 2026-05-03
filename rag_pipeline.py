"""
rag_pipeline.py
───────────────
Live RAG pipeline visualiser — see every step as it happens:
  Query → Embed → Vector Search → Retrieve Chunks → Build Prompt → Claude Response

Run with: py -m streamlit run rag_pipeline.py
Requires:  py -m pip install streamlit anthropic chromadb sentence-transformers plotly python-dotenv
"""

import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline Visualiser",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
/* Dark base */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}

/* Stage cards */
.stage-idle {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    opacity: 0.5;
}
.stage-active {
    background: #0d2137;
    border: 2px solid #1f6feb;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 0 16px rgba(31,111,235,0.3);
}
.stage-done {
    background: #0d2818;
    border: 2px solid #238636;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.stage-title { font-weight: 700; font-size: 1em; margin-bottom: 4px; }
.stage-body  { font-size: 0.85em; color: #8b949e; }
.stage-value { font-size: 0.82em; color: #58a6ff; font-family: monospace; word-break: break-all; }

/* Prompt box */
.prompt-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    font-size: 0.8em;
    font-family: monospace;
    white-space: pre-wrap;
    color: #e6edf3;
    max-height: 320px;
    overflow-y: auto;
}

/* Response box */
.response-box {
    background: #0d2137;
    border-left: 4px solid #1f6feb;
    border-radius: 6px;
    padding: 16px;
    font-size: 0.92em;
    line-height: 1.6;
    color: #e6edf3;
}

/* Chunk card */
.chunk-card {
    background: #161b22;
    border: 1px solid #238636;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.82em;
    color: #e6edf3;
}
.chunk-score { color: #3fb950; font-weight: bold; font-size: 0.9em; }

/* Token counter */
.token-box {
    background: #21262d;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: monospace;
    font-size: 0.85em;
    color: #f0883e;
}

/* Pipeline arrow */
.arrow {
    text-align: center;
    font-size: 1.4em;
    color: #30363d;
    margin: 2px 0;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    chroma  = chromadb.PersistentClient(path="./chroma_db")
    col     = chroma.get_or_create_collection("course_docs")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client  = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return col, embedder, client

@st.cache_data
def load_all_chunks():
    col, _, _ = load_resources()
    r = col.get(include=["documents", "embeddings"])
    return r["documents"], np.array(r["embeddings"])

collection, embedder, anthropic_client = load_resources()
all_docs, all_vecs = load_all_chunks()
n_chunks = len(all_docs)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 RAG Pipeline Visualiser")
st.caption("Watch every step of Retrieval-Augmented Generation happen in real time")

st.markdown(f"""
> **Your knowledge base:** `{n_chunks}` chunks indexed · `{all_vecs.shape[1]}` dimensions per vector · Model: `all-MiniLM-L6-v2` · LLM: `claude-haiku-4-5-20251001`
""")

# ── Layout: left pipeline | right detail ─────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

# ── Left: query input + pipeline stages ──────────────────────────────────────
with left:
    st.subheader("🎯 Ask a Question")

    sample_questions = [
        "Select a sample question…",
        "What are the entry requirements for the Diploma of Nursing?",
        "Which courses have government funding available?",
        "How long does Certificate III in Individual Support take?",
        "Is RPL available for the Diploma of Project Management?",
        "What is the tuition fee for the ICT Diploma?",
        "How do I enrol at Pinnacle Training Group?",
    ]
    preset = st.selectbox("Try a sample:", sample_questions)
    query  = st.text_input(
        "Or type your own:",
        value="" if preset == sample_questions[0] else preset
    )

    n_results = st.slider("Chunks to retrieve (top-k)", 1, 5, 3)
    run_btn   = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    st.divider()

    # Pipeline stage placeholders
    st.markdown("### Pipeline Stages")

    stage_query_ph   = st.empty()
    arrow1           = st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
    stage_embed_ph   = st.empty()
    arrow2           = st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
    stage_search_ph  = st.empty()
    arrow3           = st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
    stage_retrieve_ph= st.empty()
    arrow4           = st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
    stage_prompt_ph  = st.empty()
    arrow5           = st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)
    stage_llm_ph     = st.empty()

    def render_stage(ph, icon, title, body, value="", state="idle"):
        css = f"stage-{state}"
        val_html = f'<div class="stage-value">{value}</div>' if value else ""
        ph.markdown(f"""
        <div class="{css}">
          <div class="stage-title">{icon} {title}</div>
          <div class="stage-body">{body}</div>
          {val_html}
        </div>
        """, unsafe_allow_html=True)

    # Draw all stages idle on load
    render_stage(stage_query_ph,    "1️⃣", "Query Received",       "Waiting for input…",                state="idle")
    render_stage(stage_embed_ph,    "2️⃣", "Embedding Query",      "Convert text → 384-dim vector",      state="idle")
    render_stage(stage_search_ph,   "3️⃣", "Vector Search",        "Cosine similarity vs all chunks",    state="idle")
    render_stage(stage_retrieve_ph, "4️⃣", "Retrieve Top Chunks",  "Pull matching text from ChromaDB",   state="idle")
    render_stage(stage_prompt_ph,   "5️⃣", "Build Prompt",         "Assemble system + context + query",  state="idle")
    render_stage(stage_llm_ph,      "6️⃣", "Claude Generates",     "LLM reads context and responds",     state="idle")


# ── Right: detail panels ──────────────────────────────────────────────────────
with right:
    detail_header = st.empty()
    detail_header.subheader("Pipeline Detail")

    tab_vec, tab_search, tab_chunks, tab_prompt, tab_response = st.tabs([
        "📡 Query Vector",
        "📊 Similarity Scores",
        "📄 Retrieved Chunks",
        "📝 Full Prompt",
        "💬 Claude Response"
    ])

    with tab_vec:
        vec_intro   = st.empty()
        vec_chart   = st.empty()
        vec_stats   = st.empty()
        vec_pca     = st.empty()

    with tab_search:
        search_intro = st.empty()
        search_bar   = st.empty()
        search_heat  = st.empty()

    with tab_chunks:
        chunk_intro  = st.empty()
        chunk_cards  = st.empty()

    with tab_prompt:
        prompt_intro = st.empty()
        prompt_box   = st.empty()
        token_box    = st.empty()

    with tab_response:
        resp_intro   = st.empty()
        resp_box     = st.empty()
        resp_meta    = st.empty()

    # Default messages in each tab
    vec_intro.info("Run the pipeline to see your query as a vector.")
    search_intro.info("Run the pipeline to see similarity scores.")
    chunk_intro.info("Run the pipeline to see retrieved chunks.")
    prompt_intro.info("Run the pipeline to see the full prompt sent to Claude.")
    resp_intro.info("Run the pipeline to see Claude's response.")


# ── Pipeline execution ────────────────────────────────────────────────────────
if run_btn and query.strip():

    # ── STAGE 1: Query Received ───────────────────────────────────────────────
    render_stage(stage_query_ph, "1️⃣", "Query Received",
                 "Text captured and ready for embedding.",
                 value=f'"{query}"', state="done")
    time.sleep(0.3)

    # ── STAGE 2: Embed ────────────────────────────────────────────────────────
    render_stage(stage_embed_ph, "2️⃣", "Embedding Query",
                 "Running all-MiniLM-L6-v2…", state="active")

    t0 = time.time()
    q_vec = embedder.encode([query])[0]
    embed_ms = int((time.time() - t0) * 1000)

    render_stage(stage_embed_ph, "2️⃣", "Embedding Query",
                 f"Done in {embed_ms} ms · 384 floats generated",
                 value=f"[{', '.join(str(round(v,4)) for v in q_vec[:6])} … +378 more]",
                 state="done")

    # Update vector tab
    vec_intro.success(f"✅ Query embedded in {embed_ms} ms — 384-dimensional vector created")

    qv_df = px.bar(
        x=[f"D{i}" for i in range(60)],
        y=q_vec[:60],
        color=q_vec[:60],
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        labels={"x": "Dimension", "y": "Value", "color": "Value"},
        title="Your query as a 384-dim vector (first 60 shown)"
    )
    qv_df.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3", height=280, showlegend=False,
        xaxis=dict(tickangle=45, tickfont=dict(size=7), gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d")
    )
    vec_chart.plotly_chart(qv_df, use_container_width=True)

    # Stats
    vec_stats.markdown(f"""
| Stat | Value |
|---|---|
| Dimensions | 384 |
| Min | `{q_vec.min():.4f}` |
| Max | `{q_vec.max():.4f}` |
| Mean | `{q_vec.mean():.4f}` |
| L2 Norm | `{np.linalg.norm(q_vec):.4f}` |
| Embed time | `{embed_ms} ms` |
""")

    # PCA plot — query among chunks
    combined = np.vstack([all_vecs, q_vec.reshape(1, -1)])
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)
    chunk_xy = coords[:-1]
    query_xy = coords[-1]

    sims_preview = cosine_similarity(q_vec.reshape(1,-1), all_vecs)[0]
    top_preview  = np.argsort(sims_preview)[::-1][:n_results]
    colours = ["#f0883e" if i in top_preview else "#1f6feb" for i in range(n_chunks)]

    fig_pca = go.Figure()
    fig_pca.add_scatter(
        x=chunk_xy[:,0], y=chunk_xy[:,1],
        mode="markers+text",
        text=[f"C{i+1}" for i in range(n_chunks)],
        textposition="top center",
        marker=dict(size=10, color=colours, line=dict(width=1, color="#fff")),
        name="Chunks",
        hovertext=[d[:80]+"…" for d in all_docs],
        hovertemplate="%{hovertext}<extra></extra>"
    )
    for idx in top_preview:
        fig_pca.add_shape(type="line",
            x0=query_xy[0], y0=query_xy[1],
            x1=chunk_xy[idx,0], y1=chunk_xy[idx,1],
            line=dict(color="#f0883e", width=1.5, dash="dot")
        )
    fig_pca.add_scatter(
        x=[query_xy[0]], y=[query_xy[1]],
        mode="markers+text", text=["QUERY"],
        textposition="bottom center",
        marker=dict(size=20, color="#ff4f4f", symbol="star",
                    line=dict(width=2, color="#fff")),
        name="Query"
    )
    fig_pca.update_layout(
        title="Query position in vector space (PCA 2D) — orange = retrieved",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3", height=380,
        xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#21262d")
    )
    vec_pca.plotly_chart(fig_pca, use_container_width=True)
    time.sleep(0.3)

    # ── STAGE 3: Vector Search ────────────────────────────────────────────────
    render_stage(stage_search_ph, "3️⃣", "Vector Search",
                 f"Computing cosine similarity vs {n_chunks} chunks…", state="active")

    t1 = time.time()
    sims = cosine_similarity(q_vec.reshape(1,-1), all_vecs)[0]
    search_ms = int((time.time() - t1) * 1000)
    top_idx   = np.argsort(sims)[::-1][:n_results]

    render_stage(stage_search_ph, "3️⃣", "Vector Search",
                 f"Done in {search_ms} ms · top score: {sims[top_idx[0]]:.4f}",
                 value=f"Searched {n_chunks} vectors · returned top {n_results}",
                 state="done")

    # Update search tab
    search_intro.success(f"✅ Cosine similarity computed against all {n_chunks} chunks in {search_ms} ms")

    sim_df_data = sorted(
        [{"Chunk": f"C{i+1}", "Preview": all_docs[i][:55]+"…", "Similarity": round(float(sims[i]), 4)}
         for i in range(n_chunks)],
        key=lambda x: x["Similarity"], reverse=True
    )
    colours_bar = ["#f0883e" if d["Chunk"] in [f"C{i+1}" for i in top_idx] else "#1f6feb"
                   for d in sim_df_data]

    fig_bar = go.Figure(go.Bar(
        x=[d["Chunk"] for d in sim_df_data],
        y=[d["Similarity"] for d in sim_df_data],
        marker_color=colours_bar,
        hovertext=[d["Preview"] for d in sim_df_data],
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<br>%{hovertext}<extra></extra>"
    ))
    fig_bar.add_hline(
        y=float(sims[top_idx[-1]]),
        line_dash="dash", line_color="#3fb950",
        annotation_text=f"retrieval threshold ({sims[top_idx[-1]]:.4f})",
        annotation_font_color="#3fb950"
    )
    fig_bar.update_layout(
        title=f"Cosine similarity scores — orange bars are retrieved (top {n_results})",
        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font_color="#e6edf3", height=320,
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Cosine Similarity", range=[0, 1])
    )
    search_bar.plotly_chart(fig_bar, use_container_width=True)
    time.sleep(0.3)

    # ── STAGE 4: Retrieve Chunks ──────────────────────────────────────────────
    render_stage(stage_retrieve_ph, "4️⃣", "Retrieve Top Chunks",
                 f"Fetching top {n_results} chunks from ChromaDB…", state="active")

    retrieved_docs   = [all_docs[i]   for i in top_idx]
    retrieved_scores = [sims[i]        for i in top_idx]
    context_text     = "\n\n---\n\n".join(retrieved_docs)

    render_stage(stage_retrieve_ph, "4️⃣", "Retrieve Top Chunks",
                 f"{n_results} chunks retrieved · {len(context_text)} chars of context",
                 value=f"Chunks: {', '.join(f'C{i+1}' for i in top_idx)}",
                 state="done")

    # Update chunks tab
    chunk_intro.success(f"✅ {n_results} chunks retrieved from ChromaDB")
    cards_html = ""
    for rank, (doc, score, idx) in enumerate(zip(retrieved_docs, retrieved_scores, top_idx)):
        cards_html += f"""
        <div class="chunk-card">
          <span class="chunk-score">#{rank+1} &nbsp; C{idx+1} &nbsp; score: {score:.4f}</span><br>
          <span style="color:#8b949e; font-size:0.85em;">{doc}</span>
        </div>
        """
    chunk_cards.markdown(cards_html, unsafe_allow_html=True)
    time.sleep(0.3)

    # ── STAGE 5: Build Prompt ─────────────────────────────────────────────────
    render_stage(stage_prompt_ph, "5️⃣", "Build Prompt",
                 "Assembling system message + context + user query…", state="active")

    system_msg = (
        "You are a helpful course advisor for Pinnacle Training Group. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say so honestly.\n\n"
        f"CONTEXT:\n{context_text}"
    )
    full_prompt = f"SYSTEM:\n{system_msg}\n\nUSER:\n{query}"
    approx_tokens = len(full_prompt.split()) * 1.3

    render_stage(stage_prompt_ph, "5️⃣", "Build Prompt",
                 f"Prompt built · ~{int(approx_tokens)} tokens",
                 value=f"system ({len(system_msg)} chars) + user ({len(query)} chars)",
                 state="done")

    # Update prompt tab
    prompt_intro.success(f"✅ Full prompt assembled — ~{int(approx_tokens)} tokens")
    prompt_box.markdown(f'<div class="prompt-box">{full_prompt}</div>', unsafe_allow_html=True)
    token_box.markdown(f"""
<div class="token-box">
📊 Estimated tokens: ~{int(approx_tokens)} &nbsp;|&nbsp;
System: ~{int(len(system_msg.split())*1.3)} &nbsp;|&nbsp;
User: ~{int(len(query.split())*1.3)} &nbsp;|&nbsp;
Cost estimate: ~${(approx_tokens/1_000_000)*0.25:.5f} USD
</div>
""", unsafe_allow_html=True)
    time.sleep(0.3)

    # ── STAGE 6: Claude Generates ─────────────────────────────────────────────
    render_stage(stage_llm_ph, "6️⃣", "Claude Generates",
                 "Sending to claude-haiku-4-5-20251001…", state="active")

    resp_intro.info("⏳ Claude is generating a response…")

    t2 = time.time()
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_msg,
        messages=[{"role": "user", "content": query}]
    )
    llm_ms    = int((time.time() - t2) * 1000)
    answer    = response.content[0].text
    in_tokens = response.usage.input_tokens
    out_tokens= response.usage.output_tokens
    cost      = (in_tokens * 0.25 + out_tokens * 1.25) / 1_000_000

    render_stage(stage_llm_ph, "6️⃣", "Claude Generates",
                 f"Response received in {llm_ms} ms · {out_tokens} output tokens",
                 value=answer[:120] + "…",
                 state="done")

    # Update response tab
    resp_intro.success(f"✅ Response received in {llm_ms} ms")
    resp_box.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
    resp_meta.markdown(f"""
| Metric | Value |
|---|---|
| Model | `claude-haiku-4-5-20251001` |
| Input tokens | `{in_tokens}` |
| Output tokens | `{out_tokens}` |
| Response time | `{llm_ms} ms` |
| Estimated cost | `${cost:.6f} USD` |
| Chunks used | `{n_results}` |
| Context length | `{len(context_text)} chars` |
""")

    # ── Final summary banner ──────────────────────────────────────────────────
    detail_header.success(
        f"✅ Pipeline complete — {embed_ms + search_ms + llm_ms} ms total "
        f"| {in_tokens + out_tokens} tokens | ${cost:.6f} cost"
    )

elif run_btn and not query.strip():
    st.warning("Please enter a question first.")
