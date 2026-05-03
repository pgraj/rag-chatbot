"""
chat_ui.py
──────────
Clean chat interface with visible input box.
Writes every query to query_bridge.json so pipeline_monitor.py reacts in real time.

Run with: py -m streamlit run chat_ui.py --server.port 8501
"""

import os
import json
import time
import numpy as np
import streamlit as st
import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

BRIDGE_FILE = "./query_bridge.json"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Course Advisor AI",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1117;
    color: #e6edf3;
}
.chat-bubble-user {
    background: #1f6feb;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
    color: #fff;
    font-size: 0.95em;
}
.chat-bubble-ai {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 85%;
    color: #e6edf3;
    font-size: 0.95em;
    line-height: 1.6;
}
.meta-tag {
    font-size: 0.72em;
    color: #8b949e;
    margin-top: 4px;
    margin-bottom: 12px;
}
.bridge-badge {
    background: #0d2818;
    border: 1px solid #238636;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.78em;
    color: #3fb950;
    margin-bottom: 16px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    chroma   = chromadb.PersistentClient(path="./chroma_db")
    col      = chroma.get_or_create_collection("course_docs")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client   = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return col, embedder, client

@st.cache_data
def load_all_vectors():
    col, _, _ = load_resources()
    r = col.get(include=["documents", "embeddings"])
    return r["documents"], np.array(r["embeddings"])

collection, embedder, anthropic_client = load_resources()
all_docs, all_vecs = load_all_vectors()


# ── Bridge writer ─────────────────────────────────────────────────────────────
def write_bridge(status, query="", chunks=None, answer="",
                 scores=None, tokens_in=0, tokens_out=0,
                 embed_ms=0, search_ms=0, llm_ms=0):
    payload = {
        "status":     status,
        "query":      query,
        "timestamp":  time.time(),
        "chunks":     chunks or [],
        "scores":     scores or [],
        "answer":     answer,
        "tokens_in":  tokens_in,
        "tokens_out": tokens_out,
        "embed_ms":   embed_ms,
        "search_ms":  search_ms,
        "llm_ms":     llm_ms,
    }
    with open(BRIDGE_FILE, "w") as f:
        json.dump(payload, f)


# ── RAG function ──────────────────────────────────────────────────────────────
def run_rag(query, n_results=3):
    write_bridge("embedding", query=query)
    t0 = time.time()
    q_vec = embedder.encode([query])[0]
    embed_ms = int((time.time() - t0) * 1000)

    write_bridge("searching", query=query, embed_ms=embed_ms)
    t1 = time.time()
    sims = cosine_similarity(q_vec.reshape(1, -1), all_vecs)[0]
    search_ms = int((time.time() - t1) * 1000)
    top_idx = np.argsort(sims)[::-1][:n_results]

    retrieved_docs   = [all_docs[i]    for i in top_idx]
    retrieved_scores = [float(sims[i]) for i in top_idx]
    context_text     = "\n\n---\n\n".join(retrieved_docs)

    write_bridge("retrieving", query=query,
                 chunks=retrieved_docs, scores=retrieved_scores,
                 embed_ms=embed_ms, search_ms=search_ms)

    system_msg = (
        "You are a helpful course advisor for Pinnacle Training Group. "
        "Answer only using the provided context. "
        "If the answer is not in the context, say so honestly.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    write_bridge("generating", query=query,
                 chunks=retrieved_docs, scores=retrieved_scores,
                 embed_ms=embed_ms, search_ms=search_ms)

    t2 = time.time()
    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        system=system_msg,
        messages=[{"role": "user", "content": query}]
    )
    llm_ms     = int((time.time() - t2) * 1000)
    answer     = response.content[0].text
    tokens_in  = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    write_bridge("done", query=query,
                 chunks=retrieved_docs, scores=retrieved_scores,
                 answer=answer, tokens_in=tokens_in, tokens_out=tokens_out,
                 embed_ms=embed_ms, search_ms=search_ms, llm_ms=llm_ms)

    return answer, tokens_in, tokens_out, llm_ms


# ── Initialise state ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    write_bridge("idle")

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

if "prefill" not in st.session_state:
    st.session_state.prefill = ""


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🎓 Course Advisor AI")
st.markdown(
    '<div class="bridge-badge">🔗 Bridge active — pipeline_monitor.py watching on port 8502</div>',
    unsafe_allow_html=True
)

# ── INPUT SECTION ─────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask a Question")

query = st.text_input(
    label="Type your question here:",
    value=st.session_state.prefill,
    placeholder="e.g. What are the entry requirements for the Diploma of Nursing?",
    key=f"query_input_{st.session_state.input_key}"
)

col1, col2 = st.columns([3, 1])
with col1:
    ask_btn = st.button("▶  Ask", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("🗑 Clear chat", use_container_width=True)

# ── Sample questions ──────────────────────────────────────────────────────────
with st.expander("💡 Click a sample question"):
    samples = [
        "What are the entry requirements for the Diploma of Nursing?",
        "Which courses have government funding available?",
        "How long does Certificate III in Individual Support take?",
        "Is RPL available for the Diploma of Project Management?",
        "What is the tuition fee for the ICT Diploma?",
        "How do I enrol at Pinnacle Training Group?",
        "What campuses does Horizon Health Institute operate from?",
    ]
    for s in samples:
        if st.button(s, key=f"sample_{s}"):
            st.session_state.prefill = s
            st.session_state.input_key += 1
            st.rerun()

st.divider()

# ── Clear handler ─────────────────────────────────────────────────────────────
if clear_btn:
    st.session_state.messages = []
    st.session_state.input_key += 1
    write_bridge("idle")
    st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("### Conversation")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-ai">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="meta-tag">'
                f'⏱ {msg.get("llm_ms","?")} ms &nbsp;|&nbsp;'
                f'🪙 {msg.get("tokens","?")} tokens &nbsp;|&nbsp;'
                f'💰 ${msg.get("cost","?")}'
                f'</div>',
                unsafe_allow_html=True
            )
else:
    st.info("No conversation yet — type a question above and click Ask.")

# ── Handle submission ─────────────────────────────────────────────────────────
if ask_btn and query and query.strip():
    st.session_state.prefill = ""
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🔍 Searching and generating answer…"):
        answer, tokens_in, tokens_out, llm_ms = run_rag(query)

    total_tokens = tokens_in + tokens_out
    cost = (tokens_in * 0.25 + tokens_out * 1.25) / 1_000_000

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "llm_ms":  llm_ms,
        "tokens":  total_tokens,
        "cost":    f"{cost:.5f}"
    })

    st.session_state.input_key += 1
    st.rerun()

elif ask_btn and not query.strip():
    st.warning("⚠️ Please type a question first.")