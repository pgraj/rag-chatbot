import chromadb
import streamlit as st
import pandas as pd

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("course_docs")

results = collection.get(include=["documents", "embeddings"])

st.title("🔍 ChromaDB Inspector")
st.caption("Visualising your RAG vector database")

# ── Summary ──────────────────────────────────────────────
col1, col2 = st.columns(2)
col1.metric("Total Chunks", len(results["documents"]))
col2.metric("Vector Dimensions", len(results["embeddings"][0]))

st.divider()

# ── Chunk browser ─────────────────────────────────────────
st.subheader("📄 Chunk Browser")

chunk_index = st.slider(
    "Select chunk to inspect",
    min_value=1,
    max_value=len(results["documents"]),
    value=1
)

selected_doc = results["documents"][chunk_index - 1]
selected_vec = results["embeddings"][chunk_index - 1]

st.markdown("**Text Content:**")
st.info(selected_doc)

st.markdown("**Vector (first 20 of 384 dimensions):**")
st.code(str([round(v, 4) for v in selected_vec[:20]]))

st.divider()

# ── All chunks as a table ─────────────────────────────────
st.subheader("📊 All Chunks Table")

df = pd.DataFrame({
    "Chunk #": [i+1 for i in range(len(results["documents"]))],
    "Preview (first 100 chars)": [doc[:100] + "..." for doc in results["documents"]],
    "Vector dim": [len(v) for v in results["embeddings"]]
})

st.dataframe(df, use_container_width=True)

st.divider()

# ── Similarity search tester ──────────────────────────────
st.subheader("🔎 Test Similarity Search")
st.caption("See which chunks get retrieved for a query")

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")

query = st.text_input("Enter a test query:")

if query:
    q_embedding = embedder.encode([query]).tolist()[0]
    search_results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3,
        include=["documents", "distances"]
    )

    for i, (doc, dist) in enumerate(zip(
        search_results["documents"][0],
        search_results["distances"][0]
    )):
        similarity = round(1 - dist, 4)
        st.markdown(f"**Match {i+1} — Similarity score: `{similarity}`**")
        st.success(doc)
        st.divider()
