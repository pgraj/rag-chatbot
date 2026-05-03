import os
import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Rebuild collection from persisted data
# ── CHANGE THIS LINE ── reads from the same persisted folder
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("course_docs")

def ask(question):
    q_embedding = embedder.encode([question]).tolist()[0]
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3
    )
    context = "\n\n".join(results["documents"][0])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        system=(
            "You are a helpful course advisor. "
            "Answer only using the provided context. "
            "If the answer is not in the context, say so honestly.\n\n"
            f"CONTEXT:\n{context}"
        ),
        messages=[{"role": "user", "content": question}]
    )
    return response.content[0].text

# ── Streamlit UI ──────────────────────────────────────────
st.title("🎓 Course Advisor AI")
st.caption("Powered by RAG + Claude")

question = st.text_input("Ask a question about our courses:")

if question:
    with st.spinner("Searching course documents..."):
        answer = ask(question)
    st.markdown(f"**Answer:** {answer}")
