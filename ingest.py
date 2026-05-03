import os
from anthropic import Anthropic
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Clients
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# ── CHANGE THIS LINE ── saves to a folder called "chroma_db"
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.create_collection("course_docs")

# Free local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # downloads once, ~80MB

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# Load your PDF
reader = PdfReader("course_catalog.pdf")
full_text = " ".join(page.extract_text() for page in reader.pages)

chunks = chunk_text(full_text)
embeddings = embedder.encode(chunks).tolist()  # batch embeds all chunks at once

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        documents=[chunk],
        embeddings=[embedding],
        ids=[f"chunk_{i}"]
    )

print(f"Indexed {len(chunks)} chunks")




