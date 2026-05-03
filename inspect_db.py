import chromadb

chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("course_docs")

# How many chunks are stored?
count = collection.count()
print(f"\n✅ Total chunks indexed: {count}")
print("=" * 60)

# Fetch all chunks with their embeddings
results = collection.get(include=["documents", "embeddings"])

for i, (doc, embedding) in enumerate(zip(results["documents"], results["embeddings"])):
    print(f"\n📄 CHUNK {i+1}")
    print(f"Text preview : {doc[:200]}...")
    print(f"Vector length: {len(embedding)} dimensions")
    print(f"First 5 vals : {[round(v, 4) for v in embedding[:5]]}")
    print("-" * 60)
