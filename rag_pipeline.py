# ==========================================
# RAG PIPELINE (FINAL CORRECT VERSION)
# ==========================================

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# =========================
# Load embedding model
# =========================
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# Load generator model
# =========================
print("Loading generator model...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

# =========================
# Global storage
# =========================
stored_chunks = []
index = None

# =========================
# Create vector DB
# =========================
def create_vector_db(chunks):
    global stored_chunks, index

    stored_chunks = chunks

    print("Creating embeddings...")
    embeddings = embed_model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    print("Storing in FAISS...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

# =========================
# Retrieve best chunks
# =========================
def retrieve(query, top_k=3):
    global stored_chunks, index

    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(stored_chunks[idx])

    return results

# =========================
# Generate final answer
# =========================
def generate_answer(query):
    retrieved_chunks = retrieve(query)

    # limit context to avoid >512 token warning
    context = " ".join(retrieved_chunks)
    context = context[:1200]   # trim long context

    prompt = f"""
You are an AI assistant helping students learn Artificial Intelligence and Machine Learning.

Use the context below to answer correctly and clearly.
If answer not found, say "Answer not found in notes".

Context:
{context}

Question:
{query}

Answer:
"""

    result = generator(prompt)[0]['generated_text']
    return result.strip()
