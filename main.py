# ================================
# MAIN FILE
# ================================

from preprocess import load_pdfs, chunk_text
from rag_pipeline import create_vector_db, generate_answer

print("Loading PDFs...")
documents = load_pdfs("data")

print("Chunking text...")
chunks = chunk_text(documents)

print("Creating embeddings...")
create_vector_db(chunks)

print("\nRAG System Ready!\n")

# Chat loop
while True:
    query = input("Ask AI/ML Question (or type exit): ")

    if query.lower() == "exit":
        break

    answer = generate_answer(query)
    print("\nAnswer:\n", answer)
