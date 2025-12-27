from qdrant_client import QdrantClient
from embeddings import embed
from llm import generate_answer
from dotenv import load_dotenv
import os
load_dotenv()
COLLECTION_NAME = "book_docs"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT env variables missing")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

def query(question: str):
    # Step 1: convert question to embedding
    vector = embed(question)

    # Step 2: search top-k chunks from Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=3,  # increased top-k for better coverage
        with_payload=True
    )

    points = results.points
    if not points:
        print("❌ No results found in document")
        return

    # Step 3: merge context for LLM (full paragraphs)
    context = "\n".join([p.payload.get("text", "") for p in points if p.payload.get("text")])

    if not context.strip():
        print("❌ No valid text found in document")
        return

    # Step 4: generate strict document-based answer
    answer = generate_answer(question, context)

    print("\n🧠 AI Answer:\n")
    print(answer)


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        query(q)