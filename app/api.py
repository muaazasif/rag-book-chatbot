from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from .embeddings import embed
from .llm import generate_answer
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
app = FastAPI(title="DocuSaurAI Backend")

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COLLECTION_NAME = "book_docs"

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Safety check
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT env variables missing")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QuestionRequest):
    vector = embed(req.question)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
        with_payload=True
    )

    context = "\n".join([p.payload.get("text", "") for p in results.points])
    answer = generate_answer(req.question, context)

    return {"question": req.question, "answer": answer}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
