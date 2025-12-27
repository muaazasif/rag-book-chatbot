# rag.py
import os
import uuid
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams
from embeddings import embed
from dotenv import load_dotenv
import os
load_dotenv()
COLLECTION_NAME = "book_34"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT env variables missing")

# ---------- CONFIG ----------
PDF_FILE = "book.pdf"
COLLECTION_NAME = "book_34"
VECTOR_DIM = 384  # embeddings dimension

# Qdrant client setup
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ---------- EMBEDDING FUNCTION ----------
def embed_text(text):
    """
    Replace this with your embedding function.
    Should return a list/vector of length VECTOR_DIM (384).
    """
    # Example: dummy vector for testing
    return [0.0] * VECTOR_DIM

# ---------- CREATE COLLECTION ----------
def create_collection():
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance="Cosine")
        )
        print(f"Created collection '{COLLECTION_NAME}' with {VECTOR_DIM}-dim vectors ✅")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists ✅")

# ---------- INGEST PDF ----------
def ingest_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    points = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        for para in paragraphs:
            vector = embed_text(para)
            point_id = str(uuid.uuid4())  # UUID as point ID
            points.append(PointStruct(id=point_id, vector=vector, payload={"text": para}))

    if points:
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Upserted {len(points)} points into collection '{COLLECTION_NAME}' ✅")
    else:
        print("No text found to ingest ❌")

# ---------- MAIN ----------
if __name__ == "__main__":
    create_collection()
    ingest_pdf(PDF_FILE)
