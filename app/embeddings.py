# app/embeddings.py
from sentence_transformers import SentenceTransformer

# Lightweight, public, CPU-friendly model
model = SentenceTransformer("all-MiniLM-L6-v2")  

def embed(text: str) -> list[float]:
    return model.encode(text).tolist()
