# embeddings.py
from sentence_transformers import SentenceTransformer

# 384-dim FREE model (same as your collection)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text: str) -> list[float]:
    """
    Convert text to vector (FREE, local)
    """
    return model.encode(text).tolist()
