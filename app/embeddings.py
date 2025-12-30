import os
from huggingface_hub import InferenceClient

# Initialize the client (Railway will pick up the HF_TOKEN from environment)
# Using a 'Read' token is perfectly fine here.
client = InferenceClient(
    api_key=os.environ.get("HF_TOKEN")
)

def embed(text: str) -> list[float]:
    """
    Uses the Official Hugging Face InferenceClient to get vectors.
    This uses minimal RAM on Railway.
    """
    try:
        # feature_extraction is the correct method for sentence-transformers
        # it returns the numerical vector (embedding)
        embedding = client.feature_extraction(
            text,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # The result is often a NumPy-like list or a nested list.
        # We ensure it's a flat list of floats.
        if isinstance(embedding, list):
            # If the response is [[...]], we take the first item
            if len(embedding) > 0 and isinstance(embedding[0], list):
                return embedding[0]
            return embedding
            
        # In some versions it returns a numpy array, convert to list
        return embedding.tolist()

    except Exception as e:
        print(f"Hugging Face Client Error: {e}")
        raise Exception(f"Failed to get embeddings: {str(e)}")
