# Hugging Face Sentence-Transformers all-MiniLM-L6-v2 Fix

## Overview
This repository contains a step-by-step guide and resources to fix the **Internal Server Error** encountered while using Hugging Face's **Sentence-Transformers all-MiniLM-L6-v2** model. This solution is suitable for developers, AI enthusiasts, and ML engineers working with NLP models.

It also includes a minimal RAM-friendly approach using the **Hugging Face InferenceClient**, ideal for platforms like Railway.

## Features
- Fixes Internal Server Errors when loading or using the model.
- Quick troubleshooting tips.
- Minimal RAM usage method using `InferenceClient`.
- Best practices for using `all-MiniLM-L6-v2` locally or via API.

## Installation
```bash
# Install sentence-transformers and huggingface_hub
pip install sentence-transformers huggingface_hub
```

## Usage (Local Model)
```python
from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode a sample sentence
sentence = "Hello world!"
embedding = model.encode(sentence)
print(embedding)
```

## Usage (Hugging Face InferenceClient)
```python
import os
from huggingface_hub import InferenceClient

# Initialize the client
client = InferenceClient(api_key=os.environ.get("HF_TOKEN"))

def embed(text: str) -> list[float]:
    try:
        embedding = client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
        if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
            return embedding[0]
        return embedding
    except Exception as e:
        print(f"Hugging Face Client Error: {e}")
        raise Exception(f"Failed to get embeddings: {str(e)}")

# Example usage
vector = embed("Hello world!")
print(vector)
```

## Troubleshooting
- Ensure you have a stable internet connection.
- Update `sentence-transformers` and `huggingface_hub` to the latest versions.
- Ensure your `HF_TOKEN` environment variable is set correctly.
- Check system memory: the model may require sufficient RAM if running locally.

## Resources
- Hugging Face Model: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Hugging Face Documentation: [Transformers Docs](https://huggingface.co/docs/transformers/index)
- Example embedding script: [embeddings.py](https://github.com/muaazasif/rag-book-chatbot/blob/main/app/embeddings.py)

## YouTube SEO Description (for your video)
```
🚀 Learn how to fix the **Internal Server Error** when using Hugging Face's **Sentence-Transformers all-MiniLM-L6-v2**! 

This tutorial shows two approaches:
1️⃣ Using the local model with `sentence-transformers`.
2️⃣ Using the **Hugging Face InferenceClient** for minimal RAM usage on platforms like Railway.

✅ What you'll learn:
- How to resolve Internal Server Errors
- Quick troubleshooting for sentence-transformers
- Using InferenceClient for efficient embeddings
- Best practices for NLP model deployment

🔗 Resources:
- Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Documentation: https://huggingface.co/docs/transformers/index
- Script Example: https://github.com/muaazasif/rag-book-chatbot/blob/main/app/embeddings.py

👍 Like, subscribe, and share if this helped you!
```

## License
MIT License
