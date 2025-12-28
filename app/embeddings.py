# embeddings.py
import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

def embed(text: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    res = requests.post(API_URL, headers=headers, json={"inputs": text})
    return res.json()[0]
