from bs4 import BeautifulSoup
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from embeddings import embed  # local free embeddings


# Convert chunks to Document objects

# -------------------------------
# Settings
# -------------------------------
DOCS_BASE_URL = "https://muaazasif.github.io/physical-ai-robotics-website/docs"
CHAPTERS = [
    "chapter_1_introduction",
    "chapter_2_physical_ai",
    "chapter_3_visualization_animation",
    "chapter_4_ml_algorithms",
    "chapter_5_control_architecture",
    "chapter_6_sensors_perception"
    # add more chapters if you want
]
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "book_docs"

# -------------------------------
# Initialize Qdrant client
# -------------------------------
from qdrant_client import QdrantClient

client = QdrantClient(
    url=QDRANT_URL, 
    api_key=QDRANT_API_KEY,
)



# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # all-MiniLM-L6-v2 dim=384
    )

# -------------------------------
# Fetch and parse HTML chapters
# -------------------------------
all_texts = []

for chapter in CHAPTERS:
    url = f"{DOCS_BASE_URL}/{chapter}"
    print(f"[INFO] Fetching {url}...")
    r = requests.get(url)
    if r.status_code == 200:
        soup = BeautifulSoup(r.text, "html.parser")
        # remove scripts and style
        for s in soup(["script", "style"]):
            s.decompose()
        text = soup.get_text(separator="\n")
        all_texts.append(text)
    else:
        print(f"[WARN] Failed to fetch {url} (status {r.status_code})")

print(f"[INFO] Fetched {len(all_texts)} chapters.")

# -------------------------------
# Split into chunks
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []
for doc in all_texts:
    doc_chunks = splitter.split_text(doc)
    chunks.extend(doc_chunks)

print(f"[INFO] Split into {len(chunks)} chunks.")

# -------------------------------
# Generate embeddings (local, free)
# -------------------------------
vectors = []
metadatas = []

for i, chunk in enumerate(chunks):
    vector = embed(chunk)
    vectors.append(vector)
    metadatas.append({"text": chunk, "source": f"chapter_chunk_{i}"})

# -------------------------------
# Upload to Qdrant
# -------------------------------
# Prepare metadata
metadatas = [{"source": f"chapter_chunk_{i}"} for i, chunk in enumerate(chunks)]
vectors = [embed(chunk) for chunk in chunks]

# Upload to Qdrant directly from texts
client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        {
            "id": i,
            "vector": vectors[i],
            "payload": {"source": f"chapter_chunk_{i}", "text": chunks[i]}
        }
        for i in range(len(chunks))
    ]
)



print("[INFO] Ingestion complete. All chunks uploaded to Qdrant.")
