from pypdf import PdfReader
from rag import client, embed, create_collection, COLLECTION

from uuid import uuid4
from rag import client, embed, create_collection, COLLECTION
from dotenv import load_dotenv
load_dotenv()

def ingest_book(pdf_path: str):
    create_collection()
    reader = PdfReader(pdf_path)

    points = []

    for page_no, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        paragraphs = text.split("\n\n")
        for para in paragraphs:
            if len(para.strip()) < 50:
                continue

            points.append({
                "id": str(uuid4()),
                "vector": embed(para),
                "payload": {
                    "text": para,
                    "page": page_no + 1
                }
            })

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    print(f"✅ Ingested {len(points)} chunks")

if __name__ == "__main__":
    ingest_book("book.pdf")
