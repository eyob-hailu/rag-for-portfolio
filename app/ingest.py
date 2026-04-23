import os
import uuid
from qdrant_client.models import PointStruct

from embedder import embed
from schema import client, COLLECTION_NAME
from doc_loader import load_docx
from chunking import chunk_text


def ingest():
    print("🚀 Starting ingestion...")

    # --- PATH ---
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )

    print(f"📂 Loading documents from: {base_dir}")

    if not os.path.exists(base_dir):
        print(f"❌ Data folder not found: {base_dir}")
        return

    docs = load_docx(base_dir)

    if not docs:
        print("❌ No documents found!")
        return

    print(f"📄 Found {len(docs)} document(s)")

    # --- CHUNKING ---
    all_chunks = []
    for i, doc in enumerate(docs):
        chunks = chunk_text(doc)
        print(f"✂️ Doc {i+1}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("❌ No chunks generated!")
        return

    print(f"🧠 Embedding {len(all_chunks)} chunks...")

    # --- BUILD POINTS ---
    points = []

    for chunk in all_chunks:
        vector = embed(chunk)

        if len(vector) != 384:
            raise ValueError(f"❌ Invalid embedding size: {len(vector)} (expected 384)")

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,   # ✅ UNNAMED VECTOR (IMPORTANT)
                payload={
                    "text": chunk,
                    "source": "portfolio-doc"
                }
            )
        )

    print("📤 Uploading to Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,  # "portfolio-collection"
        points=points,
        wait=True
    )

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest()