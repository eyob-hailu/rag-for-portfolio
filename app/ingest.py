import os
import uuid
from qdrant_client.models import PointStruct

from app.embedder import embed_texts
from app.schema import get_client, COLLECTION_NAME, create_collection
from app.doc_loader import load_docx
from app.chunking import chunk_text


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
    vectors = embed_texts(all_chunks)

    if not vectors:
        raise ValueError("❌ No vectors returned by embedding API")

    vector_size = len(vectors[0])
    for vector in vectors:
        if len(vector) != vector_size:
            raise ValueError("❌ Inconsistent embedding dimensions returned by API")

    # Ensure the collection exists (creates it if missing)
    create_collection(vector_size=vector_size)

    # --- BUILD POINTS ---
    points = []
    for chunk, vector in zip(all_chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, "source": "portfolio-doc"},
            )
        )

    print("📤 Uploading to Qdrant...")

    client = get_client()
    client.upsert(
        collection_name=COLLECTION_NAME,  # "portfolio-collection"
        points=points,
        wait=True
    )

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest()