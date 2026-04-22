import os
from qdrant_client.models import PointStruct

# Internal imports
from embedder import embed
from schema import client, COLLECTION_NAME, create_collection
from doc_loader import load_docx
from chunking import chunk_text

def ingest():
    print("🚀 Starting ingestion...")

    # --- PATH LOGIC START ---
    # 1. Get the directory where this script is located (e.g., project/app)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up one level to the root and then into the 'data' folder
    base_dir = os.path.abspath(os.path.join(current_script_dir, "..", "data"))
    # --- PATH LOGIC END ---

    print(f"📂 Loading documents from: {base_dir}")

    # Safety check: Ensure the directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Error: The directory '{base_dir}' was not found.")
        return

    docs = load_docx(base_dir)

    if not docs:
        print("❌ No DOCX files found in data folder!")
        return

    print(f"📄 Found {len(docs)} document(s)")

    all_chunks = []

    for i, doc in enumerate(docs):
        chunks = chunk_text(doc)
        print(f"✂️ Doc {i+1}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("❌ No text chunks generated!")
        return

    # Create collection dynamically based on embedding dimensions
    vector_size = len(embed("test"))
    create_collection(vector_size)

    print(f"🧠 Embedding {len(all_chunks)} chunks...")

    points = []

    for i, chunk in enumerate(all_chunks):
        points.append(
            PointStruct(
                id=i,
                vector=embed(chunk),
                payload={
                    "text": chunk,
                    "chunk_id": i
                }
            )
        )

    print("📤 Uploading to Qdrant...")

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print("✅ Ingestion complete! Data stored in Qdrant.")


if __name__ == "__main__":
    ingest()