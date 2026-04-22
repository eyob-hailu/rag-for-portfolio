from schema import client, COLLECTION_NAME
from embedder import embed


def search(query: str):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embed(query),
        limit=5
    )

    return [r.payload["text"] for r in results]