from app.schema import get_client, COLLECTION_NAME
from app.embedder import embed


def query_points(query: str):
    vector = embed(query)

    client = get_client()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=5,
        with_payload=True
    )

    return results.points