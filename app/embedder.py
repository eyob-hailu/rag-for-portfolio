import os
from typing import List

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_embedding_model = os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1.5")


def embed(text: str) -> List[float]:
    response = _client.embeddings.create(
        model=_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = _client.embeddings.create(
        model=_embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]