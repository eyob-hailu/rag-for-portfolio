import os
import hashlib
from typing import List

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()

_provider = os.getenv("EMBEDDING_PROVIDER", "groq").lower()
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_groq_model = os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
_openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
_local_hash_dim = int(os.getenv("LOCAL_HASH_DIM", "256"))


def _local_hash_embed(text: str) -> List[float]:
    vector = [0.0] * _local_hash_dim
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % _local_hash_dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def embed(text: str) -> List[float]:
    return embed_texts([text])[0]


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    if _provider == "groq":
        try:
            response = _groq_client.embeddings.create(
                model=_groq_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise RuntimeError(
                "Groq embedding failed. Check GROQ_API_KEY and GROQ_EMBEDDING_MODEL "
                f"('{_groq_model}') access."
            ) from exc

    if _provider == "openai":
        try:
            response = _openai_client.embeddings.create(
                model=_openai_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise RuntimeError(
                "OpenAI embedding failed. Check OPENAI_API_KEY and OPENAI_EMBEDDING_MODEL "
                f"('{_openai_model}'). Provider message: {exc}"
            ) from exc

    if _provider == "local_hash":
        return [_local_hash_embed(text) for text in texts]

    raise RuntimeError(
        "Invalid EMBEDDING_PROVIDER. Use 'groq', 'openai', or 'local_hash'."
    )