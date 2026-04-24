from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

from app.search import query_points
from app.llm import generate_answer

app = FastAPI()
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str


@app.post("/rag")
def rag(query: Query):
    docs = []
    source_items = []
    warning = None
    retrieved_from_qdrant = False
    retrieval_message = ""
    try:
        results = query_points(query.query)
        for r in results:
            if not r.payload:
                continue
            text = r.payload.get("text", "")
            source = r.payload.get("source", "qdrant")
            if text:
                docs.append(text)
                source_items.append(
                    {
                        "source": source,
                        "text": text,
                    }
                )
        retrieved_from_qdrant = len(source_items) > 0
    except Exception as exc:
        logger.exception("RAG retrieval failed: %s", exc)
        warning = str(exc)
        retrieval_message = "Qdrant retrieval failed. Check embedding/model/API settings."

    docs = [d for d in docs if d]
    context = "\n".join(docs)
    answer = None

    if not docs:
        answer = "I couldn't find relevant information in the knowledge base right now."
        if warning is None:
            warning = "No matching context found in Qdrant."
            retrieval_message = "Qdrant retrieval succeeded but returned no matching context."
    else:
        answer = generate_answer(context, query.query)
        retrieval_message = f"Retrieved {len(source_items)} context chunk(s) from Qdrant."

    return {
        "query": query.query,
        "answer": answer,
        "sources": source_items,
        "retrieved_from_qdrant": retrieved_from_qdrant,
        "retrieval_message": retrieval_message,
        "warning": warning,
    }
@app.get("/health")
def health():
    return {"status": "ok"}