from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import traceback

from app.doc_loader import load_docx
from app.llm import generate_answer

app = FastAPI()
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc)},
    )

class Query(BaseModel):
    query: str


@app.post("/rag")
def rag(query: Query):
    docs = []
    source_items = []
    warning = None
    retrieved_from_file = False
    retrieval_message = ""
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
        loaded_docs = load_docx(base_dir)
        for doc in loaded_docs:
            if doc:
                docs.append(doc)
                source_items.append(
                    {
                        "source": "local_docx",
                        "text": doc,
                    }
                )
        retrieved_from_file = len(source_items) > 0
    except Exception as exc:
        logger.exception("RAG retrieval failed: %s", exc)
        warning = str(exc)
        retrieval_message = "Local document retrieval failed. Check data directory."

    docs = [d for d in docs if d]
    context = "\n".join(docs)
    answer = None

    if not docs:
        answer = "I couldn't find relevant information in the knowledge base right now."
        if warning is None:
            warning = "No context found in local documents."
            retrieval_message = "Document retrieval succeeded but returned no context."
    else:
        answer = generate_answer(context, query.query)
        retrieval_message = f"Retrieved context from local documents."

    return {
        "query": query.query,
        "answer": answer,
        "sources": source_items,
        "retrieved_from_qdrant": False,
        "retrieved_from_file": retrieved_from_file,
        "retrieval_message": retrieval_message,
        "warning": warning,
    }
@app.get("/health")
def health():
    return {"status": "ok"}