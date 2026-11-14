from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import traceback

from app.retriever import ResumeRetriever
from app.llm import generate_answer

app = FastAPI()
logger = logging.getLogger(__name__)

# Initialize in-memory retriever once on startup
retriever = ResumeRetriever()

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


class ChatMessage(BaseModel):
    role: str
    content: str


class Query(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = []


@app.post("/rag")
def rag(query: Query, request: Request):
    warning = None
    retrieved_from_file = False
    retrieval_message = ""
    source_items = []
    answer = None

    # Pure client-side history: each tab maintains its own chat;
    # refreshing the tab resets history and starts completely fresh.
    active_history = []
    if query.history:
        active_history = [{"role": m.role, "content": m.content} for m in query.history]

    try:
        search_query = query.query
        effective_query = query.query

        # Detect follow-ups like "how" or "where" after the bot suggested reaching out
        last_bot_msg = ""
        if active_history:
            last_bot_msg = next(
                (m["content"].lower() for m in reversed(active_history) if m["role"] == "assistant"), ""
            )

        clean_q = query.query.strip().lower().rstrip("?")
        is_contact_followup = (
            any(phrase in last_bot_msg for phrase in ["reach out", "contact", "touch", "email"])
            and clean_q in [
                "how", "how can i", "how so", "where", "where can i",
                "how to reach him", "how to contact him", "what is his email", "contact info"
            ]
        )

        if is_contact_followup:
            search_query = "contact email phone LinkedIn telegram reach out Eyob"
            effective_query = "How can I contact and reach out to Eyob directly?"
        elif len(search_query.split()) < 3 and active_history:
            last_user_msg = next(
                (m["content"] for m in reversed(active_history) if m["role"] == "user"), None
            )
            if last_user_msg:
                search_query = f"{last_user_msg} {search_query}"

        # Retrieve top relevant sections via in-memory RAG
        retrieved_chunks = retriever.retrieve(search_query, top_k=2)
        retrieved_from_file = len(retrieved_chunks) > 0

        for chunk in retrieved_chunks:
            source_items.append({
                "source": chunk["title"],
                "category": chunk.get("category", ""),
                "score": chunk.get("score", 0.0),
                "text": chunk["text"],
            })

        if not retrieved_chunks:
            answer = "I couldn't find relevant information in the knowledge base right now."
            warning = "No matching sections found in knowledge base."
            retrieval_message = "Retrieval returned no context."
        else:
            context = "\n\n".join(
                [f"### {item['title']}\n{item['text']}" for item in retrieved_chunks]
            )
            answer = generate_answer(context, effective_query, history=active_history)
            titles = ", ".join([c["title"] for c in retrieved_chunks])
            retrieval_message = f"Retrieved {len(retrieved_chunks)} section(s) via RAG: {titles}"

    except Exception as exc:
        logger.exception("RAG processing failed: %s", exc)
        warning = str(exc)
        retrieval_message = "Document retrieval failed."
        answer = "I encountered an issue retrieving information. Please try again or contact Eyob directly."

    return {
        "query": query.query,
        "answer": answer,
        "sources": source_items,
        "retrieved_from_file": retrieved_from_file,
        "retrieval_message": retrieval_message,
        "warning": warning,
    }


@app.head("/health")
@app.get("/health")
def health():
    return {"status": "ok"}
