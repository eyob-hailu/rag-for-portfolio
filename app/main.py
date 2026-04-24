from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.search import query_points
from app.llm import generate_answer

app = FastAPI()

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

    results = query_points(query.query)

    docs = [r.payload["text"] for r in results]

    context = "\n".join(docs)

    answer = generate_answer(context, query.query)

    return {
        "query": query.query,
        "answer": answer,
        "sources": docs
    }
@app.get("/health")
def health():
    return {"status": "ok"}