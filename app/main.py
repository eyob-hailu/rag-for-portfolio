from fastapi import FastAPI
from pydantic import BaseModel

from app.search import query_points
from app.llm import generate_answer

app = FastAPI()

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