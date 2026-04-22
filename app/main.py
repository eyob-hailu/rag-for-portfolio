from fastapi import FastAPI
from pydantic import BaseModel

from search import search
from llm import generate_answer

app = FastAPI()

class Query(BaseModel):
    query: str


@app.post("/rag")
def rag(query: Query):

    docs = search(query.query)
    context = "\n".join(docs)

    answer = generate_answer(context, query.query)

    return {
        "query": query.query,
        "answer": answer,
        "sources": docs
    }