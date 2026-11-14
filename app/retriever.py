import os
import re
import math
from collections import Counter
from typing import List, Dict, Any

from app.chunking import parse_markdown_sections

STOPWORDS = {
    "eyob", "hailu", "he", "his", "him", "what", "who", "how", "where",
    "when", "which", "tell", "me", "about", "doe", "do", "is", "are",
    "the", "a", "an", "in", "of", "for", "with", "on", "at", "to", "from",
    "by", "and", "or", "know", "can", "you", "i", "would", "like"
}

SEMANTIC_TAGS = {
    "Backend & Enterprise Security": [
        "database", "databases", "sql", "nosql", "rdbms", "oracle", "postgres",
        "postgresql", "mongodb", "mysql", "security", "auth", "jwt", "rbac", "backend"
    ],
    "Full-Stack Development": [
        "frontend", "ui", "angular", "react", "tailwind", "css", "fullstack",
        "full-stack", "web", "client", "interface"
    ],
    "Software Developer – Hibret Bank": [
        "bank", "banking", "finance", "financial", "hibret", "job", "career",
        "employment", "experience", "work"
    ],
    "1. Express Spring Security (Featured Open-Source Library)": [
        "npm", "package", "open-source", "open source", "library", "spring security", "express"
    ],
    "📬 CONTACT & PROFILES": [
        "contact", "email", "phone", "call", "reach", "telegram", "linkedin",
        "github", "hire", "message", "socials"
    ],
    "👤 PROFILE": [
        "bio", "about", "education", "degree", "assosa", "cgpa", "university",
        "bachelor", "summary"
    ]
}


def _stem(word: str) -> str:
    w = word.lower()
    if len(w) > 3:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
    return w


def _tokenize(text: str, filter_stopwords: bool = False) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    tokens = [_stem(w) for w in words if len(w) > 1]
    if filter_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


class ResumeRetriever:
    """
    High-performance In-Memory BM25 + Semantic Retriever tailored for portfolio search.
    Requires 0 external vector databases and runs within < 15MB RAM.
    """
    def __init__(self, data_path: str = None):
        if data_path is None:
            data_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "data", "eyob_hailu_resume.md")
            )
        self.data_path = data_path
        self.chunks: List[Dict[str, str]] = []
        self.doc_tokens: List[Counter] = []
        self.avgdl: float = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self.reload()

    def reload(self):
        self.chunks = parse_markdown_sections(self.data_path)
        if not self.chunks:
            self.doc_tokens = []
            self.avgdl = 0.0
            return

        total_length = sum(len(_tokenize(c["text"])) for c in self.chunks)
        self.avgdl = total_length / len(self.chunks)
        self.doc_tokens = []

        for c in self.chunks:
            bold_headers = re.findall(r"\*\*(.*?)\*\*", c["text"])
            bold_tokens = _tokenize(" ".join(bold_headers)) * 3
            t_tokens = _tokenize(c["title"]) * 4
            cat_tokens = _tokenize(c["category"]) * 2
            semantic_tokens = _tokenize(" ".join(SEMANTIC_TAGS.get(c["title"], []))) * 3
            body_tokens = _tokenize(c["text"])
            combined = t_tokens + cat_tokens + bold_tokens + semantic_tokens + body_tokens
            self.doc_tokens.append(Counter(combined))

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        q_tokens = _tokenize(query, filter_stopwords=True)
        # Fallback to Profile if query is a pure greeting or general intro
        if not q_tokens:
            return [{
                "title": self.chunks[0]["title"],
                "category": self.chunks[0]["category"],
                "text": self.chunks[0]["text"],
                "score": 1.0
            }]

        n_docs = len(self.chunks)
        scores = []

        for i, c in enumerate(self.chunks):
            doc_len = len(_tokenize(c["text"]))
            score = 0.0
            for q in q_tokens:
                df = sum(1 for d in self.doc_tokens if q in d)
                if df == 0:
                    continue
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf = self.doc_tokens[i][q]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score += idf * ((tf * (self.k1 + 1.0)) / denom)

            scores.append((score, c))

        scores.sort(key=lambda x: x[0], reverse=True)

        # If top score is 0.0 (no matches), return default overview chunk
        if scores[0][0] == 0.0:
            return [{
                "title": self.chunks[0]["title"],
                "category": self.chunks[0]["category"],
                "text": self.chunks[0]["text"],
                "score": 0.0
            }]

        results = []
        for score, chunk in scores[:top_k]:
            results.append({
                "title": chunk["title"],
                "category": chunk["category"],
                "text": chunk["text"],
                "score": round(score, 2)
            })

        return results

