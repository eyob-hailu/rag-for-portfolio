import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(context, query):
    prompt = f"""
You are an AI assistant for Eyob's portfolio website. 
Answer the user's question accurately using ONLY the information from the context below. 

IMPORTANT RULES:
- Answer naturally. DO NOT say "Based on the provided context", "According to the context", or anything similar. Just give the answer directly.
- If the answer is not in the context, politely say that you don't know.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content