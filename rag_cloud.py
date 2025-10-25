from typing import List, Dict, Any
import os
from openai import OpenAI
from openai.error import RateLimitError

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

def chat_with_context_openai(prompt: str, context_chunks: List[Dict[str, Any]], temperature: float = 0.2) -> str:
    client = OpenAI()  # reads OPENAI_API_KEY from env
    context_str = ""
    for i, ch in enumerate(context_chunks, start=1):
        context_str += f"\n[Chunk {i}] Source: {ch.get('source','unknown')} | Location: {ch.get('loc','')} \n{ch['text']}\n"
    system_msg = (
        "You are a helpful TTRPG rules assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say exactly: 'I couldn't find that in the book.' "
        "Keep answers concise and cite [Chunk #] when you use a passage."
    )
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION: {prompt}"
    try:
        # Send the chat request to OpenAI.  This may raise RateLimitError if quota is exceeded.
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=[{"role":"system","content":system_msg},{"role":"user","content":user_msg}]
        )
        return resp.choices[0].message.content
    except RateLimitError:
        # Return a helpful message if the API key has insufficient quota.
        return "⚠️ OpenAI API quota exceeded. Please check your plan and billing details."
    except Exception as e:
        # Catch any other OpenAI or network errors and return a generic message.
        return f"⚠️ Error communicating with OpenAI: {e}"