from typing import List, Dict, Any
import os
import openai
from openai import OpenAI

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

# Ensure an OpenAI API key is set for the OpenAI client.  The default key
# defined here will be used if the OPENAI_API_KEY environment variable is
# not already set.  In production, do not hard-code secrets; set them via
# environment variables or a secure vault.
os.environ.setdefault("OPENAI_API_KEY", "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")

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
    except openai.RateLimitError:
        # Return a helpful message if the API key has insufficient quota.
        return "⚠️ OpenAI API quota exceeded. Please check your plan and billing details."
    except Exception as e:
        # Catch any other OpenAI or network errors and return a generic message.
        return f"⚠️ Error communicating with OpenAI: {e}"