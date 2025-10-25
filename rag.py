import os
import requests
from typing import List, Dict, Any

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHAT_URL = f"{OLLAMA_HOST}/api/chat"
EMBED_URL = f"{OLLAMA_HOST}/api/embeddings"

CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

def _pick(vec):
    # ensure we always return a 1-D list[float]
    if vec is None:
        return None
    if isinstance(vec, list):
        # vec can be [float,...] or [[float,...], ...]
        return vec[0] if vec and isinstance(vec[0], list) else vec
    return None

def _extract_embedding(j: dict):
    """
    Support all known shapes:
      {"embedding":[...]}
      {"embeddings":[[...]]} or {"embeddings":[...]}
      {"data":[{"embedding":[...]}]}
    """
    if "embedding" in j:
        return _pick(j["embedding"])
    if "embeddings" in j:
        return _pick(j["embeddings"])
    if "data" in j and isinstance(j["data"], list) and j["data"]:
        if isinstance(j["data"][0], dict) and "embedding" in j["data"][0]:
            return _pick(j["data"][0]["embedding"])
    return None

def _post_embed(payload):
    r = requests.post(EMBED_URL, json=payload, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama embeddings HTTP {r.status_code}: {r.text[:200]}")
    emb = _extract_embedding(r.json())
    return emb

def embed_texts(texts: List[str]) -> List[List[float]]:
    out = []
    for t in texts:
        # 1) Try array input (newer behavior)
        emb = _post_embed({"model": EMBED_MODEL, "input": [t]})
        if not emb:
            # 2) Try string input
            emb = _post_embed({"model": EMBED_MODEL, "input": t})
        if not emb:
            # 3) Try legacy 'prompt'
            emb = _post_embed({"model": EMBED_MODEL, "prompt": t})
        if not emb or not isinstance(emb, list) or len(emb) == 0:
            raise RuntimeError("Got empty embedding from Ollama. Check that 'nomic-embed-text' is installed and the Ollama service is running.")
        out.append(emb)
    return out

def chat_with_context(prompt: str, context_chunks: List[Dict[str, Any]], temperature: float = 0.2) -> str:
    ctx = ""
    for i, ch in enumerate(context_chunks, 1):
        ctx += f"\n[Chunk {i}] Source: {ch.get('source','unknown')} | Location: {ch.get('loc','')} \n{ch['text']}\n"
    system_msg = (
        "You are a helpful TTRPG rules assistant. Answer ONLY from the provided context. "
        "If the answer isn't present, say exactly: 'I couldn't find that in the book.' "
        "Keep answers concise. When you use a passage, mention its [Chunk #]."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {prompt}"}
    ]
    r = requests.post(CHAT_URL, json={"model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": temperature}}, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]