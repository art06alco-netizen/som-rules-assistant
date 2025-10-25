import os
# Disable OpenTelemetry instrumentation globally before importing any Chroma or LangChain modules.
os.environ["OPENTELEMETRY_SDK_DISABLED"] = "true"
import streamlit as st
from chromadb.config import Settings
from typing import List, Dict, Any
from rag import chat_with_context

# Import the LangChain Chroma vector store and embeddings so that retrieval
# uses the same interface as ingestion.  This avoids mismatches between
# chromadb's low-level client and LangChain's wrapper and ensures we use
# the same embeddings as during indexing.
# Use the standalone langchain-chroma package for the updated Chroma class.  You
# need to install it with `pip install -U langchain-chroma`.
from langchain_chroma import Chroma  # type: ignore[import]

# Prefer the standalone langchain-ollama package for Ollama embeddings.  If
# langchain-ollama isn't installed, fall back to the (deprecated) community
# embedding class so that the app still works.  Users can install
# langchain-ollama via `pip install -U langchain-ollama`.
try:
    from langchain_ollama import OllamaEmbeddings  # type: ignore[import]
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings

# Use absolute paths for the database directory so that the app loads the index
# correctly regardless of the current working directory.  Base directory is the
# folder where this file resides.
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = str(BASE_DIR / "db")

COLLECTION_NAME = "som"

st.set_page_config(page_title="Society of Man Rules Assistant", page_icon="🛰️", layout="wide")
st.title("🛰️ Society of Man — Rules Assistant")

st.markdown("""
Welcome, Reaper.

This assistant is grounded in your **Society of Man** rulebook.
If it can't find an answer in your sources, it will say so directly.
""")

with st.sidebar:
    st.header("⚙️ Assistant Settings")
    top_k = st.slider("Number of results to retrieve", 1, 8, 4)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.2, 0.05)

def get_embeddings():
    """Return an embeddings instance matching ingestion.

    Prefer Ollama embeddings when available and fall back to OpenAI.
    """
    try:
        ef = OllamaEmbeddings(model="nomic-embed-text")
        # quick sanity check
        _ = ef.embed_query("test")
        return ef
    except Exception:
        return OpenAIEmbeddings()

# Initialize the persistent Chroma vector store.  Use the same collection
# name and embedding function as in ingest.py.  Disable anonymous telemetry
# at the chromadb client level to avoid OpenTelemetry errors.
try:
    embeddings_fn = get_embeddings()
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings_fn,
        collection_name=COLLECTION_NAME,
        client_settings=Settings(anonymized_telemetry=False),
    )
except Exception:
    st.warning("No index found. Run `python ingest.py` first to ingest your SoM docs.")
    st.stop()

def search(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Perform a similarity search against the Chroma vector store.

    This function embeds the query using the same embeddings used during
    indexing and returns up to k documents with their source metadata.
    """
    docs = vectordb.similarity_search(query, k)
    results: List[Dict[str, Any]] = []
    for doc in docs:
        results.append(
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "loc": doc.metadata.get("loc", ""),
            }
        )
    return results

query = st.text_input('Ask a question (e.g., "How do Yellow Color features work?")')
go = st.button("Search")

if go and query.strip():
    with st.spinner("Consulting the Codex..."):
        chunks = search(query, k=top_k)
        if not chunks:
            st.info("No matching text found. Add docs and rebuild the index.")
        else:
            answer = chat_with_context(query, chunks, temperature=temperature)
            st.markdown("### 📘 Answer")
            st.write(answer)

            st.markdown("### 📑 Sources from the Codex")
            for i, ch in enumerate(chunks, start=1):
                with st.expander(f"Chunk {i} — {ch['source']} {ch['loc']}"):
                    st.write(ch["text"])