import os
# Disable OpenTelemetry instrumentation globally before importing any Chroma or LangChain modules.
os.environ["OPENTELEMETRY_SDK_DISABLED"] = "true"

# Ensure an OpenAI API key is available for embedding and chat.  This
# script expects the OPENAI_API_KEY environment variable to be set
# externally (e.g., via your shell or deployment platform).  Do not
# commit API keys into source code.
import streamlit as st
from chromadb.config import Settings
# Additional import for cross-encoder reranking.  The sentence-transformers
# package provides a CrossEncoder model that jointly encodes a query and a
# document to produce a more accurate relevance score than simple vector
# similarity【442094950299547†L115-L153】.  Make sure `sentence-transformers`
# is installed (see requirements.txt) and note that CrossEncoder models
# require PyTorch as a backend.  Using a cross-encoder adds latency but
# greatly improves the quality of retrieved documents.
from sentence_transformers import CrossEncoder
from typing import List, Dict, Any
# Use the OpenAI-based chat function for local answers.  We import from
# rag_cloud to ensure the same chat mechanism across local and cloud environments.
from rag_cloud import chat_with_context_openai as chat_with_context

# Import the LangChain Chroma vector store and embeddings so that retrieval
# uses the same interface as ingestion.  This avoids mismatches between
# chromadb's low-level client and LangChain's wrapper and ensures we use
# the same embeddings as during indexing.
# Use the standalone langchain-chroma package for the updated Chroma class.  You
# need to install it with `pip install -U langchain-chroma`.
from langchain_chroma import Chroma  # type: ignore[import]

# Import only the OpenAI embeddings.  By relying solely on OpenAI for
# embeddings, we avoid any dependence on a local Ollama server and ensure
# consistent embedding dimensions between ingestion and retrieval.
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
    # Increase the maximum number of retrieved results to give the cross-encoder
    # more candidates to rank.  The default of 6 provides a good balance
    # between recall and latency.  Users can adjust this slider at runtime.
    top_k = st.slider("Number of results to retrieve", 1, 12, 6)
    temperature = st.slider("Response creativity", 0.0, 1.0, 0.2, 0.05)

def get_embeddings() -> OpenAIEmbeddings:
    """
    Return an OpenAI embeddings instance.  We always use the OpenAI embedding
    model for retrieval to avoid mismatches with ingestion.  The embedding
    model can be overridden via the OPENAI_EMBED_MODEL environment variable.
    """
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=embed_model)

# Initialize a cross-encoder reranker at module import time.  Using a
# cross-encoder allows the assistant to re-order the initially retrieved
# documents based on a joint encoding of the query and each document,
# improving relevance【442094950299547†L115-L153】.  This model runs on CPU by
# default; if GPU is available, sentence-transformers will use it.  If you
# wish to use a different cross-encoder, modify the model name here.
try:
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    # If the model fails to load, fall back to None.  Retrieval will still
    # function but without reranking.  We log the exception to stdout.
    cross_encoder = None
    print(f"[warn] Failed to load cross encoder: {e}")

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
    """
    Perform a two-stage retrieval against the Chroma vector store.

    1. Retrieve a larger set of candidate documents via vector similarity.
    2. If a cross-encoder is available, rank these candidates by
       relevance using a joint query/document encoding and return the top
       ``k`` documents.  Otherwise, return the top ``k`` documents based on
       vector similarity alone.
    """
    # Retrieve more documents than the user requested to give the re-ranker
    # a pool to select from.  We fetch up to 3×k documents but cap at
    # 50 to avoid excessive retrieval latency.
    n_candidates = min(k * 3, 50)
    raw_docs = vectordb.similarity_search(query, n_candidates)
    if not raw_docs:
        return []
    # If the cross encoder loaded successfully, use it to score each
    # candidate.  Otherwise, fall back to vector similarity order.
    if cross_encoder is not None:
        # Build (query, doc) pairs for scoring.  We pass the raw text
        # directly; the cross-encoder model handles tokenization.  The
        # predict method returns a relevance score for each pair.
        pairs = [[query, doc.page_content] for doc in raw_docs]
        scores = cross_encoder.predict(pairs)
        # Sort candidates by score, descending.  Zip docs and scores
        # together so we preserve both when sorting.  We then take the
        # top k documents.
        scored = sorted(zip(raw_docs, scores), key=lambda x: x[1], reverse=True)
        selected_docs = [doc for doc, _ in scored[:k]]
    else:
        # No cross-encoder available: just take the top k by vector similarity.
        selected_docs = raw_docs[:k]
    # Format results for display.  Each result includes the chunk text and
    # metadata about its source file and location.  This metadata was
    # stored during ingestion via the DocumentLoader and TextSplitter.
    results: List[Dict[str, Any]] = []
    for doc in selected_docs:
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