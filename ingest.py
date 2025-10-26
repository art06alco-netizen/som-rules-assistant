import os
import glob
# Disable OpenTelemetry instrumentation to suppress noisy telemetry events.
os.environ["OPENTELEMETRY_SDK_DISABLED"] = "true"

# Set a default OpenAI API key for embedding generation.  This will be used
# whenever the OPENAI_API_KEY environment variable is not already defined.
# NOTE: Do not expose this key publicly or commit it to version control in
# real-world projects.  It is included here solely so that ingestion can
# proceed without requiring external configuration.
import time
from typing import List
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use the standalone langchain-chroma package for the updated Chroma class.  It
# provides a more stable implementation than the deprecated version in
# langchain_community.  You need to install it with `pip install -U langchain-chroma`.
from langchain_chroma import Chroma  # type: ignore[import]
from chromadb.config import Settings
# Import only the OpenAI embeddings.  By relying solely on OpenAI for
# embedding generation we eliminate the need for a running Ollama server
# and avoid mixing different embedding dimensions.  The embedding model
# can be overridden via the OPENAI_EMBED_MODEL environment variable.
from langchain_community.embeddings import OpenAIEmbeddings

# === CONFIG ===
# Resolve paths relative to this script's location so that the DB and docs folder
# are created and found consistently regardless of the current working directory.
from pathlib import Path

# Base directory of the repository (location of this file)
BASE_DIR = Path(__file__).resolve().parent

# Directory to persist the Chroma database.  We convert to str for compatibility.
DB_DIR = str(BASE_DIR / "db")

COLLECTION_NAME = "som"

# Glob pattern for DOCX files.  We look inside the `docs/` folder relative to
# this file.  The double asterisk allows recursive search in subfolders.
DOCS_PATH = str(BASE_DIR / "docs" / "**" / "*.docx")

CHUNK_SIZE = 700
CHUNK_OVERLAP = 300

# === LOAD DOCS ===
def load_docs() -> List:
    print("[load] Scanning docs for source files...")
    files = glob.glob(DOCS_PATH, recursive=True)
    if not files:
        raise FileNotFoundError("No DOCX files found in docs/")
    docs = []
    for file in files:
        print(f"[load] Loading {file}...")
        loader = Docx2txtLoader(file)
        docs.extend(loader.load())
    print(f"[load] Loaded {len(docs)} documents.")
    return docs

# === SPLIT TEXT ===
def split_docs(docs):
    print("[chunk] Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"[chunk] Prepared {len(chunks)} chunk(s).")
    return chunks

# === SETUP EMBEDDINGS ===
def get_embeddings():
    """
    Create and return an OpenAI embeddings object.  We rely solely on the
    OpenAI embedding API so that the vector dimensionality is consistent
    across ingestion and retrieval.  The default model can be overridden
    via the OPENAI_EMBED_MODEL environment variable.

    Raises:
        RuntimeError: if the OPENAI_API_KEY environment variable is not set.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY must be set to generate embeddings.")
    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=embed_model)

# === BUILD DB ===
def build_db(chunks):
    print("[db] Connecting to Chroma at 'db' (persistent)...")
    embeddings = get_embeddings()
    # Disable anonymous telemetry when initializing the Chroma client.  Without
    # this setting Chroma attempts to send telemetry events via OpenTelemetry,
    # which can cause runtime errors in some environments.
    client_settings = Settings(anonymized_telemetry=False)
    # Create the persistent vector store from the provided chunks.  Using
    # Chroma.from_documents both adds the documents and persists them to the
    # specified directory; you don't need to call add_documents() or persist()
    # manually.
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
        client_settings=client_settings,
    )
    print(f"[db] ✅ Indexed {len(chunks)} chunks into {DB_DIR}/")
    return db

# === MAIN ===
def main():
    start = time.time()
    docs = load_docs()
    chunks = split_docs(docs)
    build_db(chunks)
    print(f"\n[done] All complete in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()