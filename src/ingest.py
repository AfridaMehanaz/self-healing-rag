"""Ingest documents -> chunk -> embed -> store in ChromaDB.

Usage:
    python src/ingest.py docs/
"""
import os
import sys
import glob

DB_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = "knowledge_base"
CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150


def load_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    """Simple overlapping character chunks — easy and effective to start."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]


def get_collection():
    """Lazy imports keep unit tests fast (no chromadb/torch needed to test logic)."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # small, free, CPU-friendly
    )
    return client.get_or_create_collection(COLLECTION, embedding_function=embed_fn)


def ingest(folder: str):
    col = get_collection()
    files = []
    for ext in ("*.txt", "*.md", "*.pdf"):
        files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    if not files:
        print(f"No documents found in {folder}")
        return
    for path in files:
        chunks = chunk_text(load_text(path))
        col.add(
            ids=[f"{os.path.basename(path)}-{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": path} for _ in chunks],
        )
        print(f"Ingested {path}: {len(chunks)} chunks")
    print(f"\nDone -> collection '{COLLECTION}' at {DB_PATH}")


if __name__ == "__main__":
    ingest(sys.argv[1] if len(sys.argv) > 1 else "./docs")
