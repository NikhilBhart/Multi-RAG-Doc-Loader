"""
Local vector store, backed by Chroma. Embeddings run locally via
sentence-transformers (all-MiniLM-L6-v2) so no embedding API key is required.
"""
import chromadb
from chromadb.utils import embedding_functions

EMBED_MODEL = "all-MiniLM-L6-v2"


def get_collection(persist_dir: str = "./chroma_db", collection_name: str = "docs"):
    """Returns a Chroma collection, creating it (and the on-disk store) if needed."""
    client = chromadb.PersistentClient(path=persist_dir)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return client.get_or_create_collection(name=collection_name, embedding_function=embedder)


def add_chunks(collection, chunks: list[str], metadatas: list[dict], ids: list[str]):
    """Embeds and stores a batch of chunks. Chroma handles the embedding call internally."""
    collection.add(documents=chunks, metadatas=metadatas, ids=ids)


def similarity_search(collection, query: str, top_k: int = 6):
    """Returns a list of {text, metadata, score} dicts, closest first."""
    results = collection.query(query_texts=[query], n_results=top_k)
    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        # Chroma returns a distance (lower = closer); convert to a 0-1 similarity for display
        similarity = 1 / (1 + dist)
        hits.append({"text": doc, "metadata": meta, "score": round(similarity, 3)})
    return hits
