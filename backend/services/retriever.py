"""
Retrieval Service — multilingual-e5-large + ChromaDB
======================================================
Embeds Sinhala queries and retrieves relevant chunks from the
ingested Sinhala corpus (NSINA news + Sinhala Wikipedia).

Embedding model choice (from SDLC Section 5):
    intfloat/multilingual-e5-large
    No free, production-ready Sinhala-only embedding model exists yet.
    A good multilingual model is the practical choice today.
    Revisit if a dedicated Sinhala embedding model matures.

ChromaDB: embedded file-based, free, no server needed.
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
COLLECTION_NAME = "sinhala_corpus"
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# multilingual-e5 requires the "query:" / "passage:" prefix
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class RetrieverService:
    """
    Embeds queries with multilingual-e5-large and retrieves
    the top-k most relevant Sinhala corpus chunks from ChromaDB.
    """

    def __init__(self):
        Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[Retriever] Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[Retriever] Model loaded. Collection has {self.collection.count()} chunks.")

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string with the multilingual-e5 query prefix."""
        prefixed = QUERY_PREFIX + query
        return self.embedder.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        """Embed corpus passages with the multilingual-e5 passage prefix."""
        prefixed = [PASSAGE_PREFIX + p for p in passages]
        return self.embedder.encode(prefixed, normalize_embeddings=True).tolist()

    def retrieve(self, query: str, top_k: int = TOP_K) -> dict:
        """
        Retrieve top-k relevant chunks for a Sinhala query.

        Returns:
            dict with:
                chunks (list[str])         — retrieved chunk texts
                sources (list[dict])       — source metadata for each chunk
                distances (list[float])    — cosine distances (lower = more similar)
                has_relevant_results (bool) — False if all results below threshold
        """
        if self.collection.count() == 0:
            return {
                "chunks": [],
                "sources": [],
                "distances": [],
                "has_relevant_results": False,
                "corpus_empty": True,
            }

        query_embedding = self.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        # Filter by similarity threshold (cosine distance: 0 = identical, 2 = opposite)
        # With normalized embeddings, distance > (1 - SIMILARITY_THRESHOLD) means low similarity
        relevance_cutoff = 1.0 - SIMILARITY_THRESHOLD
        relevant_indices = [i for i, d in enumerate(distances) if d <= relevance_cutoff]

        if not relevant_indices:
            return {
                "chunks": [],
                "sources": [],
                "distances": distances,
                "has_relevant_results": False,
                "corpus_empty": False,
            }

        relevant_chunks = [chunks[i] for i in relevant_indices]
        relevant_sources = [metadatas[i] for i in relevant_indices]
        relevant_distances = [distances[i] for i in relevant_indices]

        return {
            "chunks": relevant_chunks,
            "sources": relevant_sources,
            "distances": relevant_distances,
            "has_relevant_results": True,
            "corpus_empty": False,
        }

    def get_corpus_stats(self) -> dict:
        """Return basic stats about the current corpus."""
        count = self.collection.count()
        return {"chunk_count": count}


# Singleton (loaded once at startup — expensive to reload the embedding model)
_retriever_service: Optional[RetrieverService] = None


def get_retriever_service() -> RetrieverService:
    global _retriever_service
    if _retriever_service is None:
        _retriever_service = RetrieverService()
    return _retriever_service
