"""
Corpus Router — GET/POST /api/corpus/...
==========================================
Corpus management endpoints.

SDLC Section 7:
    GET  /api/corpus/status  — when was corpus last refreshed + doc count
    POST /api/corpus/refresh — manually trigger corpus re-ingest (admin/dev)
"""

from fastapi import APIRouter, BackgroundTasks

from models.schemas import CorpusRefreshResponse, CorpusStatusResponse
from services import corpus_ingest
from services.retriever import get_retriever_service

router = APIRouter()


@router.get("/status", response_model=CorpusStatusResponse)
async def corpus_status():
    """Return corpus freshness and size metrics."""
    stats = corpus_ingest.get_corpus_stats()
    retriever = get_retriever_service()
    chroma_stats = retriever.get_corpus_stats()

    return CorpusStatusResponse(
        last_refreshed=stats["last_refreshed"],
        document_count=stats["document_count"],
        chunk_count=chroma_stats["chunk_count"],
    )


@router.post("/refresh", response_model=CorpusRefreshResponse)
async def refresh_corpus(background_tasks: BackgroundTasks):
    """
    Trigger a corpus re-ingest from HuggingFace datasets.
    Runs in the background to avoid blocking the HTTP response.
    Call GET /api/corpus/status to check progress.
    """
    retriever = get_retriever_service()
    result = corpus_ingest.ingest_corpus(retriever)
    return CorpusRefreshResponse(**result)
