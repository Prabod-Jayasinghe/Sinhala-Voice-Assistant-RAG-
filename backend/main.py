"""
Sinhala Voice Assistant — FastAPI Backend
==========================================
Provides voice and text Q&A endpoints using:
  - Groq Whisper large-v3 for STT
  - multilingual-e5-large + ChromaDB for RAG retrieval
  - Google Gemini Flash for Sinhala answer generation
  - edge-tts for Sinhala voice synthesis (no API key needed)
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure audio output directory exists at import time
os.makedirs("audio_output", exist_ok=True)

load_dotenv()

from routers import voice_query, text_query, corpus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    import asyncio
    from services.retriever import get_retriever_service
    from services.corpus_ingest import ingest_corpus, scheduled_refresh_loop

    retriever = get_retriever_service()

    async def bootstrap_if_empty():
        if retriever.collection.count() == 0:
            print("[Lifespan] ChromaDB is empty. Running initial corpus ingestion...")
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, ingest_corpus, retriever)
                print("[Lifespan] Initial corpus ingestion completed.")
            except Exception as e:
                print(f"[Lifespan] Failed to run initial corpus ingestion: {e}")
        else:
            print(f"[Lifespan] ChromaDB has {retriever.collection.count()} chunks. Skipping bootstrap.")

    # Start bootstrap & scheduled refresh in background (FR-9)
    asyncio.create_task(bootstrap_if_empty())
    asyncio.create_task(scheduled_refresh_loop(retriever))
    
    yield


app = FastAPI(
    title="Sinhala Voice Assistant API",
    description=(
        "RAG-powered voice assistant for Sinhala language. "
        "Supports voice and text input, returns grounded Sinhala answers with source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Next.js frontend (update origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve synthesized audio files
app.mount("/audio", StaticFiles(directory="audio_output"), name="audio")

# Routers
app.include_router(voice_query.router, prefix="/api")
app.include_router(text_query.router, prefix="/api")
app.include_router(corpus.router, prefix="/api/corpus")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "sinhala-voice-assistant"}
