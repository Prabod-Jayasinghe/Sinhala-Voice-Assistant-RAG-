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

load_dotenv()

from routers import voice_query, text_query, corpus


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    # Ensure audio output directory exists
    os.makedirs("audio_output", exist_ok=True)
    yield
    # Cleanup on shutdown (nothing needed yet)


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
