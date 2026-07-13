"""Pydantic data models matching SDLC Section 6 data model."""

from datetime import date, datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Documents & Chunks (corpus)
# ---------------------------------------------------------------------------

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    source: str                         # e.g. "NSINA", "sinhala_wikipedia"
    title: str
    raw_text: str
    published_date: Optional[date] = None
    source_type: str                    # "news" | "encyclopedia"


class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    chunk_text: str
    chunk_index: int


class SourceCitation(BaseModel):
    """Returned to frontend with every answer."""
    title: str
    source: str                         # "NSINA" | "sinhala_wikipedia"
    published_date: Optional[str] = None


# ---------------------------------------------------------------------------
# Sessions & Turns
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    question_transcript: str
    answer_text: str
    question_audio_url: Optional[str] = None
    answer_audio_url: Optional[str] = None
    stt_confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# API Request / Response schemas
# ---------------------------------------------------------------------------

class TextQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class VoiceQueryResponse(BaseModel):
    """
    Response from POST /api/voice-query
    Matches SDLC Section 7 API design exactly.
    """
    transcript: str
    answer_text: str
    answer_audio_url: str
    sources: list[SourceCitation]
    stt_confidence: Optional[float] = None
    low_confidence_warning: bool = False  # True if stt_confidence < 0.4


class TextQueryResponse(BaseModel):
    """Response from POST /api/text-query"""
    answer_text: str
    answer_audio_url: str
    sources: list[SourceCitation]
    transliterated_question: Optional[str] = None


class CorpusStatusResponse(BaseModel):
    last_refreshed: Optional[datetime]
    document_count: int
    chunk_count: int


class CorpusRefreshResponse(BaseModel):
    ingested_count: int
    chunk_count: int
    message: str
