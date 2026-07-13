"""
Tests for the text-query and voice-query API endpoints.
Uses httpx async test client with mocked services to test API contracts.
"""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Sync test client for simple tests."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Text query tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_text_query_returns_answer():
    """POST /api/text-query should return answer_text, audio URL, and sources."""
    mock_retrieval = {
        "chunks": ["ශ්‍රී ලංකාවේ ජනාධිපති රනිල් වික්‍රමසිංහ."],
        "sources": [{"title": "News article", "source": "NSINA", "published_date": "2024-01-01"}],
        "distances": [0.2],
        "has_relevant_results": True,
        "corpus_empty": False,
    }

    with (
        patch("routers.text_query.get_retriever_service") as mock_ret,
        patch("routers.text_query.get_generator_service") as mock_gen,
        patch("routers.text_query.get_tts_service") as mock_tts,
    ):
        mock_ret.return_value.retrieve.return_value = mock_retrieval
        mock_gen.return_value.generate = AsyncMock(return_value="ශ්‍රී ලංකාවේ ජනාධිපතිය රනිල් වික්‍රමසිංහ.")
        mock_gen.return_value.is_offensive.return_value = False
        mock_tts.return_value.synthesize = AsyncMock(return_value="/audio/answer_test.mp3")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/text-query",
                json={"question": "ශ්‍රී ලංකාවේ ජනාධිපති කවුද?"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert "answer_text" in data
    assert "answer_audio_url" in data
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_text_query_empty_question_rejected():
    """Empty question should return 422 validation error."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/text-query", json={"question": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_text_query_no_corpus_results():
    """When retrieval returns no relevant chunks, should return no-info response."""
    mock_retrieval = {
        "chunks": [],
        "sources": [],
        "distances": [],
        "has_relevant_results": False,
        "corpus_empty": False,
    }

    with (
        patch("routers.text_query.get_retriever_service") as mock_ret,
        patch("routers.text_query.get_generator_service") as mock_gen,
        patch("routers.text_query.get_tts_service") as mock_tts,
    ):
        mock_ret.return_value.retrieve.return_value = mock_retrieval
        mock_gen.return_value.generate = AsyncMock(
            return_value="සිංහල දෙනෝ, මා සතු දැනුම් පදනමෙහි ඔබේ ප්‍රශ්නයට අදාළ තොරතුරු නොමැත."
        )
        mock_gen.return_value.is_offensive.return_value = False
        mock_tts.return_value.synthesize = AsyncMock(return_value="/audio/no_info.mp3")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/text-query",
                json={"question": "What is quantum mechanics?"},  # Totally off-topic for Sinhala corpus
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["sources"] == []  # No sources when no relevant chunks


# ---------------------------------------------------------------------------
# Corpus status tests
# ---------------------------------------------------------------------------

def test_corpus_status_endpoint(client):
    """GET /api/corpus/status should return status fields."""
    with (
        patch("routers.corpus.get_retriever_service") as mock_ret,
        patch("routers.corpus.corpus_ingest.get_corpus_stats") as mock_stats,
    ):
        mock_stats.return_value = {"last_refreshed": None, "document_count": 0}
        mock_ret.return_value.get_corpus_stats.return_value = {"chunk_count": 0}

        resp = client.get("/api/corpus/status")

    assert resp.status_code == 200
    data = resp.json()
    assert "document_count" in data
    assert "chunk_count" in data


@pytest.mark.asyncio
async def test_text_query_transliterates_singlish():
    """POST /api/text-query with romanized text should transliterate to Sinhala Unicode."""
    mock_retrieval = {
        "chunks": ["ශ්‍රී ලංකාවේ ජනාධිපති රනිල් වික්‍රමසිංහ."],
        "sources": [{"title": "News article", "source": "NSINA"}],
        "distances": [0.2],
        "has_relevant_results": True,
        "corpus_empty": False,
    }

    with (
        patch("routers.text_query.get_transliteration_service") as mock_trans,
        patch("routers.text_query.get_retriever_service") as mock_ret,
        patch("routers.text_query.get_generator_service") as mock_gen,
        patch("routers.text_query.get_tts_service") as mock_tts,
    ):
        # Setup mocks
        mock_trans.return_value.is_latin_script.return_value = True
        mock_trans.return_value.to_sinhala_script = AsyncMock(return_value="ශ්‍රී ලංකාවේ ජනාධිපති කවුද?")
        
        mock_ret.return_value.retrieve.return_value = mock_retrieval
        mock_gen.return_value.generate = AsyncMock(return_value="ශ්‍රී ලංකාවේ ජනාධිපති රනිල් වික්‍රමසිංහ.")
        mock_gen.return_value.is_offensive.return_value = False
        mock_tts.return_value.synthesize = AsyncMock(return_value="/audio/answer_test.mp3")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/text-query",
                json={"question": "lankave janadhipathi kavuda"},  # Singlish
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["transliterated_question"] == "ශ්‍රී ලංකාවේ ජනාධිපති කවුද?"
    assert data["answer_text"] == "ශ්‍රී ලංකාවේ ජනාධිපති රනිල් වික්‍රමසිංහ."
    mock_trans.return_value.to_sinhala_script.assert_called_once_with("lankave janadhipathi kavuda")


def test_is_offensive_heuristics():
    """Verify that is_offensive correctly flags bad words and allows clean inputs."""
    from services.generator import GeneratorService
    
    # 1. Clean inputs (should return False)
    assert GeneratorService.is_offensive("ශ්‍රී ලංකාව ඉතාම ලස්සන රටක්") is False
    assert GeneratorService.is_offensive("What is the capital of Sri Lanka?") is False
    assert GeneratorService.is_offensive("lankave janadhipathi kavuda?") is False
    assert GeneratorService.is_offensive("උපකාර සහ සහයෝගය") is False  # Contains "පකාර" which shouldn't false positive

    # 2. Sinhala bad words (should return True)
    assert GeneratorService.is_offensive("ඒක පකයෙක්") is True
    assert GeneratorService.is_offensive("හුත්තා") is True
    assert GeneratorService.is_offensive("කැරියා") is True
    assert GeneratorService.is_offensive("වේසි") is True
    assert GeneratorService.is_offensive("පොන්නයා") is True

    # 3. English bad words (should return True)
    assert GeneratorService.is_offensive("shut up you bitch") is True
    assert GeneratorService.is_offensive("fuck this") is True

    # 4. Singlish bad words (should return True)
    assert GeneratorService.is_offensive("uba paka") is True
    assert GeneratorService.is_offensive("patta ponna") is True


