"""
Text Query Router — POST /api/text-query
==========================================
Text input fallback (SDLC FR-8) — always available, even when mic fails.
Same RAG pipeline as voice-query, skipping the STT step.
"""

from fastapi import APIRouter

from models.schemas import SourceCitation, TextQueryRequest, TextQueryResponse
from services.generator import get_generator_service
from services.retriever import get_retriever_service
from services.tts import get_tts_service
from services.transliterate import get_transliteration_service

router = APIRouter()


@router.post("/text-query", response_model=TextQueryResponse)
async def text_query(request: TextQueryRequest):
    """
    Text fallback pipeline (FR-8):
    1. Transliterate Romanized Sinhala/Singlish or translate English to Sinhala script (FR-10)
    2. Retrieve relevant corpus chunks using the Sinhala script question
    3. Generate grounded Sinhala answer via Gemini Flash
    4. Synthesize answer to audio via edge-tts
    5. Return answer text, audio URL, source citations, and transliterated text
    """
    retriever = get_retriever_service()
    generator = get_generator_service()
    tts = get_tts_service()
    transliterate = get_transliteration_service()

    original_question = request.question.strip()
    query_text = original_question
    transliterated_question = None

    # FR-10: Romanized/Singlish to Sinhala script
    if translit_needed := transliterate.is_latin_script(original_question):
        transliterated_question = await transliterate.to_sinhala_script(original_question)
        query_text = transliterated_question

    # Offensive content check on both original and transliterated queries
    if generator.is_offensive(original_question) or (transliterated_question and generator.is_offensive(transliterated_question)):
        answer_text = "ඔබේ ප්‍රශ්නයට පිළිතුරු දීමට නොහැකි විය."
        answer_audio_url = await tts.synthesize(answer_text)
        return TextQueryResponse(
            answer_text=answer_text,
            answer_audio_url=answer_audio_url,
            sources=[],
            transliterated_question=transliterated_question,
        )

    retrieval_result = retriever.retrieve(query_text)
    answer_text = await generator.generate(
        question=query_text,
        retrieved_chunks=retrieval_result["chunks"],
        has_relevant_results=retrieval_result["has_relevant_results"],
    )
    answer_audio_url = await tts.synthesize(answer_text)

    sources = [
        SourceCitation(
            title=meta.get("title", "Unknown"),
            source=meta.get("source", "unknown"),
            published_date=meta.get("published_date") or None,
        )
        for meta in retrieval_result.get("sources", [])
    ]

    return TextQueryResponse(
        answer_text=answer_text,
        answer_audio_url=answer_audio_url,
        sources=sources,
        transliterated_question=transliterated_question,
    )
