"""
Voice Query Router — POST /api/voice-query
===========================================
Full pipeline: audio blob → STT → RAG retrieval → Gemini generation → TTS → response

SDLC Section 7 API contract:
    Request:  multipart audio file
    Response: {transcript, answer_text, answer_audio_url, sources[], stt_confidence}
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from models.schemas import SourceCitation, VoiceQueryResponse
from services.generator import get_generator_service
from services.retriever import get_retriever_service
from services.stt import get_stt_service
from services.tts import get_tts_service

router = APIRouter()

# Max audio file size: 25MB (Groq Whisper limit)
MAX_AUDIO_SIZE_BYTES = 25 * 1024 * 1024


@router.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(audio: UploadFile = File(...)):
    """
    Full voice pipeline:
    1. Transcribe audio via Groq Whisper (Sinhala STT)
    2. Embed transcript + retrieve relevant corpus chunks
    3. Generate grounded Sinhala answer via Gemini Flash
    4. Synthesize answer to audio via edge-tts
    5. Return transcript, answer, audio URL, sources, and STT confidence

    SDLC Section 10 edge cases handled:
    - Low-confidence transcription: flagged in response, UI shows warning
    - No relevant chunks: honest "no info" response, no hallucination
    - Offensive input: basic heuristic filter before generation
    """
    # Read audio bytes
    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio file too large. Maximum size is 25MB.",
        )
    if len(audio_bytes) < 100:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty or too short.",
        )

    stt = get_stt_service()
    retriever = get_retriever_service()
    generator = get_generator_service()
    tts = get_tts_service()

    # --- Step 1: STT ---
    stt_result = await stt.transcribe(audio_bytes, filename=audio.filename or "audio.webm")
    transcript = stt_result["transcript"]

    if not transcript or len(transcript.strip()) < 2:
        raise HTTPException(
            status_code=422,
            detail="Could not transcribe audio. Please speak clearly in Sinhala or use text input.",
        )

    # --- Offensive content check (before retrieval + generation) ---
    if generator.is_offensive(transcript):
        answer_text = "ඔබේ ප්‍රශ්නයට පිළිතුරු දීමට නොහැකි විය."
        answer_audio_url = await tts.synthesize(answer_text)
        return VoiceQueryResponse(
            transcript=transcript,
            answer_text=answer_text,
            answer_audio_url=answer_audio_url,
            sources=[],
            stt_confidence=stt_result.get("stt_confidence"),
            low_confidence_warning=stt_result.get("low_confidence", False),
        )

    # --- Step 2: Retrieval ---
    retrieval_result = retriever.retrieve(transcript)

    # --- Step 3: Generation ---
    answer_text = await generator.generate(
        question=transcript,
        retrieved_chunks=retrieval_result["chunks"],
        has_relevant_results=retrieval_result["has_relevant_results"],
    )

    # --- Step 4: TTS ---
    answer_audio_url = await tts.synthesize(answer_text)

    # --- Build source citations ---
    sources = []
    for meta in retrieval_result.get("sources", []):
        sources.append(SourceCitation(
            title=meta.get("title", "Unknown"),
            source=meta.get("source", "unknown"),
            published_date=meta.get("published_date") or None,
        ))

    return VoiceQueryResponse(
        transcript=transcript,
        answer_text=answer_text,
        answer_audio_url=answer_audio_url,
        sources=sources,
        stt_confidence=stt_result.get("stt_confidence"),
        low_confidence_warning=stt_result.get("low_confidence", False),
    )
