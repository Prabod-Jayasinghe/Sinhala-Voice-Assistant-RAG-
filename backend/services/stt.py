"""
STT Service — Groq Whisper large-v3
=====================================
Wraps Groq's Whisper API for Sinhala audio transcription.

SDLC Note (Section 3, NFR):
    Sinhala is a confirmed low-resource language. Whisper's Sinhala WER is
    meaningfully higher than English. Phase 0 testing confirmed confidence
    scores typically 0.22–0.34 on clean TTS audio. This is documented, expected
    behavior — not a bug. The stt_confidence score is surfaced in the API
    response and UI for transparency (SDLC FR-6, Section 10 edge cases).
"""

import os
import tempfile
from pathlib import Path

from groq import AsyncGroq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
STT_CONFIDENCE_THRESHOLD = 0.4  # Below this → show "Did I hear that right?" in UI


class STTService:
    def __init__(self):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        self.client = AsyncGroq(api_key=GROQ_API_KEY)

    async def transcribe(self, audio_bytes: bytes, filename: str = "audio.webm") -> dict:
        """
        Transcribe Sinhala audio using Groq Whisper large-v3.

        Args:
            audio_bytes: Raw audio bytes from the browser (WebM/MP3/WAV)
            filename: Original filename (helps Whisper detect format)

        Returns:
            dict with:
                transcript (str)
                detected_language (str)
                duration_s (float | None)
                stt_confidence (float | None)  — proxy from avg_logprob, 0-1 scale
                low_confidence (bool)          — True if below STT_CONFIDENCE_THRESHOLD
                segments (list)                — raw Whisper segments
        """
        # Write to temp file (Groq SDK requires file-like with name)
        with tempfile.NamedTemporaryFile(
            suffix=Path(filename).suffix or ".webm", delete=False
        ) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                response = await self.client.audio.transcriptions.create(
                    file=(filename, f),
                    model="whisper-large-v3",
                    language="si",             # Force Sinhala — reduces hallucination to other languages
                    response_format="verbose_json",
                    temperature=0.0,           # Deterministic output
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Extract confidence proxy from segment avg_logprob
        segments = getattr(response, "segments", []) or []
        if segments:
            avg_logprob = sum(
                s.get("avg_logprob", -1.0) if isinstance(s, dict) else getattr(s, "avg_logprob", -1.0)
                for s in segments
            ) / len(segments)
            confidence = round(min(1.0, max(0.0, 1.0 + avg_logprob)), 3)
        else:
            avg_logprob = None
            confidence = None

        transcript = (response.text or "").strip()
        detected_language = getattr(response, "language", "unknown")
        duration_s = getattr(response, "duration", None)

        return {
            "transcript": transcript,
            "detected_language": detected_language,
            "duration_s": duration_s,
            "stt_confidence": confidence,
            "low_confidence": (confidence is not None and confidence < STT_CONFIDENCE_THRESHOLD),
            "segments": [
                (s if isinstance(s, dict) else s.model_dump()) for s in segments
            ],
        }


# Singleton
_stt_service: STTService | None = None


def get_stt_service() -> STTService:
    global _stt_service
    if _stt_service is None:
        _stt_service = STTService()
    return _stt_service
