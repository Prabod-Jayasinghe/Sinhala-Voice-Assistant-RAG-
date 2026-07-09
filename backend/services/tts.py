"""
TTS Service — edge-tts (Microsoft Edge Neural TTS)
====================================================
Provides Sinhala text-to-speech using edge-tts, which wraps
Microsoft's free Edge browser TTS service.

Available Sinhala voices (confirmed in Phase 0 feasibility check):
  - si-LK-ThiliniNeural  (Female) — default
  - si-LK-SameeraNeural  (Male)

IMPORTANT — SDLC README caveat:
    edge-tts is an unofficial open-source wrapper around a Microsoft
    service not meant for third-party programmatic use. It is free and
    requires no API key, but may be rate-limited or changed by Microsoft
    without notice. This is appropriate for a portfolio demo; not suitable
    for production use without migrating to an official TTS API.
"""

import os
import uuid
from pathlib import Path

import edge_tts

TTS_VOICE = os.getenv("TTS_VOICE", "si-LK-ThiliniNeural")
AUDIO_OUTPUT_DIR = Path("audio_output")

SINHALA_VOICES = {
    "female": "si-LK-ThiliniNeural",
    "male": "si-LK-SameeraNeural",
}

# No relevant source chunks found — return this pre-built Sinhala response
NO_INFO_RESPONSE = "මට ඔබේ ප්‍රශ්නයට අදාළ තොරතුරු ලබා දීමට නොහැකි විය. කරුණාකර වෙනත් ප්‍රශ්නයක් අසන්න."
OFFENSIVE_RESPONSE = "ඔබේ ප්‍රශ්නයට පිළිතුරු දීමට නොහැකි විය."


class TTSService:
    def __init__(self, voice: str = TTS_VOICE):
        self.voice = voice
        AUDIO_OUTPUT_DIR.mkdir(exist_ok=True)

    async def synthesize(self, text: str, filename: str | None = None) -> str:
        """
        Convert Sinhala text to MP3 audio using edge-tts.

        Args:
            text: Sinhala text to speak
            filename: Optional specific filename (without path). If None, generates UUID.

        Returns:
            Relative URL path to the audio file, e.g. "/audio/answer_abc123.mp3"
        """
        if not filename:
            filename = f"answer_{uuid.uuid4().hex[:12]}.mp3"

        output_path = AUDIO_OUTPUT_DIR / filename

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))

        return f"/audio/{filename}"

    async def synthesize_no_info(self) -> str:
        """Pre-built response for when no relevant corpus chunks are found."""
        return await self.synthesize(NO_INFO_RESPONSE, "no_info_response.mp3")

    async def synthesize_error(self, message: str | None = None) -> str:
        """Pre-built error response."""
        text = message or "සේවාදායකයේ දෝෂයක් ඇති විය. කරුණාකර නැවත උත්සාහ කරන්න."
        return await self.synthesize(text, "error_response.mp3")


# Singleton
_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(voice=TTS_VOICE)
    return _tts_service
