"""
Transliteration & Translation Service
======================================
Handles Romanized Sinhala (Singlish) and English text query inputs,
converting them to proper Sinhala Unicode script before embedding
and RAG retrieval (SDLC FR-10).

Utilizes Google Gemini Flash for contextual phonetic transliteration
and translation, which is highly accurate for low-resource languages.
"""

import os
import re
from typing import Optional
from google.generativeai import GenerativeModel
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"


class TransliterationService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = GenerativeModel(model_name=GEMINI_MODEL)

    def is_latin_script(self, text: str) -> bool:
        """
        Check if the input text is predominantly Latin script (Singlish/English).
        If > 50% of the characters are Latin letters, returns True.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        total_letters = sum(1 for c in cleaned if c.isalpha())
        if total_letters == 0:
            return False

        latin_letters = sum(1 for c in cleaned if c.isascii() and c.isalpha())
        return (latin_letters / total_letters) > 0.5

    async def to_sinhala_script(self, text: str) -> str:
        """
        Convert Singlish (romanized Sinhala) or English query text into proper
        Sinhala Unicode script using Gemini Flash.

        Example:
            "subha udasanak" -> "සුබ උදෑසනක්"
            "lankave janadhipathi kavuda" -> "ලංකාවේ ජනාධිපති කවුද?"
            "who is the president of sri lanka" -> "ශ්‍රී ලංකාවේ ජනාධිපති කවුද?"
        """
        if not self.is_latin_script(text):
            return text.strip()

        prompt = f"""You are a Sinhala language expert translator and transliterator.
Convert the following input query into natural, grammatically correct Sinhala Unicode script.

Instructions:
1. If the input is Romanized Sinhala (Singlish) (e.g. "lankave janadhipathi kavuda"), phonetically transliterate it to proper Sinhala script ("ලංකාවේ ජනාධිපති කවුද?").
2. If the input is English (e.g. "Who is the president of Sri Lanka?"), translate it to natural Sinhala script ("ශ්‍රී ලංකාවේ ජනාධිපති කවුද?").
3. Preserve the original meaning and punctuation.
4. Output ONLY the resulting Sinhala Unicode script text. Do not include any explanation or extra text.

Input Query: {text}

Sinhala Script:"""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=256,
                ),
            )
            result = response.text.strip() if response.text else text
            print(f"[Transliterate] Input: '{text}' -> Output: '{result}'")
            return result
        except Exception as e:
            print(f"[Transliterate] Error during transliteration: {e}")
            return text


# Singleton
_transliteration_service: Optional[TransliterationService] = None


def get_transliteration_service() -> TransliterationService:
    global _transliteration_service
    if _transliteration_service is None:
        _transliteration_service = TransliterationService()
    return _transliteration_service
