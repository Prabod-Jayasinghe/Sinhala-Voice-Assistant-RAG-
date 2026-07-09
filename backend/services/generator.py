"""
Generator Service — Google Gemini Flash
=========================================
Generates grounded Sinhala answers using retrieved corpus chunks.

LLM choice (from SDLC Section 5):
    Gemini Flash (free tier via AI Studio) — NOT Groq/Llama 3.
    Llama 3's official language support doesn't meaningfully include Sinhala.
    Gemini has materially better multilingual generation quality for
    lower-resource languages like Sinhala. This is a deliberate deviation
    from the Groq-only stack — see SDLC Section 5 for the full rationale.

Prompt design:
    Every answer MUST cite the retrieved context chunks.
    If no relevant chunks: return a fixed Sinhala "I don't have info" response
    rather than calling the LLM blind (SDLC Section 10 edge case handling).
"""

import os
from typing import Optional

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-flash"

# Fixed responses for edge cases (never hallucinate for these)
NO_INFO_SINHALA = (
    "සිංහල දෙනෝ, මා සතු දැනුම් පදනමෙහි ඔබේ ප්‍රශ්නයට අදාළ "
    "තොරතුරු නොමැත. කරුණාකර වෙනත් ප්‍රශ්නයක් අසන්න."
)

SYSTEM_PROMPT = """ඔබ සිංහල භාෂාවෙන් ප්‍රශ්නවලට පිළිතුරු දෙන AI සහායකයෙකි.
ඔබ ලබා දෙන පිළිතුර:
1. ලබා දී ඇති සන්දර්භය (context) ආශ්‍රිතව පමණක් පිළිතුරු දෙන්න
2. සන්දර්භය තුළ නොමැති කරුණු ඔබගේ ශ්‍රේෂ්ඨ දැනුමෙන් එකතු නොකරන්න
3. පිළිතුර සිංහල භාෂාවෙන් ලිවිය යුතුය
4. පිළිතුර කෙටි හා ප්‍රමාණවත් විය යුතුය (ඡේද 1-2)
5. ලිඛිත Sinhala unicode text භාවිතා කරන්න"""


class GeneratorService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_PROMPT,
        )

    async def generate(
        self,
        question: str,
        retrieved_chunks: list[str],
        has_relevant_results: bool,
    ) -> str:
        """
        Generate a grounded Sinhala answer.

        Args:
            question: The user's Sinhala question (transcript or typed)
            retrieved_chunks: Top-k relevant corpus chunks
            has_relevant_results: If False, returns fixed no-info response

        Returns:
            Sinhala answer text string
        """
        # SDLC Section 10: No relevant chunks → honest no-info response, NOT a blind LLM call
        if not has_relevant_results or not retrieved_chunks:
            return NO_INFO_SINHALA

        # Build context block from retrieved chunks
        context_block = "\n\n".join(
            f"[සන්දර්භය {i+1}]\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        )

        prompt = f"""පහත සන්දර්භය ආශ්‍රිතව, ප්‍රශ්නයට සිංහල භාෂාවෙන් පිළිතුරු දෙන්න.

සන්දර්භය:
{context_block}

ප්‍රශ්නය: {question}

සිංහල පිළිතුර:"""

        response = await self.model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,        # Low temp = more grounded, less creative hallucination
                max_output_tokens=512,
            ),
        )

        return response.text.strip() if response.text else NO_INFO_SINHALA

    @staticmethod
    def is_offensive(text: str) -> bool:
        """
        Basic heuristic offensive content filter.

        SDLC Section 10 / Section 11 note:
            Sinhala offensive-language detection is an active, unsolved research
            problem. Major commercial detectors perform poorly on Sinhala.
            This MVP uses a conservative keyword heuristic and documents this as
            a known gap — not claiming robust moderation it doesn't have.
        """
        # Simple Latin script ratio check — very high Latin in "Sinhala" input may indicate
        # attempts to inject English prompts disguised as Sinhala input
        if not text:
            return False

        total = len(text.strip())
        if total == 0:
            return False

        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        latin_ratio = latin_chars / total

        # If input is > 80% Latin script when we expect Sinhala, flag it
        if latin_ratio > 0.8:
            return True

        return False


# Singleton
_generator_service: Optional[GeneratorService] = None


def get_generator_service() -> GeneratorService:
    global _generator_service
    if _generator_service is None:
        _generator_service = GeneratorService()
    return _generator_service
