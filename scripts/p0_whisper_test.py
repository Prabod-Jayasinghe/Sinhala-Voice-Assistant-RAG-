# -*- coding: utf-8 -*-
"""
Phase 0 — Whisper Feasibility Check
=====================================
Tests Groq Whisper large-v3 Sinhala transcription quality
and edge-tts Sinhala voice synthesis.

Usage:
    # Test TTS first (no key needed):
    python scripts/p0_whisper_test.py --tts-only

    # Test STT with a local audio file:
    python scripts/p0_whisper_test.py --audio path/to/sinhala_sample.mp3

    # Test STT + TTS round trip (TTS → save audio → STT → compare):
    python scripts/p0_whisper_test.py --round-trip

    # List available Sinhala TTS voices:
    python scripts/p0_whisper_test.py --list-voices

IMPORTANT — from SDLC Section 14, Phase 0:
    This is the single riskiest assumption in the whole project.
    Whisper's Sinhala word-error-rate is meaningfully higher than English.
    Run this before building anything else. Document the results in
    PHASE0_FINDINGS.md — especially if quality is rough, because that
    changes the demo strategy (curated sample questions instead of
    live open-mic) before the rest of the architecture is committed.
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (Sinhala Unicode chars need it)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SINHALA_VOICES = ["si-LK-ThiliniNeural", "si-LK-SameeraNeural"]
DEFAULT_VOICE = "si-LK-ThiliniNeural"
SCRIPTS_DIR = Path(__file__).parent
AUDIO_OUT_DIR = SCRIPTS_DIR / "test_audio_output"

# Sample Sinhala phrases to use when no audio file is provided
SINHALA_SAMPLES = [
    "ආයුබෝවන්, ඔබට කොහොමද?",                          # Hello, how are you?
    "ශ්‍රී ලංකාව ඉතාම ලස්සන රටක්.",                    # Sri Lanka is a very beautiful country.
    "අද කාලගුණය හොඳයි.",                               # The weather is nice today.
    "මම සිංහල භාෂාව ඉගෙන ගනිමින් සිටිමි.",            # I am learning the Sinhala language.
    "ඔබේ නම කුමක්ද?",                                  # What is your name?
]


# ---------------------------------------------------------------------------
# TTS — edge-tts (no API key needed)
# ---------------------------------------------------------------------------
async def synthesize_tts(text: str, voice: str = DEFAULT_VOICE, output_path: str = None) -> str:
    """Synthesize Sinhala text to audio using edge-tts."""
    try:
        import edge_tts
    except ImportError:
        print("[ERROR] edge-tts not installed. Run: pip install edge-tts")
        sys.exit(1)

    AUDIO_OUT_DIR.mkdir(exist_ok=True)
    if output_path is None:
        output_path = str(AUDIO_OUT_DIR / f"tts_output_{int(time.time())}.mp3")

    print(f"\n[TTS] Voice: {voice}")
    print(f"[TTS] Text: {text}")
    print(f"[TTS] Synthesizing...")

    t_start = time.time()
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    elapsed = time.time() - t_start

    file_size = Path(output_path).stat().st_size
    print(f"[TTS] ✅ Done in {elapsed:.2f}s → {output_path} ({file_size} bytes)")
    return output_path


async def list_sinhala_voices():
    """List all available Sinhala voices from edge-tts."""
    try:
        import edge_tts
    except ImportError:
        print("[ERROR] edge-tts not installed. Run: pip install edge-tts")
        sys.exit(1)

    voices = await edge_tts.list_voices()
    sinhala = [v for v in voices if "si-LK" in v.get("ShortName", "")]
    print(f"\n[TTS] Available Sinhala voices ({len(sinhala)} found):")
    for v in sinhala:
        print(f"  - {v['ShortName']} | {v.get('Gender', '?')} | {v.get('FriendlyName', '')}")
    return sinhala


# ---------------------------------------------------------------------------
# STT — Groq Whisper large-v3
# ---------------------------------------------------------------------------
def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe a Sinhala audio file using Groq Whisper large-v3.

    Returns dict with:
        transcript (str), language (str), duration (float), elapsed_s (float)
    Note: Groq's Whisper API returns segments but not per-token confidence.
    We compute a proxy confidence from avg_logprob if available.
    """
    if not GROQ_API_KEY:
        print("[ERROR] GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    try:
        from groq import Groq
    except ImportError:
        print("[ERROR] groq not installed. Run: pip install groq")
        sys.exit(1)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"[ERROR] Audio file not found: {audio_path}")
        sys.exit(1)

    print(f"\n[STT] Audio file: {audio_path}")
    print(f"[STT] File size: {audio_path.stat().st_size} bytes")
    print(f"[STT] Sending to Groq Whisper large-v3...")

    client = Groq(api_key=GROQ_API_KEY)

    t_start = time.time()
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, f),
            model="whisper-large-v3",
            language="si",          # Sinhala language code
            response_format="verbose_json",
            temperature=0.0,
        )
    elapsed = time.time() - t_start

    # Extract transcript and confidence proxy
    transcript = response.text.strip()
    
    # Compute avg_logprob-based confidence from segments (higher = better)
    segments = getattr(response, "segments", []) or []
    if segments:
        avg_logprob = sum(s.get("avg_logprob", -1.0) for s in segments) / len(segments)
        # Convert log-prob to 0–1 scale (rough proxy: logprob of 0 = 1.0, -1.0 ≈ 0.37)
        confidence_proxy = min(1.0, max(0.0, 1.0 + avg_logprob))
    else:
        avg_logprob = None
        confidence_proxy = None

    detected_language = getattr(response, "language", "unknown")
    duration = getattr(response, "duration", None)

    result = {
        "transcript": transcript,
        "detected_language": detected_language,
        "duration_s": duration,
        "elapsed_s": round(elapsed, 2),
        "segment_count": len(segments),
        "avg_logprob": round(avg_logprob, 4) if avg_logprob is not None else None,
        "confidence_proxy_0_1": round(confidence_proxy, 3) if confidence_proxy is not None else None,
    }

    print(f"\n[STT] ✅ Transcription complete in {elapsed:.2f}s")
    print(f"  Detected language : {detected_language}")
    print(f"  Audio duration    : {duration}s")
    print(f"  Transcript        : {transcript}")
    print(f"  Avg log-prob      : {avg_logprob}")
    print(f"  Confidence proxy  : {confidence_proxy} (0=low, 1=high)")

    return result


# ---------------------------------------------------------------------------
# Round-trip test: TTS → audio file → STT → compare
# ---------------------------------------------------------------------------
async def run_round_trip():
    """
    Full round-trip feasibility test:
    1. Synthesize known Sinhala text to audio (edge-tts)
    2. Transcribe that audio back (Groq Whisper)
    3. Compare input vs output to get a rough quality signal
    """
    results = []
    AUDIO_OUT_DIR.mkdir(exist_ok=True)

    print("\n" + "="*60)
    print("PHASE 0 — ROUND-TRIP FEASIBILITY TEST")
    print("edge-tts synthesis → Groq Whisper transcription")
    print("="*60)

    for i, text in enumerate(SINHALA_SAMPLES):
        print(f"\n--- Sample {i+1}/{len(SINHALA_SAMPLES)} ---")
        print(f"Original text: {text}")

        # Step 1: TTS synthesis
        audio_path = str(AUDIO_OUT_DIR / f"sample_{i+1}.mp3")
        await synthesize_tts(text, DEFAULT_VOICE, audio_path)

        # Step 2: STT transcription
        stt_result = transcribe_audio(audio_path)

        # Step 3: Simple character-level similarity
        def char_overlap(a, b):
            a_chars = set(a)
            b_chars = set(b)
            if not a_chars:
                return 0.0
            return len(a_chars & b_chars) / len(a_chars)

        similarity = char_overlap(text, stt_result["transcript"])

        result_entry = {
            "sample_id": i + 1,
            "original_text": text,
            **stt_result,
            "char_set_similarity": round(similarity, 3),
        }
        results.append(result_entry)

        print(f"  Character-set overlap: {similarity:.1%}")

    # Summary
    print("\n" + "="*60)
    print("ROUND-TRIP SUMMARY")
    print("="*60)
    for r in results:
        status = "✅" if (r.get("confidence_proxy_0_1") or 0) > 0.5 else "⚠️"
        print(f"{status} Sample {r['sample_id']}: confidence={r.get('confidence_proxy_0_1')}, "
              f"similarity={r['char_set_similarity']:.1%}")
        print(f"   IN:  {r['original_text']}")
        print(f"   OUT: {r['transcript']}")

    # Save results JSON for PHASE0_FINDINGS.md
    results_path = SCRIPTS_DIR.parent / "PHASE0_RESULTS.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] Full results saved to: {results_path}")
    print("[INFO] Use these results to fill in PHASE0_FINDINGS.md")

    return results


# ---------------------------------------------------------------------------
# TTS-only test
# ---------------------------------------------------------------------------
async def run_tts_only():
    """Test edge-tts Sinhala synthesis without needing an API key."""
    print("\n" + "="*60)
    print("PHASE 0 — TTS-ONLY TEST (edge-tts, no key needed)")
    print("="*60)

    await list_sinhala_voices()

    AUDIO_OUT_DIR.mkdir(exist_ok=True)
    for voice in SINHALA_VOICES:
        text = SINHALA_SAMPLES[0]
        out = str(AUDIO_OUT_DIR / f"tts_test_{voice}.mp3")
        await synthesize_tts(text, voice, out)

    print(f"\n[TTS] ✅ All voices tested. Audio files saved to: {AUDIO_OUT_DIR}")
    print("[TTS] Play them to verify Sinhala quality.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 0 feasibility check for Sinhala Voice Assistant"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--audio", type=str,
        help="Path to a Sinhala audio file to transcribe via Groq Whisper"
    )
    group.add_argument(
        "--round-trip", action="store_true",
        help="Full round-trip test: TTS synthesis → Groq STT → compare"
    )
    group.add_argument(
        "--tts-only", action="store_true",
        help="Test edge-tts Sinhala voices only (no Groq key needed)"
    )
    group.add_argument(
        "--list-voices", action="store_true",
        help="List all available Sinhala voices from edge-tts"
    )
    args = parser.parse_args()

    if args.list_voices:
        asyncio.run(list_sinhala_voices())
    elif args.tts_only:
        asyncio.run(run_tts_only())
    elif args.round_trip:
        asyncio.run(run_round_trip())
    elif args.audio:
        transcribe_audio(args.audio)
    else:
        # Default: run TTS-only test first (safe, no key needed)
        print("[INFO] No arguments provided. Running TTS-only test.")
        print("[INFO] For full STT test: python scripts/p0_whisper_test.py --round-trip")
        asyncio.run(run_tts_only())


if __name__ == "__main__":
    main()
