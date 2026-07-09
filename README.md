# Sinhala Voice Assistant (RAG)

> A full voice-in/voice-out RAG assistant for Sinhala (සිංහල) — the first open attempt at a complete voice-first AI assistant for this language. Accepts spoken Sinhala questions, retrieves grounded answers from a real Sinhala corpus (NSINA news + Sinhala Wikipedia), and speaks the answer back in natural Sinhala.

[![CI](https://github.com/j-coder-shan/Sinhala-Voice-Assistant-RAG-/actions/workflows/ci.yml/badge.svg)](https://github.com/j-coder-shan/Sinhala-Voice-Assistant-RAG-/actions/workflows/ci.yml)

---

## What This Is

**Problem:** Sinhala speakers (17M+) have almost no working voice-AI assistant in their own language. Sinhala is a confirmed low-resource language across speech, transcription, and generation.

**Solution:** A web app where a user speaks (or types) a question in Sinhala → the system transcribes it → retrieves relevant facts from a Sinhala knowledge base → generates a grounded Sinhala answer → and speaks it back.

---

## Architecture

```
User mic → Next.js frontend → FastAPI backend
                                  ├── Groq Whisper large-v3 (STT)
                                  ├── multilingual-e5-large + ChromaDB (RAG)
                                  ├── Gemini Flash (answer generation)
                                  └── edge-tts (Sinhala TTS)
```

| Layer | Technology | Why |
|---|---|---|
| Frontend | Next.js + TypeScript + Tailwind | Free Vercel hosting, browser MediaRecorder API |
| Backend | FastAPI | Async, fast, familiar |
| STT | Groq Whisper large-v3 | Free tier, fast, hosted (no local GPU needed) |
| LLM | Google Gemini Flash (free tier) | Materially better Sinhala generation than Llama 3 |
| Embeddings | `intfloat/multilingual-e5-large` | Free, runs on CPU, best available option for Sinhala |
| Vector store | ChromaDB (embedded, file-based) | Free, no separate server |
| TTS | `edge-tts` (`si-LK-ThiliniNeural`) | Two real Sinhala neural voices, free, no API key |
| Corpus | NSINA Sinhala News + Sinhala Wikipedia (HuggingFace) | Real, citable, research-grade |

---

## Honest Accuracy Statement

> **This is important to read before using or demoing this system.**

Sinhala is a **confirmed low-resource language** for speech and text AI. This affects the system in documented, known ways:

### STT (Speech-to-Text) Quality

Phase 0 feasibility testing (2026-07-09) found:
- Groq Whisper large-v3 **correctly identifies the language as Sinhala** in all test cases
- Whisper **produces Sinhala Unicode script output** — it understands the domain
- **Average word-level confidence: 0.296** (0–1 scale) on clean TTS audio
- **Average character-set overlap: 46.5%** between input and transcribed text
- This means: transcripts are in the right phonetic neighborhood but often contain wrong words

This is consistent with published research on Whisper's performance on low-resource languages. It is a **documented limitation, not a bug.**

**How the system handles this:**
- STT confidence is shown in the UI for every voice query
- Below threshold (< 0.4): UI shows "Did I hear that right?" with retry option
- Text input fallback is **always visible** — not hidden behind the mic UI
- Curated sample questions with pronunciation guides are provided in the UI

### Generation (LLM) Quality

- Gemini Flash has meaningfully better Sinhala support than Llama 3 — this is why Gemini was chosen over Groq/Llama 3 for this project
- All answers are **grounded in retrieved corpus chunks** — the LLM is not called blind
- If no relevant corpus chunks are found, the system returns a fixed "I don't have information on this" response **rather than generating a hallucinated answer**

### Content Safety

Sinhala offensive-language detection is an **active, unsolved research problem**. Major commercial detectors perform poorly on Sinhala. This MVP uses a conservative keyword/heuristic filter and documents this explicitly — it does not claim robust moderation it doesn't have.

### TTS (Text-to-Speech)

`edge-tts` is an unofficial open-source wrapper around Microsoft's Edge browser TTS service. Both confirmed Sinhala neural voices (`si-LK-ThiliniNeural` female, `si-LK-SameeraNeural` male) produce natural, clear Sinhala speech. **This service is free and requires no API key, but is not officially supported by Microsoft for third-party programmatic use** — suitable for a portfolio demo, not production.

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Groq API key (free — [console.groq.com](https://console.groq.com))
- Google Gemini API key (free — [aistudio.google.com](https://aistudio.google.com))
- No other keys needed (edge-tts is keyless)

### Backend

```bash
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env   # Fill in GROQ_API_KEY and GEMINI_API_KEY
uvicorn main:app --reload
```

### Corpus Ingestion

Run once before first use (pulls NSINA + Sinhala Wikipedia from HuggingFace):

```bash
# Via API (after backend is running):
curl -X POST http://localhost:8000/api/corpus/refresh

# Or check status:
curl http://localhost:8000/api/corpus/status
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # Set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

---

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── routers/
│   │   ├── voice_query.py      # POST /api/voice-query
│   │   ├── text_query.py       # POST /api/text-query (FR-8 fallback)
│   │   └── corpus.py           # GET/POST /api/corpus/...
│   ├── services/
│   │   ├── stt.py              # Groq Whisper wrapper
│   │   ├── tts.py              # edge-tts Sinhala wrapper
│   │   ├── retriever.py        # multilingual-e5 + ChromaDB
│   │   ├── generator.py        # Gemini Flash prompting
│   │   └── corpus_ingest.py    # HF dataset pull + chunk + embed
│   └── models/schemas.py       # Pydantic data models
├── frontend/                   # Next.js app (Phase 1)
├── scripts/
│   └── p0_whisper_test.py      # Phase 0 feasibility test script
├── tests/
│   └── test_api.py             # pytest + httpx backend tests
├── PHASE0_FINDINGS.md          # Phase 0 STT/TTS feasibility findings
├── PHASE0_RESULTS.json         # Raw Phase 0 test data
└── Sinhala-Voice-Assistant-SDLC.md  # Full project documentation
```

---

## Data Sources

- **NSINA Sinhala News Corpus** — research-grade Sinhala news dataset (HuggingFace)
- **Sinhala Wikipedia** — `wikipedia` dataset config `si` (HuggingFace)

Both datasets are freely available research releases. Check their specific license terms before any use beyond personal portfolio demo.

---

## Build Roadmap

- [x] **Phase 0** — Feasibility check (Whisper STT + edge-tts) → [PHASE0_FINDINGS.md](PHASE0_FINDINGS.md)
- [/] **Phase 1** — MVP core: full voice pipeline + Next.js frontend
- [ ] **Phase 2** — Scheduled corpus refresh, Singlish input detection
- [ ] **Phase 3** — Multi-turn conversation, Sinhala model fine-tuning exploration

---

## Running Tests

```bash
cd tests
python -m pytest test_api.py -v
```

---

## Deployment

- **Backend → Render free tier**: set env vars (`GROQ_API_KEY`, `GEMINI_API_KEY`), deploy `backend/` as web service
- **Frontend → Vercel free tier**: set `NEXT_PUBLIC_API_URL` to Render backend URL
- Note: Render free tier has cold start delays (~30s) — document this expectation in the UI

---

*Built as a portfolio project demonstrating RAG architecture for a genuine low-resource language challenge.*
