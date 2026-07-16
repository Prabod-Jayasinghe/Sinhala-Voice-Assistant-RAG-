# 🎙️ Sinhala Voice Assistant (RAG)

> The first open, complete voice-in / voice-out AI assistant built for Sinhala (සිංහල) — a confirmed low-resource language spoken by 17M+ people with almost no working voice-AI tooling of its own.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-Next.js-3178C6?logo=typescript&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?logo=nextdotjs&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind%20CSS-Styling-06B6D4?logo=tailwindcss&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Schemas-E92063?logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-ASGI%20Server-2C3E50?logo=gunicorn&logoColor=white)

![Groq](https://img.shields.io/badge/Groq-Whisper%20large--v3-F55036?logo=speedtest&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-Flash-8E75B2?logo=googlegemini&logoColor=white)
![edge-tts](https://img.shields.io/badge/edge--tts-Sinhala%20TTS-0078D4?logo=microsoftedge&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6F00?logo=databricks&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-FFD21E?logo=huggingface&logoColor=black)

![pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?logo=pytest&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?logo=githubactions&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Frontend%20Hosting-000000?logo=vercel&logoColor=white)
![Render](https://img.shields.io/badge/Render-Backend%20Hosting-46E3B7?logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)


---

## Table of Contents

- [Overview](#overview)
- [Screenshots](#screenshots)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tools & Technologies](#tools--technologies)
- [Why This Project Is Different](#why-this-project-is-different)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Deployment](#deployment)
- [Known Limitations](#known-limitations)
- [License](#license)
- [Author](#author)

---

## Overview

Sinhala speakers have almost no native voice-AI assistant, because Sinhala is a confirmed low-resource language across speech recognition, transcription, and generation. This project closes that gap end-to-end: a user speaks (or types) a question in Sinhala, the system transcribes it, retrieves grounded facts from a real Sinhala knowledge base, generates a Sinhala answer, and speaks that answer back — all in one pipeline, deployed on free-tier infrastructure.

It was built as a research-grounded portfolio project: every design decision (model choice, corpus, fallback UX) is backed by a documented feasibility study rather than assumption, and every limitation of working with a low-resource language is measured and surfaced to the user instead of hidden.

## Screenshots

<!--
  Add images to a /docs/screenshots (or /assets) folder in the repo and update the
  paths below. Recommended shots: the main voice-query screen, the confidence /
  "Did I hear that right?" retry UI, and a system architecture diagram.
-->

| Voice Query Interface |
|---|
| ![Sinhala Voice Assistant UI](./docs/screenshots/app-screenshot.png) |

https://sinhala-voice-assistant-9a4elg9wy-j-coder-shans-projects.vercel.app/


## Key Features

- 🎤 **Voice-in, voice-out** — full spoken-question-to-spoken-answer loop, with a visible text-input fallback for reliability
- 📚 **Grounded, citable answers** — retrieval-augmented generation over a real Sinhala corpus (NSINA news + Sinhala Wikipedia), so the model answers from evidence instead of hallucinating
- 🗣️ **Natural Sinhala speech synthesis** — two neural Sinhala voices (`si-LK-ThiliniNeural`, `si-LK-SameeraNeural`) via `edge-tts`, free and keyless
- 📊 **Transparent confidence UX** — live STT confidence shown per query, with a "Did I hear that right?" retry prompt below a measured threshold
- 🧠 **Multi-turn conversation** — maintains context across follow-up questions instead of treating each query in isolation
- 🔤 **Singlish input detection** — recognizes romanized Sinhala typed in Latin script and routes it correctly
- 🔄 **Scheduled corpus refresh** — automated pipeline keeps the Sinhala news corpus current without manual re-ingestion
- 🛡️ **Documented content-safety layer** — conservative keyword/heuristic filter for Sinhala, built with explicit acknowledgment that robust Sinhala moderation is still an open research problem
- ✅ **Fully tested backend** — pytest + httpx test suite, CI pipeline on every push

## Architecture

```
User mic → Next.js frontend → FastAPI backend
                                  ├── Groq Whisper large-v3       (Speech-to-Text)
                                  ├── multilingual-e5-large + ChromaDB  (Retrieval)
                                  ├── Google Gemini Flash          (Answer generation)
                                  └── edge-tts                     (Sinhala Text-to-Speech)
```

| Layer | Technology | Why |
|---|---|---|
| Frontend | Next.js + TypeScript + Tailwind | Free Vercel hosting, browser MediaRecorder API |
| Backend | FastAPI | Async, fast, production-familiar |
| STT | Groq Whisper large-v3 | Free tier, fast, hosted — no local GPU needed |
| LLM | Google Gemini Flash | Measurably better Sinhala generation than Llama 3 |
| Embeddings | `intfloat/multilingual-e5-large` | Free, CPU-friendly, best available option for Sinhala |
| Vector store | ChromaDB (embedded, file-based) | Free, no separate server to manage |
| TTS | `edge-tts` (`si-LK-ThiliniNeural` / `si-LK-SameeraNeural`) | Two real Sinhala neural voices, free, no API key |
| Corpus | NSINA Sinhala News + Sinhala Wikipedia (HuggingFace) | Real, citable, research-grade data |

## Tools & Technologies

**Languages**
- Python 3.11+ — backend, ML pipeline, data ingestion
- TypeScript — frontend application

**Backend & API**
- [FastAPI](https://fastapi.tiangolo.com/) — async REST API framework
- [Pydantic](https://docs.pydantic.dev/) — request/response schema validation
- [Uvicorn](https://www.uvicorn.org/) — ASGI server

**Speech & Language AI**
- [Groq Whisper large-v3](https://console.groq.com/) — speech-to-text (STT), hosted inference
- [Google Gemini Flash](https://aistudio.google.com/) — grounded answer generation (LLM)
- [`edge-tts`](https://github.com/rany2/edge-tts) — Sinhala text-to-speech, using the `si-LK-ThiliniNeural` and `si-LK-SameeraNeural` neural voices
- [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) — multilingual sentence embeddings for retrieval

**Retrieval & Data**
- [ChromaDB](https://www.trychroma.com/) — embedded vector store for RAG retrieval
- [HuggingFace Datasets](https://huggingface.co/docs/datasets) — corpus loading (NSINA Sinhala News, Sinhala Wikipedia)

**Frontend**
- [Next.js](https://nextjs.org/) — React framework, SSR/routing
- [Tailwind CSS](https://tailwindcss.com/) — utility-first styling
- Browser `MediaRecorder` API — in-browser microphone capture

**Testing & CI/CD**
- [pytest](https://docs.pytest.org/) + [httpx](https://www.python-httpx.org/) — backend API test suite
- [GitHub Actions](https://github.com/features/actions) — continuous integration pipeline

**Hosting & Infrastructure**
- [Vercel](https://vercel.com/) — frontend hosting (free tier)
- [Render](https://render.com/) — backend hosting (free tier)

**Development Tooling**
- Git & GitHub — version control, feature-branch workflow with conventional commits
- Google Antigravity — agentic IDE used for structured, spec-first development

## Why This Project Is Different

Most "low-resource language" demos quietly cherry-pick easy examples and never mention failure modes. This project runs the opposite way — a Phase 0 feasibility study measured real STT performance on Sinhala before any product code was written, and those numbers directly shaped the UX:

- Groq Whisper correctly identifies Sinhala and produces Sinhala Unicode output in all test cases, but word-level transcription confidence on clean audio averages well below what English pipelines typically achieve
- Rather than hide this, the UI surfaces per-query confidence, offers retry, and keeps text input always visible — never buried behind a mic-only interface
- The generation layer refuses to answer outside its retrieved context: if no relevant corpus chunk is found, it returns a fixed "I don't have information on this" response instead of a hallucinated one

This measure-first, document-honestly approach is the core engineering decision behind the whole project.

## Tech Stack

- **Backend:** Python, FastAPI, ChromaDB, pytest, httpx
- **Frontend:** Next.js, TypeScript, Tailwind CSS
- **AI/ML:** Groq Whisper large-v3 (STT), Google Gemini Flash (generation), `multilingual-e5-large` (embeddings), `edge-tts` (TTS)
- **Data:** NSINA Sinhala News Corpus, Sinhala Wikipedia (via HuggingFace Datasets)
- **Infra:** GitHub Actions (CI), Vercel (frontend hosting), Render (backend hosting)

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Groq API key (free — [console.groq.com](https://console.groq.com))
- Google Gemini API key (free — [aistudio.google.com](https://aistudio.google.com))
- No other keys needed — `edge-tts` is keyless

### Backend

```bash
cd backend
pip install -r requirements.txt
cp ../.env.example ../.env   # add GROQ_API_KEY and GEMINI_API_KEY
uvicorn main:app --reload
```

### Corpus Ingestion

Run once before first use — pulls NSINA + Sinhala Wikipedia from HuggingFace:

```bash
curl -X POST http://localhost:8000/api/corpus/refresh
curl http://localhost:8000/api/corpus/status   # check progress
```

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev
```

### Running Tests

```bash
cd tests
python -m pytest test_api.py -v
```

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── routers/
│   │   ├── voice_query.py      # POST /api/voice-query
│   │   ├── text_query.py       # POST /api/text-query (text fallback)
│   │   └── corpus.py           # GET/POST /api/corpus/...
│   ├── services/
│   │   ├── stt.py              # Groq Whisper wrapper
│   │   ├── tts.py              # edge-tts Sinhala wrapper
│   │   ├── retriever.py        # multilingual-e5 + ChromaDB
│   │   ├── generator.py        # Gemini Flash prompting
│   │   └── corpus_ingest.py    # HuggingFace dataset pull + chunk + embed
│   └── models/schemas.py       # Pydantic data models
├── frontend/                   # Next.js application
├── scripts/
│   └── p0_whisper_test.py      # Phase 0 feasibility test script
├── tests/
│   └── test_api.py             # pytest + httpx backend tests
├── PHASE0_FINDINGS.md          # Phase 0 STT/TTS feasibility findings
├── PHASE0_RESULTS.json         # Raw Phase 0 test data
└── Sinhala-Voice-Assistant-SDLC.md   # Full SDLC documentation
```

## Data Sources

- **NSINA Sinhala News Corpus** — research-grade Sinhala news dataset (HuggingFace)
- **Sinhala Wikipedia** — `wikipedia` dataset, `si` config (HuggingFace)

Both datasets are freely available research releases; review their license terms before any use beyond a personal portfolio demo.

## Deployment

- **Backend → Render** (free tier): set `GROQ_API_KEY` and `GEMINI_API_KEY`, deploy `backend/` as a web service
- **Frontend → Vercel** (free tier): set `NEXT_PUBLIC_API_URL` to the deployed Render backend URL
- Render's free tier has cold-start delays (~30s), documented in the UI so first-query latency isn't mistaken for a bug

## Known Limitations

- Sinhala speech-to-text confidence trails English-grade pipelines — this is a documented, measured limitation of current STT models on low-resource languages, not a bug, and the UX is designed around it
- Sinhala offensive-language detection remains an unsolved research problem industry-wide; the built-in filter is intentionally conservative and not presented as a complete moderation solution
- `edge-tts` is an unofficial wrapper around Microsoft's Edge TTS service — free and reliable for a portfolio-scale demo, but not intended for production traffic at scale

## License

MIT

## Author

**Prabod Jayasinghe (Shan)**
Final-year Electronics & Computer Science undergraduate, University of Kelaniya

- GitHub: [@j-coder-shan](https://github.com/j-coder-shan)
- LinkedIn: [shanprabodh](https://www.linkedin.com/in/prabod-jayasinghe-76323830a/)

---

