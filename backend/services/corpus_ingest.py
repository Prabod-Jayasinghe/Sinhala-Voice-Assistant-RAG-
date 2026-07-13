"""
Corpus Ingestion Service
=========================
Pulls the Sinhala corpus from Hugging Face datasets, cleans,
chunks, embeds, and stores in ChromaDB.

Corpora used (from SDLC Section 5):
    1. NSINA Sinhala News Corpus (HF: "Ransaka/sinhala-news-data" or similar)
    2. Sinhala Wikipedia dump (HF: "wikipedia" config "si")

Both are real, freely available, research-grade datasets.
Neither requires scraping — they are static corpora suitable for an MVP.

Corpus scope (SDLC Section 15, assumption 1):
    General Sinhala Q&A: NSINA news + Sinhala Wikipedia subset.
    Static at build time (Phase 1). Scheduled refresh is Phase 2.
"""

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CORPUS_CACHE_DIR = Path(os.getenv("CORPUS_CACHE_DIR", "data/corpus_cache"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))        # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))   # characters
MAX_WIKI_DOCS = int(os.getenv("MAX_WIKI_DOCS", "2000")) # Wikipedia doc limit for free-tier
MAX_NEWS_DOCS = int(os.getenv("MAX_NEWS_DOCS", "3000")) # News doc limit

# Corpus ingestion state
_last_refreshed: Optional[datetime] = None
_document_count: int = 0


def get_corpus_stats() -> dict:
    return {
        "last_refreshed": _last_refreshed,
        "document_count": _document_count,
    }


def clean_sinhala_text(text: str) -> str:
    """
    Basic Sinhala text cleaning:
    - Remove excessive whitespace
    - Remove URLs
    - Normalize Unicode (NFC)
    - Remove lines that are mostly non-Sinhala (Latin-heavy lines often = metadata/bylines)
    """
    import unicodedata
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+", "", text)           # Remove URLs
    text = re.sub(r"\s+", " ", text).strip()           # Normalize whitespace

    # Filter lines: keep if Sinhala Unicode range (\u0D80-\u0DFF) is majority
    lines = text.split(".")
    sinhala_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sinhala_chars = sum(1 for c in line if "\u0D80" <= c <= "\u0DFF")
        total_alpha = sum(1 for c in line if c.isalpha())
        if total_alpha == 0 or sinhala_chars / total_alpha >= 0.5:
            sinhala_lines.append(line)

    return ". ".join(sinhala_lines).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Simple character-based chunking with overlap.
    Tries to split on sentence boundaries (। or . ) when possible.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a sentence boundary near the end
        boundary = -1
        for delim in ["।", ". ", ".\n"]:
            pos = text.rfind(delim, start + overlap, end)
            if pos != -1:
                boundary = pos + len(delim)
                break

        if boundary == -1:
            boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        start = boundary - overlap

    return [c for c in chunks if len(c.strip()) > 20]  # Drop tiny chunks


def ingest_corpus(retriever_service) -> dict:
    """
    Main ingestion function:
    1. Pull NSINA news corpus from HuggingFace
    2. Pull Sinhala Wikipedia subset from HuggingFace
    3. Clean + chunk each document
    4. Embed chunks with multilingual-e5-large (via RetrieverService)
    5. Upsert into ChromaDB

    Args:
        retriever_service: RetrieverService instance (provides embed_passages + collection)

    Returns:
        dict with ingested_count, chunk_count, message
    """
    global _last_refreshed, _document_count

    CORPUS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    all_docs = []

    # ------------------------------------------------------------------
    # 1. NSINA Sinhala News Corpus
    # ------------------------------------------------------------------
    print("[Ingest] Loading NSINA Sinhala News corpus from HuggingFace...")
    try:
        news_dataset = load_dataset(
            "Ransaka/sinhala-news-data",
            split="train",
            trust_remote_code=True,
        )
        news_count = 0
        for row in news_dataset:
            if news_count >= MAX_NEWS_DOCS:
                break
            title = str(row.get("title", "") or "")
            content = str(row.get("content", "") or row.get("text", "") or "")
            date_str = str(row.get("date", "") or row.get("published_date", "") or "")

            text = f"{title}\n{content}".strip()
            text = clean_sinhala_text(text)
            if len(text) < 50:  # Skip very short/empty docs
                continue

            all_docs.append({
                "id": str(uuid.uuid4()),
                "source": "NSINA",
                "source_type": "news",
                "title": title[:200],
                "text": text,
                "published_date": date_str[:10] if date_str else None,
            })
            news_count += 1

        print(f"[Ingest] Loaded {news_count} NSINA news documents.")
    except Exception as e:
        print(f"[Ingest] WARNING: Could not load NSINA corpus: {e}")
        print("[Ingest] Trying alternative news dataset...")
        try:
            # Fallback: try loading sinhala news from alternative source
            news_dataset = load_dataset(
                "Hiruni-Weerasekara/sinhala_news",
                split="train",
                trust_remote_code=True,
            )
            news_count = 0
            for row in news_dataset:
                if news_count >= MAX_NEWS_DOCS:
                    break
                text = clean_sinhala_text(str(row.get("text", "") or row.get("content", "") or ""))
                if len(text) < 50:
                    continue
                all_docs.append({
                    "id": str(uuid.uuid4()),
                    "source": "sinhala_news",
                    "source_type": "news",
                    "title": str(row.get("title", "Sinhala News"))[:200],
                    "text": text,
                    "published_date": None,
                })
                news_count += 1
            print(f"[Ingest] Loaded {news_count} Sinhala news documents (fallback source).")
        except Exception as e2:
            print(f"[Ingest] WARNING: Fallback news corpus also failed: {e2}")

    # ------------------------------------------------------------------
    # 2. Sinhala Wikipedia
    # ------------------------------------------------------------------
    print("[Ingest] Loading Sinhala Wikipedia from HuggingFace...")
    try:
        wiki_dataset = load_dataset(
            "wikipedia",
            "20231101.si",
            split="train",
            trust_remote_code=True,
        )
        wiki_count = 0
        for row in wiki_dataset:
            if wiki_count >= MAX_WIKI_DOCS:
                break
            title = str(row.get("title", "") or "")
            text = str(row.get("text", "") or "")
            text = clean_sinhala_text(f"{title}\n{text}")
            if len(text) < 50:
                continue
            all_docs.append({
                "id": str(uuid.uuid4()),
                "source": "sinhala_wikipedia",
                "source_type": "encyclopedia",
                "title": title[:200],
                "text": text,
                "published_date": None,
            })
            wiki_count += 1
        print(f"[Ingest] Loaded {wiki_count} Sinhala Wikipedia documents.")
    except Exception as e:
        print(f"[Ingest] WARNING: Could not load Sinhala Wikipedia: {e}")

    if not all_docs:
        return {
            "ingested_count": 0,
            "chunk_count": 0,
            "message": "No documents ingested — check HuggingFace dataset availability.",
        }

    # ------------------------------------------------------------------
    # 3. Chunk all documents
    # ------------------------------------------------------------------
    print(f"[Ingest] Chunking {len(all_docs)} documents...")
    all_chunks = []
    for doc in all_docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_id": doc["id"],
                "chunk_text": chunk,
                "chunk_index": i,
                "source": doc["source"],
                "source_type": doc["source_type"],
                "title": doc["title"],
                "published_date": doc.get("published_date"),
            })

    print(f"[Ingest] Total chunks: {len(all_chunks)}")

    # ------------------------------------------------------------------
    # 4. Embed and upsert in batches
    # ------------------------------------------------------------------
    BATCH_SIZE = 64
    total_upserted = 0
    collection = retriever_service.collection

    for batch_start in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "source": c["source"],
                "source_type": c["source_type"],
                "title": c["title"],
                "published_date": c["published_date"] or "",
                "chunk_index": c["chunk_index"],
                "document_id": c["document_id"],
            }
            for c in batch
        ]

        embeddings = retriever_service.embed_passages(texts)

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_upserted += len(batch)
        if total_upserted % 500 == 0:
            print(f"[Ingest] Upserted {total_upserted}/{len(all_chunks)} chunks...")

    _last_refreshed = datetime.now(timezone.utc)
    _document_count = len(all_docs)

    return {
        "ingested_count": len(all_docs),
        "chunk_count": total_upserted,
        "message": f"Successfully ingested {len(all_docs)} documents ({total_upserted} chunks) into ChromaDB.",
    }


async def scheduled_refresh_loop(retriever_service, interval_seconds: int = 86400):
    """Background task loop that refreshes the corpus periodically (FR-9)."""
    import asyncio
    print(f"[Scheduler] Starting scheduled refresh loop. Interval: {interval_seconds}s")
    while True:
        await asyncio.sleep(interval_seconds)
        print("[Scheduler] Triggering scheduled corpus refresh...")
        try:
            loop = asyncio.get_running_loop()
            # Run in executor to avoid blocking the main event loop
            await loop.run_in_executor(None, ingest_corpus, retriever_service)
            print("[Scheduler] Scheduled corpus refresh completed successfully.")
        except Exception as e:
            print(f"[Scheduler] Error during scheduled corpus refresh: {e}")

