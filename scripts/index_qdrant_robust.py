#!/usr/bin/env python3
"""
Robust Qdrant Indexer with Rate Limiting and Checkpoint Support
- Rate-limited embedding calls (0.5s delay)
- Checkpoint saving for resume capability
- Batch upload (100 vectors/batch)
- Progress tracking with tqdm
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "https://b3f75d28-bcfc-4f69-aeb8-6f3124af0735.eu-central-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate required credentials
if not QDRANT_API_KEY:
    print("❌ Error: QDRANT_API_KEY environment variable is required")
    print("Usage: QDRANT_API_KEY=xxx GROQ_API_KEY=xxx python index_qdrant_robust.py")
    sys.exit(1)

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY environment variable is required")
    print("Usage: QDRANT_API_KEY=xxx GROQ_API_KEY=xxx python index_qdrant_robust.py")
    sys.exit(1)

COLLECTION_NAME = "autocognitix"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 0.3  # seconds between API calls
MAX_RETRIES = 3

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "qdrant_checkpoint.json"


class CheckpointManager:
    """Manages checkpoint state for resumable indexing."""

    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.state = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {
            "dtc_indexed": [],
            "complaints_indexed": [],
            "recalls_indexed": [],
            "last_updated": None
        }

    def save(self):
        self.state["last_updated"] = datetime.now().isoformat()
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def is_indexed(self, category: str, item_id: str) -> bool:
        return item_id in self.state.get(f"{category}_indexed", [])

    def mark_indexed(self, category: str, item_id: str):
        key = f"{category}_indexed"
        if key not in self.state:
            self.state[key] = []
        if item_id not in self.state[key]:
            self.state[key].append(item_id)


class EmbeddingService:
    """Service for generating embeddings with rate limiting and retry."""

    def __init__(self, groq_api_key: str):
        self.groq_api_key = groq_api_key
        self.client = httpx.AsyncClient(timeout=120)
        self.last_call_time = 0

    async def close(self):
        await self.client.aclose()

    async def _rate_limit(self):
        """Ensure minimum delay between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_call_time = time.time()

    async def get_embedding(self, text: str) -> Optional[list]:
        """Get embedding for text with retry logic."""
        if not self.groq_api_key:
            return None

        await self._rate_limit()

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.post(
                    "https://api.groq.com/openai/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": EMBEDDING_MODEL,
                        "input": text[:8000]  # Truncate to avoid token limits
                    }
                )

                if response.status_code == 429:
                    # Rate limited - wait longer
                    retry_after = int(response.headers.get("retry-after", 60))
                    print(f"\n  ⚠️ Rate limited, waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code == 200:
                    data = response.json()
                    return data["data"][0]["embedding"]
                else:
                    print(f"\n  ⚠️ Embedding error {response.status_code}: {response.text[:100]}")

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = 2 ** attempt
                    print(f"\n  ⚠️ Retry {attempt + 1}/{MAX_RETRIES} after {delay}s: {str(e)[:50]}")
                    await asyncio.sleep(delay)

        return None


class RobustQdrantIndexer:
    """Robust indexer for Qdrant with checkpoint support."""

    def __init__(self):
        self.qdrant = None
        self.embedding_service = None
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)
        self.stats = {
            "dtc": 0,
            "complaints": 0,
            "recalls": 0,
            "errors": 0
        }

    async def connect(self):
        """Connect to Qdrant and embedding service."""
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Create collection if not exists
        collections = self.qdrant.get_collections()
        if COLLECTION_NAME not in [c.name for c in collections.collections]:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
        else:
            info = self.qdrant.get_collection(COLLECTION_NAME)
            print(f"✅ Connected to Qdrant (existing points: {info.points_count:,})")

        self.embedding_service = EmbeddingService(GROQ_API_KEY)

    async def close(self):
        """Close connections."""
        if self.embedding_service:
            await self.embedding_service.close()

    async def index_dtc_codes(self):
        """Index DTC codes."""
        all_codes = []

        # Load merged DTC codes
        merged_file = DATA_DIR / "dtc" / "all_codes_merged.json"
        if merged_file.exists():
            with open(merged_file) as f:
                data = json.load(f)
                codes = data.get("codes", data) if isinstance(data, dict) else data
                if isinstance(codes, list):
                    all_codes.extend(codes)

        # Load dtcdb codes
        dtcdb_file = DATA_DIR / "dtc" / "dtcdb_codes.json"
        if dtcdb_file.exists():
            with open(dtcdb_file) as f:
                data = json.load(f)
                codes = data.get("codes", [])
                all_codes.extend(codes)

        # Deduplicate and filter already indexed
        seen = set()
        unique_codes = []
        for c in all_codes:
            code = c.get("code", "")
            if code and code not in seen:
                if not self.checkpoint.is_indexed("dtc", code):
                    seen.add(code)
                    unique_codes.append(c)

        if not unique_codes:
            print("⏭️  All DTC codes already indexed, skipping...")
            return

        print(f"Indexing {len(unique_codes)} DTC codes...")
        points = []

        with tqdm(total=len(unique_codes), desc="DTC Embeddings") as pbar:
            for c in unique_codes:
                code = c.get("code", "")
                text = f"{code}: {c.get('description', '')} - {c.get('category', '')} {c.get('subcategory', '')}"

                embedding = await self.embedding_service.get_embedding(text)
                if embedding:
                    points.append(PointStruct(
                        id=hash(code) & 0x7FFFFFFF,
                        vector=embedding,
                        payload={
                            "type": "dtc",
                            "code": code,
                            "description": c.get("description", ""),
                            "category": c.get("category", ""),
                            "subcategory": c.get("subcategory", "")
                        }
                    ))
                    self.checkpoint.mark_indexed("dtc", code)
                    self.stats["dtc"] += 1
                else:
                    self.stats["errors"] += 1

                pbar.update(1)

                # Batch upload
                if len(points) >= BATCH_SIZE:
                    self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                    self.checkpoint.save()
                    points = []

        # Upload remaining
        if points:
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            self.checkpoint.save()

        print(f"✅ Indexed {self.stats['dtc']} DTC codes")

    async def index_complaints(self, limit: int = 20000):
        """Index NHTSA complaints (limited to most recent)."""
        complaints_dir = DATA_DIR / "nhtsa" / "complaints"
        if not complaints_dir.exists():
            print(f"⚠️  Complaints directory not found: {complaints_dir}")
            return

        # Collect complaints from all files
        all_complaints = []
        for file_path in sorted(complaints_dir.glob("complaints_*.json"), reverse=True):
            with open(file_path) as f:
                data = json.load(f)
                # Handle both formats: list or dict with 'complaints' key
                if isinstance(data, dict):
                    complaints = data.get("complaints", [])
                else:
                    complaints = data
                all_complaints.extend(complaints)
                if len(all_complaints) >= limit * 2:  # Get more for deduplication
                    break

        # Filter already indexed
        to_index = []
        for c in all_complaints:
            odi_id = str(c.get("ODI_ID", c.get("CMPLID", "")))
            if odi_id and not self.checkpoint.is_indexed("complaints", odi_id):
                to_index.append(c)
                if len(to_index) >= limit:
                    break

        if not to_index:
            print("⏭️  All complaints already indexed, skipping...")
            return

        print(f"Indexing {len(to_index)} complaints...")
        points = []

        with tqdm(total=len(to_index), desc="Complaint Embeddings") as pbar:
            for c in to_index:
                odi_id = str(c.get("ODI_ID", c.get("CMPLID", "")))
                make = c.get("MAKETXT", c.get("make", ""))
                model = c.get("MODELTXT", c.get("model", ""))
                year = c.get("YEARTXT", c.get("year", ""))
                component = c.get("COMPNAME", c.get("component", ""))
                description = c.get("CDESCR", c.get("description", ""))

                text = f"{make} {model} {year} - {component}: {description}"

                embedding = await self.embedding_service.get_embedding(text)
                if embedding:
                    points.append(PointStruct(
                        id=hash(odi_id) & 0x7FFFFFFF,
                        vector=embedding,
                        payload={
                            "type": "complaint",
                            "odi_id": odi_id,
                            "make": make,
                            "model": model,
                            "year": year,
                            "component": component,
                            "description": description[:1000]
                        }
                    ))
                    self.checkpoint.mark_indexed("complaints", odi_id)
                    self.stats["complaints"] += 1
                else:
                    self.stats["errors"] += 1

                pbar.update(1)

                # Batch upload
                if len(points) >= BATCH_SIZE:
                    self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                    self.checkpoint.save()
                    points = []

        # Upload remaining
        if points:
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            self.checkpoint.save()

        print(f"✅ Indexed {self.stats['complaints']} complaints")

    async def index_recalls(self):
        """Index NHTSA recalls."""
        recalls_dir = DATA_DIR / "nhtsa" / "recalls"
        if not recalls_dir.exists():
            print(f"⚠️  Recalls directory not found: {recalls_dir}")
            return

        # Collect all recalls
        all_recalls = []
        for file_path in recalls_dir.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
                recalls = data if isinstance(data, list) else data.get("recalls", [])
                all_recalls.extend(recalls)

        # Filter already indexed
        to_index = []
        for r in all_recalls:
            campaign = r.get("NHTSACampaignNumber", r.get("campaign_number", ""))
            if campaign and not self.checkpoint.is_indexed("recalls", campaign):
                to_index.append(r)

        if not to_index:
            print("⏭️  All recalls already indexed, skipping...")
            return

        print(f"Indexing {len(to_index)} recalls...")
        points = []

        with tqdm(total=len(to_index), desc="Recall Embeddings") as pbar:
            for r in to_index:
                campaign = r.get("NHTSACampaignNumber", r.get("campaign_number", ""))
                make = r.get("Make", r.get("make", ""))
                model = r.get("Model", r.get("model", ""))
                year = str(r.get("ModelYear", r.get("year", "")))
                component = r.get("Component", r.get("component", ""))
                summary = r.get("Summary", r.get("summary", "")) or ""
                consequence = r.get("Consequence", r.get("consequence", "")) or ""
                remedy = r.get("Remedy", r.get("remedy", "")) or ""

                text = f"{make} {model} {year} - {component}. {summary} Consequence: {consequence} Remedy: {remedy}"

                embedding = await self.embedding_service.get_embedding(text)
                if embedding:
                    points.append(PointStruct(
                        id=hash(campaign) & 0x7FFFFFFF,
                        vector=embedding,
                        payload={
                            "type": "recall",
                            "campaign_number": campaign,
                            "make": make,
                            "model": model,
                            "year": year,
                            "component": component,
                            "summary": summary[:1000],
                            "consequence": consequence[:500],
                            "remedy": remedy[:500]
                        }
                    ))
                    self.checkpoint.mark_indexed("recalls", campaign)
                    self.stats["recalls"] += 1
                else:
                    self.stats["errors"] += 1

                pbar.update(1)

                # Batch upload
                if len(points) >= BATCH_SIZE:
                    self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                    self.checkpoint.save()
                    points = []

        # Upload remaining
        if points:
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            self.checkpoint.save()

        print(f"✅ Indexed {self.stats['recalls']} recalls")

    async def run(self, resume: bool = True):
        """Run the full indexing process."""
        print("=" * 60)
        print("ROBUST QDRANT INDEXER")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Resume mode: {resume}")
        if resume and self.checkpoint.state["last_updated"]:
            print(f"Last checkpoint: {self.checkpoint.state['last_updated']}")
        print()

        if not GROQ_API_KEY:
            print("❌ Error: GROQ_API_KEY environment variable not set")
            print("Usage: GROQ_API_KEY=xxx python index_qdrant_robust.py")
            return

        try:
            await self.connect()

            await self.index_dtc_codes()
            await self.index_recalls()
            await self.index_complaints(limit=20000)

            # Final stats
            info = self.qdrant.get_collection(COLLECTION_NAME)

            print()
            print("=" * 60)
            print("INDEXING COMPLETE")
            print("=" * 60)
            print(f"DTC Codes: {self.stats['dtc']:,}")
            print(f"Complaints: {self.stats['complaints']:,}")
            print(f"Recalls: {self.stats['recalls']:,}")
            print(f"Errors: {self.stats['errors']:,}")
            print(f"Total in collection: {info.points_count:,}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Progress saved to checkpoint. Run with --resume to continue.")
            raise
        finally:
            await self.close()


async def main():
    resume = "--resume" in sys.argv or "-r" in sys.argv
    indexer = RobustQdrantIndexer()
    await indexer.run(resume=resume)


if __name__ == "__main__":
    asyncio.run(main())
