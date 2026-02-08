#!/usr/bin/env python3
"""
Index ALL data to Qdrant Cloud for vector search.

Creates embeddings for:
- DTC codes (descriptions)
- NHTSA Complaints (summaries)
- NHTSA Recalls (summaries + remedies)

Uses Groq API for embeddings (fast and free).
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)
from tqdm import tqdm

# Qdrant Cloud connection - Load from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Groq for embeddings (fast, free tier)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Validate required credentials
if not QDRANT_URL or not QDRANT_API_KEY:
    print("Error: QDRANT_URL and QDRANT_API_KEY environment variables are required")
    print("Usage: QDRANT_URL=xxx QDRANT_API_KEY=xxx GROQ_API_KEY=xxx python index_all_to_qdrant.py")
    print("Or set them in your .env file")
    sys.exit(1)

if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY environment variable is required")
    print("Usage: QDRANT_URL=xxx QDRANT_API_KEY=xxx GROQ_API_KEY=xxx python index_all_to_qdrant.py")
    print("Or set them in your .env file")
    sys.exit(1)
EMBEDDING_MODEL = "nomic-embed-text"  # 768 dimensions
EMBEDDING_DIM = 768

DATA_DIR = Path(__file__).parent.parent / "data"
COLLECTION_NAME = "autocognitix"


class QdrantIndexer:
    def __init__(self):
        self.client = None
        self.http_client = None
        self.stats = {
            "dtc_codes": 0,
            "complaints": 0,
            "recalls": 0,
            "total_vectors": 0
        }

    async def connect(self):
        """Connect to Qdrant Cloud."""
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        self.http_client = httpx.AsyncClient(timeout=60.0)

        # Check/create collection
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection: {COLLECTION_NAME}")
        else:
            info = self.client.get_collection(COLLECTION_NAME)
            print(f"✅ Connected to Qdrant Cloud (existing points: {info.points_count:,})")

    async def close(self):
        """Close connections."""
        if self.http_client:
            await self.http_client.aclose()

    async def get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding from Groq API."""
        if not text or not GROQ_API_KEY:
            return None

        try:
            response = await self.http_client.post(
                "https://api.groq.com/openai/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": EMBEDDING_MODEL,
                    "input": text[:8000]  # Limit text length
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"  Embedding error: {e}")
            return None

    def generate_id(self, prefix: str, content: str) -> str:
        """Generate consistent ID from content."""
        hash_input = f"{prefix}:{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    async def index_dtc_codes(self):
        """Index DTC codes from dtcdb."""
        dtcdb_file = DATA_DIR / "dtc" / "dtcdb_codes.json"
        if not dtcdb_file.exists():
            print("⚠️ No dtcdb codes file found")
            return

        with open(dtcdb_file) as f:
            data = json.load(f)

        codes = data.get("codes", [])
        print(f"Indexing {len(codes)} DTC codes...")

        points = []
        for code_data in tqdm(codes, desc="DTC Embeddings"):
            code = code_data.get("code", "")
            description = code_data.get("description", "")

            if not code or not description:
                continue

            # Create searchable text
            text = f"DTC {code}: {description}"

            embedding = await self.get_embedding(text)
            if embedding:
                point_id = self.generate_id("dtc", code)
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "type": "dtc",
                        "code": code,
                        "description": description,
                        "category": code_data.get("category", ""),
                        "subcategory": code_data.get("subcategory", ""),
                        "text": text
                    }
                ))
                self.stats["dtc_codes"] += 1

            # Batch upload every 100 points
            if len(points) >= 100:
                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                self.stats["total_vectors"] += len(points)
                points = []

            # Rate limiting
            await asyncio.sleep(0.1)

        # Upload remaining
        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            self.stats["total_vectors"] += len(points)

        print(f"✅ Indexed {self.stats['dtc_codes']} DTC codes")

    async def index_complaints(self, limit: int = 5000):
        """Index NHTSA complaints (limited for free tier)."""
        complaints_file = DATA_DIR / "nhtsa" / "complaints.json"
        if not complaints_file.exists():
            print("⚠️ No complaints file found")
            return

        with open(complaints_file) as f:
            data = json.load(f)

        complaints = data.get("complaints", [])[:limit]
        print(f"Indexing {len(complaints)} complaints (limited to {limit})...")

        points = []
        for complaint in tqdm(complaints, desc="Complaint Embeddings"):
            odi_number = complaint.get("odi_number") or complaint.get("ODI_ID")
            summary = complaint.get("summary") or complaint.get("CDESCR", "")

            if not odi_number or not summary:
                continue

            make = complaint.get("make") or complaint.get("MAKETXT", "")
            model = complaint.get("model") or complaint.get("MODELTXT", "")
            year = complaint.get("model_year") or complaint.get("YEARTXT", "")
            component = complaint.get("component") or complaint.get("COMPDESC", "")

            # Create searchable text
            text = f"{make} {model} {year} - {component}: {summary[:500]}"

            embedding = await self.get_embedding(text)
            if embedding:
                point_id = self.generate_id("complaint", str(odi_number))
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "type": "complaint",
                        "odi_number": str(odi_number),
                        "make": make,
                        "model": model,
                        "year": str(year),
                        "component": component,
                        "summary": summary[:1000],
                        "text": text
                    }
                ))
                self.stats["complaints"] += 1

            # Batch upload
            if len(points) >= 100:
                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                self.stats["total_vectors"] += len(points)
                points = []

            await asyncio.sleep(0.1)

        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            self.stats["total_vectors"] += len(points)

        print(f"✅ Indexed {self.stats['complaints']} complaints")

    async def index_recalls(self):
        """Index NHTSA recalls."""
        recalls_file = DATA_DIR / "nhtsa" / "recalls.json"
        if not recalls_file.exists():
            print("⚠️ No recalls file found")
            return

        with open(recalls_file) as f:
            data = json.load(f)

        recalls = data.get("recalls", [])
        print(f"Indexing {len(recalls)} recalls...")

        points = []
        for recall in tqdm(recalls, desc="Recall Embeddings"):
            campaign = recall.get("campaign_number") or recall.get("NHTSACampaignNumber")
            summary = recall.get("summary") or recall.get("Summary", "")
            remedy = recall.get("remedy") or recall.get("Remedy", "")

            if not campaign or not summary:
                continue

            make = recall.get("make") or recall.get("Make", "")
            model = recall.get("model") or recall.get("Model", "")
            year = recall.get("model_year") or recall.get("ModelYear", "")
            component = recall.get("component") or recall.get("Component", "")

            # Create searchable text
            text = f"{make} {model} {year} RECALL - {component}: {summary[:300]}. Remedy: {remedy[:200]}"

            embedding = await self.get_embedding(text)
            if embedding:
                point_id = self.generate_id("recall", campaign)
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "type": "recall",
                        "campaign_number": campaign,
                        "make": make,
                        "model": model,
                        "year": str(year),
                        "component": component,
                        "summary": summary[:1000],
                        "remedy": remedy[:500],
                        "text": text
                    }
                ))
                self.stats["recalls"] += 1

            if len(points) >= 50:
                self.client.upsert(collection_name=COLLECTION_NAME, points=points)
                self.stats["total_vectors"] += len(points)
                points = []

            await asyncio.sleep(0.1)

        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            self.stats["total_vectors"] += len(points)

        print(f"✅ Indexed {self.stats['recalls']} recalls")

    def print_stats(self):
        """Print final statistics."""
        info = self.client.get_collection(COLLECTION_NAME)

        print("\n" + "=" * 60)
        print("QDRANT CLOUD STATISTICS")
        print("=" * 60)
        print(f"  Collection: {COLLECTION_NAME}")
        print(f"  Total vectors: {info.points_count:,}")
        print("-" * 60)
        print(f"  DTC codes indexed: {self.stats['dtc_codes']:,}")
        print(f"  Complaints indexed: {self.stats['complaints']:,}")
        print(f"  Recalls indexed: {self.stats['recalls']:,}")
        print("=" * 60)


async def main():
    print("=" * 60)
    print("INDEXING ALL DATA TO QDRANT CLOUD")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not set! Cannot generate embeddings.")
        return

    indexer = QdrantIndexer()

    try:
        await indexer.connect()

        # Index all data sources
        await indexer.index_dtc_codes()
        await indexer.index_recalls()
        await indexer.index_complaints(limit=5000)  # Limit to avoid rate limits

        indexer.print_stats()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await indexer.close()


if __name__ == "__main__":
    asyncio.run(main())
