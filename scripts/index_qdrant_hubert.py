#!/usr/bin/env python3
"""
HuBERT-based Qdrant Indexer - LOCAL Embeddings (No API Rate Limits)

Uses the backend's HungarianEmbeddingService with huBERT model:
- SZTAKI-HLT/hubert-base-cc model (Hungarian optimized)
- 768-dimensional embeddings
- GPU acceleration (CUDA/MPS) when available
- No external API calls = No rate limits

Features:
- Batch processing (64 items/batch on GPU)
- Checkpoint saving for resume capability
- Progress tracking with tqdm
- Estimated time remaining
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

# Add backend to path for embedding service
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "backend"))

# Configuration - Load from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# Validate required credentials
if not QDRANT_URL or not QDRANT_API_KEY:
    print("Error: QDRANT_URL and QDRANT_API_KEY environment variables are required")
    print("Usage: QDRANT_URL=xxx QDRANT_API_KEY=xxx python index_qdrant_hubert.py")
    print("Or set them in your .env file")
    sys.exit(1)

COLLECTION_NAME = "autocognitix"
EMBEDDING_DIM = 768
BATCH_SIZE = 64  # Optimal for GPU, will be adjusted by service
QDRANT_UPLOAD_BATCH = 100

# Paths
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "qdrant_hubert_checkpoint.json"


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
            "last_updated": None,
            "total_indexed": 0
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
            self.state["total_indexed"] = self.state.get("total_indexed", 0) + 1


class HuBERTQdrantIndexer:
    """Qdrant indexer using local HuBERT embeddings."""

    def __init__(self):
        self.qdrant = None
        self.embedding_service = None
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)
        self.stats = {
            "dtc": 0,
            "complaints": 0,
            "recalls": 0,
            "errors": 0,
            "start_time": None
        }

    def connect(self):
        """Connect to Qdrant and initialize embedding service."""
        print("Connecting to Qdrant Cloud...")
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

        # Initialize embedding service
        print("Loading HuBERT embedding model...")
        from app.services.embedding_service import get_embedding_service
        self.embedding_service = get_embedding_service()
        self.embedding_service.warmup()
        print(f"✅ HuBERT ready: device={self.embedding_service.device}, "
              f"batch_size={self.embedding_service._optimal_batch_size}")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        return self.embedding_service.embed_batch(
            texts,
            preprocess=False,  # Keep original text for better matching
            use_cache=False    # Don't use Redis cache in batch mode
        )

    def _upload_points(self, points: list[PointStruct]):
        """Upload points to Qdrant in batches."""
        for i in range(0, len(points), QDRANT_UPLOAD_BATCH):
            batch = points[i:i + QDRANT_UPLOAD_BATCH]
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    def index_dtc_codes(self):
        """Index DTC codes with HuBERT embeddings."""
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

        print(f"\n📊 Indexing {len(unique_codes)} DTC codes...")

        # Prepare texts for batch embedding
        texts = []
        code_data = []
        for c in unique_codes:
            code = c.get("code", "")
            text = f"{code}: {c.get('description', '')} - {c.get('category', '')} {c.get('subcategory', '')}"
            texts.append(text)
            code_data.append(c)

        # Process in batches
        points = []
        with tqdm(total=len(texts), desc="DTC Embeddings") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_codes = code_data[i:i + BATCH_SIZE]

                try:
                    embeddings = self._embed_batch(batch_texts)

                    for j, (c, embedding) in enumerate(zip(batch_codes, embeddings)):
                        code = c.get("code", "")
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

                    pbar.update(len(batch_texts))

                except Exception as e:
                    print(f"\n  ⚠️ Batch error: {str(e)[:100]}")
                    self.stats["errors"] += len(batch_texts)
                    pbar.update(len(batch_texts))

                # Upload in batches
                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    self.checkpoint.save()
                    points = []

        # Upload remaining
        if points:
            self._upload_points(points)
            self.checkpoint.save()

        print(f"✅ Indexed {self.stats['dtc']} DTC codes")

    def index_complaints(self, limit: int = 50000):
        """Index NHTSA complaints with HuBERT embeddings."""
        complaints_dir = DATA_DIR / "nhtsa" / "complaints"
        if not complaints_dir.exists():
            print(f"⚠️  Complaints directory not found: {complaints_dir}")
            return

        # Collect complaints from all files
        all_complaints = []
        for file_path in sorted(complaints_dir.glob("complaints_*.json"), reverse=True):
            with open(file_path) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    complaints = data.get("complaints", [])
                else:
                    complaints = data
                all_complaints.extend(complaints)
                if len(all_complaints) >= limit * 2:
                    break

        # Filter already indexed
        to_index = []
        for c in all_complaints:
            # Support both NHTSA original format and normalized format
            odi_id = str(c.get("odi_number", c.get("ODI_ID", c.get("CMPLID", ""))))
            if odi_id and not self.checkpoint.is_indexed("complaints", odi_id):
                to_index.append(c)
                if len(to_index) >= limit:
                    break

        if not to_index:
            print("⏭️  All complaints already indexed, skipping...")
            return

        print(f"\n📊 Indexing {len(to_index)} complaints...")

        # Prepare texts
        texts = []
        complaint_data = []
        for c in to_index:
            # Support both NHTSA original format and normalized format
            make = c.get("make", c.get("MAKETXT", ""))
            model = c.get("model", c.get("MODELTXT", ""))
            year = str(c.get("model_year", c.get("YEARTXT", c.get("year", ""))))
            component = c.get("component", c.get("COMPNAME", ""))
            description = c.get("summary", c.get("CDESCR", c.get("description", "")))

            text = f"{make} {model} {year} - {component}: {description}"
            texts.append(text[:8000])  # Truncate very long texts
            complaint_data.append(c)

        # Process in batches
        points = []
        with tqdm(total=len(texts), desc="Complaint Embeddings") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_complaints = complaint_data[i:i + BATCH_SIZE]

                try:
                    embeddings = self._embed_batch(batch_texts)

                    for c, embedding in zip(batch_complaints, embeddings):
                        # Support both NHTSA original format and normalized format
                        odi_id = str(c.get("odi_number", c.get("ODI_ID", c.get("CMPLID", ""))))
                        make = c.get("make", c.get("MAKETXT", ""))
                        model = c.get("model", c.get("MODELTXT", ""))
                        year = str(c.get("model_year", c.get("YEARTXT", c.get("year", ""))))
                        component = c.get("component", c.get("COMPNAME", ""))
                        description = c.get("summary", c.get("CDESCR", c.get("description", "")))

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

                    pbar.update(len(batch_texts))

                except Exception as e:
                    print(f"\n  ⚠️ Batch error: {str(e)[:100]}")
                    self.stats["errors"] += len(batch_texts)
                    pbar.update(len(batch_texts))

                # Upload and save checkpoint periodically
                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    self.checkpoint.save()
                    points = []

        # Upload remaining
        if points:
            self._upload_points(points)
            self.checkpoint.save()

        print(f"✅ Indexed {self.stats['complaints']} complaints")

    def index_recalls(self):
        """Index NHTSA recalls with HuBERT embeddings."""
        recalls_dir = DATA_DIR / "nhtsa" / "recalls"
        if not recalls_dir.exists():
            print(f"⚠️  Recalls directory not found: {recalls_dir}")
            return

        # Collect all recalls (recursive search for JSON files)
        all_recalls = []
        for file_path in recalls_dir.rglob("*.json"):
            if file_path.name == "import_progress.json":
                continue
            try:
                with open(file_path) as f:
                    data = json.load(f)
                    recalls = data if isinstance(data, list) else data.get("recalls", data.get("Results", []))
                    if isinstance(recalls, list):
                        all_recalls.extend(recalls)
            except Exception as e:
                print(f"  ⚠️ Error reading {file_path.name}: {e}")

        # Filter already indexed
        to_index = []
        for r in all_recalls:
            campaign = r.get("NHTSACampaignNumber", r.get("campaign_number", ""))
            if campaign and not self.checkpoint.is_indexed("recalls", campaign):
                to_index.append(r)

        if not to_index:
            print("⏭️  All recalls already indexed, skipping...")
            return

        print(f"\n📊 Indexing {len(to_index)} recalls...")

        # Prepare texts
        texts = []
        recall_data = []
        for r in to_index:
            make = r.get("Make", r.get("make", ""))
            model = r.get("Model", r.get("model", ""))
            year = str(r.get("ModelYear", r.get("year", "")))
            component = r.get("Component", r.get("component", ""))
            summary = r.get("Summary", r.get("summary", "")) or ""
            consequence = r.get("Consequence", r.get("consequence", "")) or ""
            remedy = r.get("Remedy", r.get("remedy", "")) or ""

            text = f"{make} {model} {year} - {component}. {summary} Consequence: {consequence} Remedy: {remedy}"
            texts.append(text[:8000])
            recall_data.append(r)

        # Process in batches
        points = []
        with tqdm(total=len(texts), desc="Recall Embeddings") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_recalls = recall_data[i:i + BATCH_SIZE]

                try:
                    embeddings = self._embed_batch(batch_texts)

                    for r, embedding in zip(batch_recalls, embeddings):
                        campaign = r.get("NHTSACampaignNumber", r.get("campaign_number", ""))
                        make = r.get("Make", r.get("make", ""))
                        model = r.get("Model", r.get("model", ""))
                        year = str(r.get("ModelYear", r.get("year", "")))
                        component = r.get("Component", r.get("component", ""))
                        summary = r.get("Summary", r.get("summary", "")) or ""
                        consequence = r.get("Consequence", r.get("consequence", "")) or ""
                        remedy = r.get("Remedy", r.get("remedy", "")) or ""

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

                    pbar.update(len(batch_texts))

                except Exception as e:
                    print(f"\n  ⚠️ Batch error: {str(e)[:100]}")
                    self.stats["errors"] += len(batch_texts)
                    pbar.update(len(batch_texts))

                # Upload in batches
                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    self.checkpoint.save()
                    points = []

        # Upload remaining
        if points:
            self._upload_points(points)
            self.checkpoint.save()

        print(f"✅ Indexed {self.stats['recalls']} recalls")

    def run(self, resume: bool = True, complaint_limit: int = 50000):
        """Run the full indexing process."""
        print("=" * 60)
        print("HuBERT QDRANT INDEXER - LOCAL EMBEDDINGS")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Resume mode: {resume}")
        if resume and self.checkpoint.state["last_updated"]:
            print(f"Last checkpoint: {self.checkpoint.state['last_updated']}")
            print(f"Already indexed: {self.checkpoint.state.get('total_indexed', 0):,}")
        print()

        self.stats["start_time"] = time.time()

        try:
            self.connect()

            self.index_dtc_codes()
            self.index_recalls()
            self.index_complaints(limit=complaint_limit)

            # Final stats
            elapsed = time.time() - self.stats["start_time"]
            info = self.qdrant.get_collection(COLLECTION_NAME)

            print()
            print("=" * 60)
            print("INDEXING COMPLETE")
            print("=" * 60)
            print(f"DTC Codes: {self.stats['dtc']:,}")
            print(f"Recalls: {self.stats['recalls']:,}")
            print(f"Complaints: {self.stats['complaints']:,}")
            print(f"Errors: {self.stats['errors']:,}")
            print(f"Total in collection: {info.points_count:,}")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Speed: {(self.stats['dtc'] + self.stats['recalls'] + self.stats['complaints']) / elapsed:.1f} items/sec")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Progress saved to checkpoint. Run with --resume to continue.")
            import traceback
            traceback.print_exc()
            raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Index data to Qdrant using local HuBERT embeddings")
    parser.add_argument("--resume", "-r", action="store_true", default=True,
                        help="Resume from checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh, ignore checkpoint")
    parser.add_argument("--complaints", "-c", type=int, default=50000,
                        help="Max complaints to index (default: 50000)")
    args = parser.parse_args()

    resume = not args.fresh and args.resume

    indexer = HuBERTQdrantIndexer()
    indexer.run(resume=resume, complaint_limit=args.complaints)


if __name__ == "__main__":
    main()
