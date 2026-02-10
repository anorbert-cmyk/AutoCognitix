#!/usr/bin/env python3
"""
Sprint 9 Qdrant Reindex.

Adds DTC, complaint, and EPA vehicle embeddings to existing Qdrant collection.
Uses SZTAKI-HLT/hubert-base-cc (768-dim) for Hungarian-optimized embeddings.

Data sources:
  1. DTC Codes: data/dtc_codes/all_codes_complete.json (~6,814 codes)
  2. NHTSA Complaints: data/nhtsa/complaints_flat/*.json (prioritized top ~50K)
  3. EPA Engine Specs: data/epa/engine_specs.json (~31K records, deduplicated)

Usage:
  QDRANT_URL=xxx QDRANT_API_KEY=xxx python scripts/sync_qdrant_sprint9.py --all
  QDRANT_URL=xxx QDRANT_API_KEY=xxx python scripts/sync_qdrant_sprint9.py --dtc
  QDRANT_URL=xxx QDRANT_API_KEY=xxx python scripts/sync_qdrant_sprint9.py --complaints
  QDRANT_URL=xxx QDRANT_API_KEY=xxx python scripts/sync_qdrant_sprint9.py --epa
  QDRANT_URL=xxx QDRANT_API_KEY=xxx python scripts/sync_qdrant_sprint9.py --reset
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

COLLECTION_NAME = "autocognitix"
EMBEDDING_DIM = 768
HUBERT_MODEL = "SZTAKI-HLT/hubert-base-cc"
BATCH_SIZE = 32  # Embedding batch
QDRANT_UPLOAD_BATCH = 100  # Upload batch
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "qdrant_sprint9.json"

# Complaint sampling limits
MAX_COMPLAINTS = 50_000
COMPLAINT_CHUNK_SIZE = 5_000

# Top 30 makes for recent complaint prioritization
TOP_30_MAKES: Set[str] = {
    "FORD",
    "CHEVROLET",
    "TOYOTA",
    "HONDA",
    "JEEP",
    "DODGE",
    "NISSAN",
    "HYUNDAI",
    "KIA",
    "CHRYSLER",
    "GMC",
    "VOLKSWAGEN",
    "BMW",
    "SUBARU",
    "RAM",
    "PONTIAC",
    "MAZDA",
    "TESLA",
    "BUICK",
    "MERCEDES-BENZ",
    "SATURN",
    "CADILLAC",
    "ACURA",
    "MERCURY",
    "AUDI",
    "LEXUS",
    "LINCOLN",
    "MERCEDES BENZ",
    "VOLVO",
    "MITSUBISHI",
}

# Complaint flat file order (newest first for priority)
COMPLAINT_FILES_ORDER = [
    "2025-2026.json",
    "2020-2024.json",
    "2015-2019.json",
    "2010-2014.json",
    "2005-2009.json",
    "2000-2004.json",
]


# ---------------------------------------------------------------------------
# Utility: stable ID generation
# ---------------------------------------------------------------------------
def generate_stable_id(content: str) -> int:
    """Generate a stable numeric ID from content using MD5 hash."""
    hash_bytes = hashlib.md5(content.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# HuBERT embedding service (standalone, no app deps)
# ---------------------------------------------------------------------------
class StandaloneHuBERTService:
    """Standalone HuBERT embedding service without app dependencies."""

    def __init__(self) -> None:
        self.device = self._detect_device()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self._optimal_batch_size = self._get_optimal_batch_size()
        print(f"Device: {self.device}, Batch size: {self._optimal_batch_size}")

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _get_optimal_batch_size(self) -> int:
        if self.device.type == "cuda":
            return 64
        elif self.device.type == "mps":
            return 32
        return 16

    def warmup(self) -> None:
        """Load model and warm up with a test embedding."""
        print(f"Loading HuBERT model: {HUBERT_MODEL}...")
        self.tokenizer = AutoTokenizer.from_pretrained(HUBERT_MODEL, use_fast=True)

        use_fp16 = self.device.type == "cuda"
        self.model = AutoModel.from_pretrained(
            HUBERT_MODEL,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        # Warmup inference
        _ = self.embed("warmup test")
        print("HuBERT model loaded and ready")

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with OOM fallback."""
        if not texts:
            return []

        embeddings: List[List[float]] = []
        current_batch_size = self._optimal_batch_size

        i = 0
        while i < len(texts):
            batch = texts[i : i + current_batch_size]
            try:
                batch_embeddings = self._embed_single_batch(batch)
                embeddings.extend(batch_embeddings)
                i += current_batch_size
                # Restore batch size if it was reduced
                if current_batch_size < self._optimal_batch_size:
                    current_batch_size = min(
                        current_batch_size * 2, self._optimal_batch_size
                    )
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # OOM: halve batch size and retry
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    print(
                        f"\n  OOM detected, reducing batch size to {current_batch_size}"
                    )
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    # Single item still OOM - skip it
                    print(f"\n  OOM on single item (len={len(batch[0])}), skipping")
                    embeddings.append([0.0] * EMBEDDING_DIM)
                    i += 1

        # Cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return embeddings

    def _embed_single_batch(self, batch: List[str]) -> List[List[float]]:
        """Embed a single batch of texts. May raise RuntimeError on OOM."""
        inputs = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state

            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask

            # L2 normalize
            batch_embeddings = torch.nn.functional.normalize(
                batch_embeddings, p=2, dim=1
            )

            return batch_embeddings.cpu().numpy().tolist()


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------
class CheckpointManager:
    """Manages checkpoint state for resumable indexing."""

    def __init__(self, checkpoint_file: Path) -> None:
        self.checkpoint_file = checkpoint_file
        self.state = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: corrupt checkpoint, starting fresh: {exc}")
        return self._default_state()

    @staticmethod
    def _default_state() -> dict:
        return {
            "dtc_done": False,
            "dtc_last_idx": 0,
            "complaints_done": False,
            "complaints_last_file": "",
            "complaints_last_idx": 0,
            "complaints_total_indexed": 0,
            "epa_done": False,
            "epa_last_idx": 0,
            "last_updated": None,
        }

    def save(self) -> None:
        self.state["last_updated"] = datetime.now().isoformat()
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.checkpoint_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2)
        tmp.replace(self.checkpoint_file)

    def reset(self) -> None:
        self.state = self._default_state()
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        print("Checkpoint reset.")


# ---------------------------------------------------------------------------
# Complaint priority sampler
# ---------------------------------------------------------------------------
def _complaint_priority_score(c: dict) -> Tuple[int, int, int, int, int]:
    """Return a sort key tuple for complaint priority (higher = more important).

    Priority order:
      1. deaths > 0
      2. injuries > 0
      3. fire = true
      4. crash = true
      5. Recent (2020-2026) from top 30 makes
    """
    deaths = int(c.get("deaths") or 0)
    injuries = int(c.get("injuries") or 0)
    fire = 1 if c.get("fire") else 0
    crash = 1 if c.get("crash") else 0
    year = int(c.get("model_year") or 0)
    make_upper = str(c.get("make", "")).upper()
    recency = 1 if (year >= 2020 and make_upper in TOP_30_MAKES) else 0

    return (
        1 if deaths > 0 else 0,
        1 if injuries > 0 else 0,
        fire,
        crash,
        recency,
    )


def load_prioritized_complaints(
    complaints_dir: Path,
    limit: int = MAX_COMPLAINTS,
) -> List[dict]:
    """Load and prioritize complaints from flat files, up to *limit*."""
    all_complaints: List[dict] = []

    for fname in COMPLAINT_FILES_ORDER:
        fpath = complaints_dir / fname
        if not fpath.exists():
            print(f"  Skipping missing file: {fpath.name}")
            continue

        print(f"  Loading {fpath.name}...")
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  Error loading {fpath.name}: {exc}")
            continue

        records = data.get("complaints", [])
        print(f"    -> {len(records):,} records")
        all_complaints.extend(records)

    total_loaded = len(all_complaints)
    print(f"  Total complaints loaded: {total_loaded:,}")

    if total_loaded <= limit:
        print(f"  All {total_loaded:,} complaints fit within limit of {limit:,}")
        return all_complaints

    # Deduplicate by odi_number (some complaints appear in multiple components)
    seen_odi: Dict[str, dict] = {}
    for c in all_complaints:
        odi = str(c.get("odi_number", ""))
        if odi and odi not in seen_odi:
            seen_odi[odi] = c
    deduped = list(seen_odi.values())
    print(f"  After deduplication by odi_number: {len(deduped):,}")

    # Sort by priority (highest first)
    deduped.sort(key=_complaint_priority_score, reverse=True)

    sampled = deduped[:limit]
    print(f"  Sampled top {len(sampled):,} by priority")

    # Log priority breakdown
    n_deaths = sum(1 for c in sampled if int(c.get("deaths") or 0) > 0)
    n_injuries = sum(1 for c in sampled if int(c.get("injuries") or 0) > 0)
    n_fire = sum(1 for c in sampled if c.get("fire"))
    n_crash = sum(1 for c in sampled if c.get("crash"))
    print(
        f"  Priority breakdown: deaths={n_deaths:,}, injuries={n_injuries:,}, "
        f"fire={n_fire:,}, crash={n_crash:,}"
    )

    return sampled


# ---------------------------------------------------------------------------
# EPA deduplication
# ---------------------------------------------------------------------------
def load_deduplicated_epa(epa_file: Path) -> List[dict]:
    """Load EPA engine specs deduplicated by (make, model, engine)."""
    print(f"  Loading {epa_file.name}...")
    with open(epa_file) as f:
        records = json.load(f)

    print(f"  Total EPA records: {len(records):,}")

    seen: Dict[str, dict] = {}
    for rec in records:
        make = str(rec.get("make", "")).strip()
        model = str(rec.get("model", "")).strip()
        engine = str(rec.get("engine", "")).strip()
        key = f"{make}|{model}|{engine}"
        if key not in seen:
            seen[key] = rec

    deduped = list(seen.values())
    print(f"  Unique (make, model, engine) combinations: {len(deduped):,}")
    return deduped


# ---------------------------------------------------------------------------
# Main indexer
# ---------------------------------------------------------------------------
class Sprint9QdrantIndexer:
    """Indexes Sprint 9 data sources to Qdrant with HuBERT embeddings."""

    def __init__(self) -> None:
        self.qdrant: Optional[QdrantClient] = None
        self.embedding_service: Optional[StandaloneHuBERTService] = None
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)
        self.stats: Dict[str, int] = {
            "dtc_indexed": 0,
            "complaints_indexed": 0,
            "epa_indexed": 0,
            "errors": 0,
            "skipped_existing": 0,
        }
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> None:
        """Connect to Qdrant and initialize embedding service."""
        print("Connecting to Qdrant Cloud...")
        self.qdrant = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60,
        )

        # Ensure collection exists (do NOT recreate)
        collections = self.qdrant.get_collections()
        coll_names = [c.name for c in collections.collections]
        if COLLECTION_NAME not in coll_names:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            info = self.qdrant.get_collection(COLLECTION_NAME)
            print(f"Connected to existing collection (points: {info.points_count:,})")

        # Initialize standalone embedding service
        self.embedding_service = StandaloneHuBERTService()
        self.embedding_service.warmup()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _upload_points(self, points: List[PointStruct]) -> None:
        """Upload points to Qdrant in batches with retry."""
        for i in range(0, len(points), QDRANT_UPLOAD_BATCH):
            batch = points[i : i + QDRANT_UPLOAD_BATCH]
            retries = 3
            for attempt in range(retries):
                try:
                    self.qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
                    break
                except Exception as exc:
                    if attempt < retries - 1:
                        wait = 2 ** (attempt + 1)
                        print(
                            f"\n  Upload error (attempt {attempt + 1}/{retries}): "
                            f"{str(exc)[:80]}. Retrying in {wait}s..."
                        )
                        time.sleep(wait)
                    else:
                        print(f"\n  Upload failed after {retries} attempts: {exc}")
                        raise

    # ------------------------------------------------------------------
    # DTC Codes
    # ------------------------------------------------------------------
    def index_dtc_codes(self) -> None:
        """Index DTC codes from all_codes_complete.json."""
        if self.checkpoint.state.get("dtc_done"):
            print("\nDTC codes already indexed (checkpoint). Skipping.")
            return

        dtc_file = DATA_DIR / "dtc_codes" / "all_codes_complete.json"
        if not dtc_file.exists():
            print(f"\nDTC file not found: {dtc_file}")
            return

        print("\n" + "=" * 60)
        print("INDEXING DTC CODES")
        print("=" * 60)

        with open(dtc_file) as f:
            data = json.load(f)

        codes = data.get("codes", [])
        total = len(codes)
        start_idx = self.checkpoint.state.get("dtc_last_idx", 0)
        print(f"Total DTC codes: {total:,}, resuming from index {start_idx}")

        if start_idx >= total:
            self.checkpoint.state["dtc_done"] = True
            self.checkpoint.save()
            print("All DTC codes already processed.")
            return

        remaining = codes[start_idx:]
        points: List[PointStruct] = []

        with tqdm(total=len(remaining), desc="DTC Embeddings", initial=0) as pbar:
            for batch_start in range(0, len(remaining), BATCH_SIZE):
                batch_codes = remaining[batch_start : batch_start + BATCH_SIZE]

                # Build text for each code
                texts: List[str] = []
                for c in batch_codes:
                    code = c.get("code", "")
                    desc = c.get("description_hu") or c.get("description", "")
                    symptoms = c.get("symptoms", [])[:5]
                    causes = c.get("possible_causes", [])[:5]

                    text = f"{code}: {desc}"
                    if symptoms:
                        text += f". Tunetek: {', '.join(str(s) for s in symptoms)}"
                    if causes:
                        text += (
                            f". Lehetseges okok: {', '.join(str(s) for s in causes)}"
                        )
                    texts.append(text)

                try:
                    embeddings = self.embedding_service.embed_batch(texts)
                except Exception as exc:
                    print(f"\n  Embedding error: {str(exc)[:100]}")
                    self.stats["errors"] += len(batch_codes)
                    pbar.update(len(batch_codes))
                    continue

                for c, embedding in zip(batch_codes, embeddings):
                    code = c.get("code", "")
                    point_id = generate_stable_id(f"dtc_{code}")

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "type": "dtc",
                                "source": "all_codes_complete",
                                "code": code,
                                "description": (
                                    c.get("description_hu") or c.get("description", "")
                                ),
                                "description_en": c.get("description", ""),
                                "category": c.get("category", ""),
                                "severity": c.get("severity", ""),
                                "system": c.get("system", ""),
                                "is_generic": c.get("is_generic", False),
                            },
                        )
                    )
                    self.stats["dtc_indexed"] += 1

                pbar.update(len(batch_codes))

                # Flush to Qdrant periodically
                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    points = []
                    self.checkpoint.state["dtc_last_idx"] = (
                        start_idx + batch_start + len(batch_codes)
                    )
                    self.checkpoint.save()

        # Flush remaining
        if points:
            self._upload_points(points)
            points = []

        self.checkpoint.state["dtc_done"] = True
        self.checkpoint.state["dtc_last_idx"] = total
        self.checkpoint.save()
        print(f"DTC indexing complete: {self.stats['dtc_indexed']:,} codes")

        # Free memory
        gc.collect()

    # ------------------------------------------------------------------
    # NHTSA Complaints
    # ------------------------------------------------------------------
    def index_complaints(self) -> None:
        """Index prioritized NHTSA complaints."""
        if self.checkpoint.state.get("complaints_done"):
            print("\nComplaints already indexed (checkpoint). Skipping.")
            return

        complaints_dir = DATA_DIR / "nhtsa" / "complaints_flat"
        if not complaints_dir.exists():
            print(f"\nComplaints directory not found: {complaints_dir}")
            return

        print("\n" + "=" * 60)
        print("INDEXING NHTSA COMPLAINTS")
        print("=" * 60)

        # Check for pre-sampled file first (from Agent 5)
        sampled_file = complaints_dir / "sampled_50k_embedding.json"
        if sampled_file.exists():
            print("Found pre-sampled complaint file.")
            with open(sampled_file) as f:
                complaints = json.load(f)
            if isinstance(complaints, dict) and "complaints" in complaints:
                complaints = complaints["complaints"]
        else:
            print("No pre-sampled file. Loading and prioritizing from flat files...")
            complaints = load_prioritized_complaints(complaints_dir)

        total = len(complaints)
        start_idx = self.checkpoint.state.get("complaints_last_idx", 0)
        print(f"Total complaints to index: {total:,}, resuming from index {start_idx}")

        if start_idx >= total:
            self.checkpoint.state["complaints_done"] = True
            self.checkpoint.save()
            print("All complaints already processed.")
            return

        remaining = complaints[start_idx:]
        points: List[PointStruct] = []

        # Process in chunks to manage memory
        with tqdm(total=len(remaining), desc="Complaint Embeddings", initial=0) as pbar:
            for chunk_start in range(0, len(remaining), COMPLAINT_CHUNK_SIZE):
                chunk = remaining[chunk_start : chunk_start + COMPLAINT_CHUNK_SIZE]

                for batch_start in range(0, len(chunk), BATCH_SIZE):
                    batch = chunk[batch_start : batch_start + BATCH_SIZE]

                    texts: List[str] = []
                    for c in batch:
                        make = str(c.get("make", "")).strip()
                        model = str(c.get("model", "")).strip()
                        year = c.get("model_year", "")
                        component = str(c.get("component", "")).strip()
                        summary = str(c.get("summary", ""))[:500]

                        text = f"{make} {model} {year} - {component}: {summary}"
                        texts.append(text)

                    try:
                        embeddings = self.embedding_service.embed_batch(texts)
                    except Exception as exc:
                        print(f"\n  Embedding error: {str(exc)[:100]}")
                        self.stats["errors"] += len(batch)
                        pbar.update(len(batch))
                        continue

                    for c, embedding in zip(batch, embeddings):
                        odi = str(c.get("odi_number", ""))
                        if not odi:
                            self.stats["errors"] += 1
                            continue

                        point_id = generate_stable_id(f"complaint_{odi}")

                        points.append(
                            PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "type": "complaint",
                                    "source": "nhtsa_flat",
                                    "odi_number": odi,
                                    "make": str(c.get("make", "")),
                                    "model": str(c.get("model", "")),
                                    "model_year": int(c.get("model_year") or 0),
                                    "component": str(c.get("component", "")),
                                    "crash": bool(c.get("crash")),
                                    "fire": bool(c.get("fire")),
                                    "injuries": int(c.get("injuries") or 0),
                                    "deaths": int(c.get("deaths") or 0),
                                },
                            )
                        )
                        self.stats["complaints_indexed"] += 1

                    pbar.update(len(batch))

                    # Flush periodically
                    if len(points) >= QDRANT_UPLOAD_BATCH:
                        self._upload_points(points)
                        points = []
                        global_idx = start_idx + chunk_start + batch_start + len(batch)
                        self.checkpoint.state["complaints_last_idx"] = global_idx
                        self.checkpoint.state["complaints_total_indexed"] = self.stats[
                            "complaints_indexed"
                        ]
                        self.checkpoint.save()

                # GC between chunks
                gc.collect()

        # Flush remaining
        if points:
            self._upload_points(points)
            points = []

        self.checkpoint.state["complaints_done"] = True
        self.checkpoint.state["complaints_last_idx"] = total
        self.checkpoint.state["complaints_total_indexed"] = self.stats[
            "complaints_indexed"
        ]
        self.checkpoint.save()
        print(
            f"Complaint indexing complete: "
            f"{self.stats['complaints_indexed']:,} complaints"
        )

        # Free memory
        del complaints
        gc.collect()

    # ------------------------------------------------------------------
    # EPA Engine Specs
    # ------------------------------------------------------------------
    def index_epa_specs(self) -> None:
        """Index deduplicated EPA engine specs."""
        if self.checkpoint.state.get("epa_done"):
            print("\nEPA specs already indexed (checkpoint). Skipping.")
            return

        epa_file = DATA_DIR / "epa" / "engine_specs.json"
        if not epa_file.exists():
            print(f"\nEPA file not found: {epa_file}")
            return

        print("\n" + "=" * 60)
        print("INDEXING EPA ENGINE SPECS")
        print("=" * 60)

        specs = load_deduplicated_epa(epa_file)
        total = len(specs)
        start_idx = self.checkpoint.state.get("epa_last_idx", 0)
        print(f"Total unique specs: {total:,}, resuming from index {start_idx}")

        if start_idx >= total:
            self.checkpoint.state["epa_done"] = True
            self.checkpoint.save()
            print("All EPA specs already processed.")
            return

        remaining = specs[start_idx:]
        points: List[PointStruct] = []

        with tqdm(total=len(remaining), desc="EPA Embeddings", initial=0) as pbar:
            for batch_start in range(0, len(remaining), BATCH_SIZE):
                batch = remaining[batch_start : batch_start + BATCH_SIZE]

                texts: List[str] = []
                for rec in batch:
                    make = str(rec.get("make", "")).strip()
                    model = str(rec.get("model", "")).strip()
                    engine = str(rec.get("engine", "")).strip()
                    fuel_cat = str(rec.get("fuel_category", "")).strip()
                    mpg = rec.get("mpg_combined", "")
                    drive = str(rec.get("drive", "")).strip()

                    text = f"{make} {model}: {engine}, {fuel_cat}"
                    if mpg:
                        text += f", {mpg} MPG"
                    if drive:
                        text += f", {drive}"
                    texts.append(text)

                try:
                    embeddings = self.embedding_service.embed_batch(texts)
                except Exception as exc:
                    print(f"\n  Embedding error: {str(exc)[:100]}")
                    self.stats["errors"] += len(batch)
                    pbar.update(len(batch))
                    continue

                for rec, embedding in zip(batch, embeddings):
                    make = str(rec.get("make", "")).strip()
                    model = str(rec.get("model", "")).strip()
                    engine = str(rec.get("engine", "")).strip()
                    point_id = generate_stable_id(f"epa_{make}_{model}_{engine}")

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "type": "epa_vehicle",
                                "source": "epa_engine_specs",
                                "make": make,
                                "model": model,
                                "engine": engine,
                                "cylinders": rec.get("cylinders"),
                                "displacement": rec.get("displacement"),
                                "fuel_type": str(rec.get("fuel_type", "")),
                                "fuel_category": str(rec.get("fuel_category", "")),
                                "mpg_combined": rec.get("mpg_combined"),
                                "drive": str(rec.get("drive", "")),
                                "has_turbo": bool(rec.get("has_turbo")),
                                "has_supercharger": bool(rec.get("has_supercharger")),
                                "co2_grams_per_mile": rec.get("co2_grams_per_mile"),
                            },
                        )
                    )
                    self.stats["epa_indexed"] += 1

                pbar.update(len(batch))

                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    points = []
                    self.checkpoint.state["epa_last_idx"] = (
                        start_idx + batch_start + len(batch)
                    )
                    self.checkpoint.save()

        # Flush remaining
        if points:
            self._upload_points(points)
            points = []

        self.checkpoint.state["epa_done"] = True
        self.checkpoint.state["epa_last_idx"] = total
        self.checkpoint.save()
        print(f"EPA indexing complete: {self.stats['epa_indexed']:,} specs")

        # Free memory
        gc.collect()

    # ------------------------------------------------------------------
    # Semantic search verification
    # ------------------------------------------------------------------
    def verify_semantic_search(self) -> None:
        """Run sample queries to verify embeddings work correctly."""
        print("\n" + "=" * 60)
        print("SEMANTIC SEARCH VERIFICATION")
        print("=" * 60)

        test_queries = [
            ("motor gyujtas hiba", "dtc"),
            ("engine misfire cylinder", "dtc"),
            ("airbag deployment failure crash", "complaint"),
            ("Toyota Camry brake problem", "complaint"),
            ("Honda Civic 2.0L turbo fuel economy", "epa_vehicle"),
            ("Tesla electric motor battery", "epa_vehicle"),
            ("oxigen szenzor meghibasodas", "dtc"),
            ("fek rendszer nyomas csokkenese", "dtc"),
        ]

        for query, expected_type in test_queries:
            try:
                query_embedding = self.embedding_service.embed(query)
                results = self.qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=3,
                )

                print(f"\nQuery: '{query}' (expecting: {expected_type})")
                for idx, result in enumerate(results, 1):
                    payload = result.payload
                    ptype = payload.get("type", "unknown")
                    score = result.score

                    if ptype == "dtc":
                        desc = str(payload.get("description", ""))[:60]
                        code = payload.get("code", "")
                        print(
                            f"  {idx}. [{ptype}] {code} - {desc} (score: {score:.3f})"
                        )
                    elif ptype == "complaint":
                        make = payload.get("make", "")
                        model = payload.get("model", "")
                        comp = payload.get("component", "")
                        print(
                            f"  {idx}. [{ptype}] {make} {model} - {comp} "
                            f"(score: {score:.3f})"
                        )
                    elif ptype == "epa_vehicle":
                        make = payload.get("make", "")
                        model = payload.get("model", "")
                        eng = payload.get("engine", "")
                        print(
                            f"  {idx}. [{ptype}] {make} {model} {eng} "
                            f"(score: {score:.3f})"
                        )
                    else:
                        print(
                            f"  {idx}. [{ptype}] "
                            f"{str(payload)[:60]} (score: {score:.3f})"
                        )

            except Exception as exc:
                print(f"\nQuery '{query}' failed: {exc}")

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def run(
        self,
        do_dtc: bool = True,
        do_complaints: bool = True,
        do_epa: bool = True,
        do_verify: bool = True,
    ) -> None:
        """Run the indexing process for selected sections."""
        print("=" * 60)
        print("SPRINT 9 QDRANT REINDEX - HuBERT Embeddings")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        last = self.checkpoint.state.get("last_updated")
        if last:
            print(f"Last checkpoint: {last}")
        print(f"Sections: dtc={do_dtc}, complaints={do_complaints}, epa={do_epa}")
        print()

        self._start_time = time.time()

        try:
            self.connect()

            if do_dtc:
                self.index_dtc_codes()
            if do_complaints:
                self.index_complaints()
            if do_epa:
                self.index_epa_specs()

            if do_verify:
                self.verify_semantic_search()

            # Final stats
            elapsed = time.time() - self._start_time
            info = self.qdrant.get_collection(COLLECTION_NAME)

            print()
            print("=" * 60)
            print("INDEXING COMPLETE")
            print("=" * 60)
            print(f"DTC Codes indexed:   {self.stats['dtc_indexed']:,}")
            print(f"Complaints indexed:  {self.stats['complaints_indexed']:,}")
            print(f"EPA Specs indexed:   {self.stats['epa_indexed']:,}")
            print(f"Errors:              {self.stats['errors']:,}")
            print(f"Total in collection: {info.points_count:,}")
            print(f"Time elapsed:        {elapsed / 60:.1f} minutes")

            total_new = (
                self.stats["dtc_indexed"]
                + self.stats["complaints_indexed"]
                + self.stats["epa_indexed"]
            )
            if elapsed > 0 and total_new > 0:
                print(f"Speed:               {total_new / elapsed:.1f} items/sec")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved to checkpoint.")
            self.checkpoint.save()
            sys.exit(1)
        except Exception as exc:
            print(f"\nFatal error: {exc}")
            print("Progress saved to checkpoint. Run again to resume.")
            self.checkpoint.save()
            traceback.print_exc()
            raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sprint 9 Qdrant reindex with HuBERT embeddings"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Index all sections (DTC + Complaints + EPA)",
    )
    group.add_argument("--dtc", action="store_true", help="Index DTC codes only")
    group.add_argument(
        "--complaints", action="store_true", help="Index complaints only"
    )
    group.add_argument("--epa", action="store_true", help="Index EPA specs only")
    group.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and exit",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip semantic search verification",
    )
    args = parser.parse_args()

    # Validate env vars (unless just resetting)
    if not args.reset:
        if not QDRANT_URL or not QDRANT_API_KEY:
            print(
                "Error: QDRANT_URL and QDRANT_API_KEY environment "
                "variables are required"
            )
            print(
                "Usage: QDRANT_URL=xxx QDRANT_API_KEY=xxx "
                "python scripts/sync_qdrant_sprint9.py --all"
            )
            sys.exit(1)

    if args.reset:
        mgr = CheckpointManager(CHECKPOINT_FILE)
        mgr.reset()
        print("Done. Run again with --all, --dtc, --complaints, or --epa.")
        return

    # Default to --all if nothing specified
    do_all = args.all or not (args.dtc or args.complaints or args.epa)

    indexer = Sprint9QdrantIndexer()
    indexer.run(
        do_dtc=do_all or args.dtc,
        do_complaints=do_all or args.complaints,
        do_epa=do_all or args.epa,
        do_verify=not args.no_verify,
    )


if __name__ == "__main__":
    main()
