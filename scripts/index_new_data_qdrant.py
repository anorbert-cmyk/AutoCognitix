#!/usr/bin/env python3
"""
Enhanced Qdrant Indexer for New Data Sources.

Indexes the following new data to Qdrant with HuBERT embeddings:
1. python_obd_codes.json - 3,070 DTC codes
2. UCI automobile_specs.json - 205 vehicle specifications
3. OBDb signalsets - 732 vehicle signal definitions

Uses SZTAKI-HLT/hubert-base-cc model (768-dim embeddings).
Standalone script - no dependency on app config.
"""

import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

# Paths
try:
    SCRIPT_DIR = Path(__file__).parent
except NameError:
    SCRIPT_DIR = Path("/Users/norbertbarna/Library/CloudStorage/ProtonDrive-anorbert@proton.me-folder/Munka/AutoCognitix/scripts")
PROJECT_DIR = SCRIPT_DIR.parent

# Configuration from environment
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("Error: QDRANT_URL and QDRANT_API_KEY environment variables are required")
    print("Usage: QDRANT_URL=xxx QDRANT_API_KEY=xxx python index_new_data_qdrant.py")
    sys.exit(1)

COLLECTION_NAME = "autocognitix"
EMBEDDING_DIM = 768
HUBERT_MODEL = "SZTAKI-HLT/hubert-base-cc"
BATCH_SIZE = 32
QDRANT_UPLOAD_BATCH = 100

# Data paths
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "new_data_checkpoint.json"


def generate_stable_id(content: str) -> int:
    """Generate a stable numeric ID from content using MD5 hash."""
    hash_bytes = hashlib.md5(content.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder='big') & 0x7FFFFFFFFFFFFFFF


class StandaloneHuBERTService:
    """Standalone HuBERT embedding service without app dependencies."""

    def __init__(self):
        self.device = self._detect_device()
        self.tokenizer = None
        self.model = None
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

    def warmup(self):
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
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        embeddings = []

        # Process in optimal batches
        for i in range(0, len(texts), self._optimal_batch_size):
            batch = texts[i:i + self._optimal_batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over tokens
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state

                # Mask padding tokens
                mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask

                # Normalize
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

                embeddings.extend(batch_embeddings.cpu().numpy().tolist())

        # Cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return embeddings


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
            "python_obd_indexed": [],
            "uci_vehicles_indexed": [],
            "obdb_signals_indexed": [],
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


class NewDataQdrantIndexer:
    """Indexes new data sources to Qdrant with HuBERT embeddings."""

    def __init__(self):
        self.qdrant = None
        self.embedding_service = None
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)
        self.stats = {
            "python_obd": 0,
            "uci_vehicles": 0,
            "obdb_signals": 0,
            "errors": 0,
            "duplicates_skipped": 0,
            "start_time": None
        }

    def connect(self):
        """Connect to Qdrant and initialize embedding service."""
        print("Connecting to Qdrant Cloud...")
        self.qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Ensure collection exists
        collections = self.qdrant.get_collections()
        if COLLECTION_NAME not in [c.name for c in collections.collections]:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            info = self.qdrant.get_collection(COLLECTION_NAME)
            print(f"Connected to Qdrant (existing points: {info.points_count:,})")

        # Initialize standalone embedding service
        self.embedding_service = StandaloneHuBERTService()
        self.embedding_service.warmup()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        return self.embedding_service.embed_batch(texts)

    def _upload_points(self, points: List[PointStruct]):
        """Upload points to Qdrant in batches."""
        for i in range(0, len(points), QDRANT_UPLOAD_BATCH):
            batch = points[i:i + QDRANT_UPLOAD_BATCH]
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)

    def _check_existing_ids(self, ids: List[int]) -> set:
        """Check which IDs already exist in Qdrant to avoid duplicates."""
        existing = set()
        try:
            for i in range(0, len(ids), 100):
                batch_ids = ids[i:i + 100]
                results = self.qdrant.retrieve(
                    collection_name=COLLECTION_NAME,
                    ids=batch_ids,
                    with_payload=False,
                    with_vectors=False
                )
                existing.update(p.id for p in results)
        except Exception as e:
            print(f"Warning: Could not check existing IDs: {e}")
        return existing

    def index_python_obd_codes(self):
        """Index python-OBD DTC codes."""
        codes_file = DATA_DIR / "dtc_codes" / "python_obd_codes.json"
        if not codes_file.exists():
            print(f"File not found: {codes_file}")
            return

        with open(codes_file) as f:
            data = json.load(f)

        codes = data.get("codes", [])
        print(f"\nFound {len(codes)} python-OBD codes")

        # Filter already indexed
        to_index = []
        for c in codes:
            code = c.get("code", "")
            if code and not self.checkpoint.is_indexed("python_obd", code):
                to_index.append(c)

        if not to_index:
            print("All python-OBD codes already indexed, skipping...")
            return

        print(f"Indexing {len(to_index)} new python-OBD codes...")

        # Prepare texts
        texts = []
        code_data = []
        for c in to_index:
            code = c.get("code", "")
            desc_en = c.get("description_en", "")
            desc_hu = c.get("description_hu_hint", "") or ""
            category = c.get("category", "")
            severity = c.get("severity", "")

            text = f"{code}: {desc_en}"
            if desc_hu:
                text += f" | {desc_hu}"
            text += f" - Category: {category}, Severity: {severity}"

            texts.append(text)
            code_data.append(c)

        # Check for existing IDs
        ids_to_check = [generate_stable_id(f"python_obd_{c.get('code', '')}") for c in code_data]
        existing_ids = self._check_existing_ids(ids_to_check)
        print(f"Found {len(existing_ids)} existing entries, will skip duplicates")

        # Process in batches
        points = []
        with tqdm(total=len(texts), desc="Python-OBD Embeddings") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_codes = code_data[i:i + BATCH_SIZE]

                try:
                    embeddings = self._embed_batch(batch_texts)

                    for c, embedding in zip(batch_codes, embeddings):
                        code = c.get("code", "")
                        point_id = generate_stable_id(f"python_obd_{code}")

                        if point_id in existing_ids:
                            self.stats["duplicates_skipped"] += 1
                            self.checkpoint.mark_indexed("python_obd", code)
                            continue

                        points.append(PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "type": "dtc",
                                "source": "python_obd",
                                "code": code,
                                "description_en": c.get("description_en", ""),
                                "description_hu": c.get("description_hu_hint", ""),
                                "category": c.get("category", ""),
                                "severity": c.get("severity", ""),
                                "is_generic": c.get("is_generic", False)
                            }
                        ))
                        self.checkpoint.mark_indexed("python_obd", code)
                        self.stats["python_obd"] += 1

                    pbar.update(len(batch_texts))

                except Exception as e:
                    print(f"\n  Warning: Batch error: {str(e)[:100]}")
                    self.stats["errors"] += len(batch_texts)
                    pbar.update(len(batch_texts))

                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    self.checkpoint.save()
                    points = []

        if points:
            self._upload_points(points)
            self.checkpoint.save()

        print(f"Indexed {self.stats['python_obd']} python-OBD codes")

    def index_uci_vehicles(self):
        """Index UCI automobile specifications."""
        specs_file = DATA_DIR / "uci" / "automobile_specs.json"
        if not specs_file.exists():
            print(f"File not found: {specs_file}")
            return

        with open(specs_file) as f:
            data = json.load(f)

        vehicles = data.get("vehicles", [])
        print(f"\nFound {len(vehicles)} UCI vehicle specs")

        # Filter already indexed
        to_index = []
        for v in vehicles:
            vid = v.get("id", "")
            if vid and not self.checkpoint.is_indexed("uci_vehicles", vid):
                to_index.append(v)

        if not to_index:
            print("All UCI vehicles already indexed, skipping...")
            return

        print(f"Indexing {len(to_index)} new UCI vehicles...")

        # Prepare texts
        texts = []
        vehicle_data = []
        for v in to_index:
            make = v.get("make", "")
            body = v.get("body_style", "")
            fuel = v.get("fuel_type", "")
            engine = v.get("engine_type", "")
            cylinders = v.get("num_of_cylinders", "")
            hp = v.get("horsepower", "")
            drive = v.get("drive_wheels", "")

            text = f"{make} {body} vehicle. Fuel: {fuel}, Engine: {engine} {cylinders} cylinder"
            if hp:
                text += f", {hp} horsepower"
            text += f". Drive: {drive}"

            trans = v.get("translations", {}).get("hu", {})
            if trans:
                text += f" | HU: {trans.get('body_style', '')} {trans.get('fuel_type', '')}"

            texts.append(text)
            vehicle_data.append(v)

        # Check for existing IDs
        ids_to_check = [generate_stable_id(f"uci_{v.get('id', '')}") for v in vehicle_data]
        existing_ids = self._check_existing_ids(ids_to_check)

        # Process in batches
        points = []
        with tqdm(total=len(texts), desc="UCI Vehicle Embeddings") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_vehicles = vehicle_data[i:i + BATCH_SIZE]

                try:
                    embeddings = self._embed_batch(batch_texts)

                    for v, embedding in zip(batch_vehicles, embeddings):
                        vid = v.get("id", "")
                        point_id = generate_stable_id(f"uci_{vid}")

                        if point_id in existing_ids:
                            self.stats["duplicates_skipped"] += 1
                            self.checkpoint.mark_indexed("uci_vehicles", vid)
                            continue

                        points.append(PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "type": "vehicle_spec",
                                "source": "uci",
                                "vehicle_id": vid,
                                "make": v.get("make", ""),
                                "body_style": v.get("body_style", ""),
                                "fuel_type": v.get("fuel_type", ""),
                                "engine_type": v.get("engine_type", ""),
                                "num_of_cylinders": v.get("num_of_cylinders"),
                                "horsepower": v.get("horsepower"),
                                "drive_wheels": v.get("drive_wheels", ""),
                                "engine_size": v.get("engine_size"),
                                "price": v.get("price")
                            }
                        ))
                        self.checkpoint.mark_indexed("uci_vehicles", vid)
                        self.stats["uci_vehicles"] += 1

                    pbar.update(len(batch_texts))

                except Exception as e:
                    print(f"\n  Warning: Batch error: {str(e)[:100]}")
                    self.stats["errors"] += len(batch_texts)
                    pbar.update(len(batch_texts))

                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    self.checkpoint.save()
                    points = []

        if points:
            self._upload_points(points)
            self.checkpoint.save()

        print(f"Indexed {self.stats['uci_vehicles']} UCI vehicle specs")

    def index_obdb_signals(self):
        """Index OBDb vehicle signal definitions."""
        signalsets_dir = DATA_DIR / "obdb" / "signalsets"
        if not signalsets_dir.exists():
            print(f"Directory not found: {signalsets_dir}")
            return

        signal_files = list(signalsets_dir.glob("*.json"))
        signal_files = [f for f in signal_files if not f.name.startswith(".")]

        print(f"\nFound {len(signal_files)} OBDb signal files")

        # Prepare all signals for indexing
        all_signals = []
        for file_path in signal_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                name_parts = file_path.stem.split("-", 1)
                make = name_parts[0] if name_parts else ""
                model = name_parts[1].replace("-", " ") if len(name_parts) > 1 else ""

                commands = data.get("commands", [])
                for cmd in commands:
                    signals = cmd.get("signals", [])
                    for sig in signals:
                        signal_id = sig.get("id", sig.get("name", ""))
                        if signal_id:
                            unique_key = f"{make}_{model}_{signal_id}"
                            if not self.checkpoint.is_indexed("obdb_signals", unique_key):
                                all_signals.append({
                                    "make": make,
                                    "model": model,
                                    "signal_id": signal_id,
                                    "signal_name": sig.get("name", ""),
                                    "description": sig.get("description", ""),
                                    "unit": sig.get("fmt", {}).get("unit", ""),
                                    "unique_key": unique_key
                                })

            except Exception as e:
                print(f"Warning: Error reading {file_path.name}: {e}")

        if not all_signals:
            print("All OBDb signals already indexed, skipping...")
            return

        print(f"Indexing {len(all_signals)} new OBDb signals...")

        # Prepare texts
        texts = []
        for sig in all_signals:
            text = f"{sig['make']} {sig['model']} - {sig['signal_name'] or sig['signal_id']}"
            if sig['description']:
                text += f": {sig['description']}"
            if sig['unit']:
                text += f" ({sig['unit']})"
            texts.append(text)

        # Check for existing IDs (sample)
        ids_to_check = [generate_stable_id(f"obdb_{s['unique_key']}") for s in all_signals[:1000]]
        existing_ids = self._check_existing_ids(ids_to_check)

        # Process in batches
        points = []
        with tqdm(total=len(texts), desc="OBDb Signal Embeddings") as pbar:
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_signals = all_signals[i:i + BATCH_SIZE]

                try:
                    embeddings = self._embed_batch(batch_texts)

                    for sig, embedding in zip(batch_signals, embeddings):
                        point_id = generate_stable_id(f"obdb_{sig['unique_key']}")

                        if point_id in existing_ids:
                            self.stats["duplicates_skipped"] += 1
                            self.checkpoint.mark_indexed("obdb_signals", sig["unique_key"])
                            continue

                        points.append(PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload={
                                "type": "vehicle_signal",
                                "source": "obdb",
                                "make": sig["make"],
                                "model": sig["model"],
                                "signal_id": sig["signal_id"],
                                "signal_name": sig["signal_name"],
                                "description": sig["description"][:500] if sig["description"] else "",
                                "unit": sig["unit"]
                            }
                        ))
                        self.checkpoint.mark_indexed("obdb_signals", sig["unique_key"])
                        self.stats["obdb_signals"] += 1

                    pbar.update(len(batch_texts))

                except Exception as e:
                    print(f"\n  Warning: Batch error: {str(e)[:100]}")
                    self.stats["errors"] += len(batch_texts)
                    pbar.update(len(batch_texts))

                if len(points) >= QDRANT_UPLOAD_BATCH:
                    self._upload_points(points)
                    self.checkpoint.save()
                    points = []

        if points:
            self._upload_points(points)
            self.checkpoint.save()

        print(f"Indexed {self.stats['obdb_signals']} OBDb signals")

    def verify_semantic_search(self):
        """Test semantic search to verify embeddings are useful."""
        print("\n" + "=" * 60)
        print("SEMANTIC SEARCH VERIFICATION")
        print("=" * 60)

        test_queries = [
            ("engine misfire cylinder", "dtc"),
            ("oxygen sensor malfunction", "dtc"),
            ("Tesla battery voltage", "vehicle_signal"),
            ("fuel pump pressure", "dtc"),
            ("convertible sports car", "vehicle_spec"),
            ("motor gyujtas hiba", "dtc"),  # Hungarian: engine ignition error
        ]

        for query, expected_type in test_queries:
            try:
                query_embedding = self.embedding_service.embed(query)

                results = self.qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_embedding,
                    limit=3
                )

                print(f"\nQuery: '{query}' (expecting: {expected_type})")
                for i, result in enumerate(results, 1):
                    payload = result.payload
                    ptype = payload.get("type", "unknown")
                    score = result.score
                    source = payload.get("source", "")

                    if ptype == "dtc":
                        desc = payload.get("description_en", payload.get("description", ""))[:50]
                        print(f"  {i}. [{ptype}/{source}] {payload.get('code', '')} - {desc}... (score: {score:.3f})")
                    elif ptype == "vehicle_spec":
                        print(f"  {i}. [{ptype}] {payload.get('make', '')} {payload.get('body_style', '')} (score: {score:.3f})")
                    elif ptype == "vehicle_signal":
                        print(f"  {i}. [{ptype}] {payload.get('make', '')} {payload.get('model', '')} - {payload.get('signal_name', '')[:30]} (score: {score:.3f})")
                    else:
                        print(f"  {i}. [{ptype}] {str(payload)[:50]}... (score: {score:.3f})")

            except Exception as e:
                print(f"\nQuery: '{query}' - Error: {e}")

    def run(self):
        """Run the full indexing process."""
        print("=" * 60)
        print("NEW DATA QDRANT INDEXER - HuBERT Embeddings")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        if self.checkpoint.state["last_updated"]:
            print(f"Last checkpoint: {self.checkpoint.state['last_updated']}")
            print(f"Already indexed: {self.checkpoint.state.get('total_indexed', 0):,}")
        print()

        self.stats["start_time"] = time.time()

        try:
            self.connect()

            self.index_python_obd_codes()
            self.index_uci_vehicles()
            self.index_obdb_signals()

            self.verify_semantic_search()

            # Final stats
            elapsed = time.time() - self.stats["start_time"]
            info = self.qdrant.get_collection(COLLECTION_NAME)

            print()
            print("=" * 60)
            print("INDEXING COMPLETE")
            print("=" * 60)
            print(f"Python-OBD Codes: {self.stats['python_obd']:,}")
            print(f"UCI Vehicles: {self.stats['uci_vehicles']:,}")
            print(f"OBDb Signals: {self.stats['obdb_signals']:,}")
            print(f"Duplicates Skipped: {self.stats['duplicates_skipped']:,}")
            print(f"Errors: {self.stats['errors']:,}")
            print(f"Total in collection: {info.points_count:,}")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")

            total_new = self.stats['python_obd'] + self.stats['uci_vehicles'] + self.stats['obdb_signals']
            if elapsed > 0 and total_new > 0:
                print(f"Speed: {total_new / elapsed:.1f} items/sec")

        except Exception as e:
            print(f"\nError: {e}")
            print("Progress saved to checkpoint. Run again to continue.")
            import traceback
            traceback.print_exc()
            raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Index new data to Qdrant using HuBERT embeddings")
    parser.add_argument("--fresh", action="store_true", help="Start fresh, ignore checkpoint")
    args = parser.parse_args()

    if args.fresh:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print("Checkpoint cleared, starting fresh")

    indexer = NewDataQdrantIndexer()
    indexer.run()


if __name__ == "__main__":
    main()
