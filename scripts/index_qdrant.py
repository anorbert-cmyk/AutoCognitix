#!/usr/bin/env python3
"""
Qdrant Vector Database Indexer for AutoCognitix.

This script indexes DTC codes and symptoms into Qdrant vector database
for semantic search capabilities using Hungarian huBERT embeddings.

Usage:
    python scripts/index_qdrant.py --dtc         # Index DTC codes only
    python scripts/index_qdrant.py --symptoms    # Index symptoms only
    python scripts/index_qdrant.py --all         # Index everything

Requirements:
    - Qdrant running on localhost:6333
    - huBERT model available for embeddings (SZTAKI-HLT/hubert-base-cc)
"""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Configuration - can be overridden by environment variables
# Qdrant Cloud support: if QDRANT_URL is set, use cloud; otherwise use local
QDRANT_URL = os.getenv("QDRANT_URL", "")  # e.g., https://xxx.cloud.qdrant.io
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
HUBERT_MODEL = os.getenv("HUBERT_MODEL", "SZTAKI-HLT/hubert-base-cc")
EMBEDDING_DIMENSION = 768  # huBERT output dimension

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Collection names - Hungarian versions with huBERT embeddings (768-dim)
DTC_COLLECTION = "dtc_embeddings_hu"
SYMPTOM_COLLECTION = "symptom_embeddings_hu"

# Legacy collection names (English, all-MiniLM-L6-v2, 384-dim)
DTC_COLLECTION_LEGACY = "dtc_embeddings"
SYMPTOM_COLLECTION_LEGACY = "symptom_embeddings"


class HuBERTEmbedder:
    """Handles Hungarian text embedding using huBERT model."""

    def __init__(self, model_name: str = HUBERT_MODEL):
        """Initialize the huBERT model."""
        self.model_name = model_name
        self.device = self._detect_device()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        logger.info(f"HuBERTEmbedder initialized with device: {self.device}")

    def _detect_device(self) -> torch.device:
        """Detect the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple MPS device detected")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device

    def _load_model(self) -> None:
        """Load huBERT model and tokenizer lazily."""
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading huBERT model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        logger.info("huBERT model loaded successfully")

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling to token embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()

        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Handle empty texts
            non_empty_indices = []
            non_empty_texts = []
            for idx, text in enumerate(batch_texts):
                if text and text.strip():
                    non_empty_indices.append(idx)
                    non_empty_texts.append(text)

            # Initialize batch embeddings with zeros
            batch_embeddings = [[0.0] * EMBEDDING_DIMENSION for _ in range(len(batch_texts))]

            if non_empty_texts:
                # Tokenize batch
                encoded = self._tokenizer(
                    non_empty_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Generate embeddings
                with torch.no_grad():
                    model_output = self._model(**encoded)

                # Apply mean pooling
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to list
                embeddings_list = embeddings.cpu().tolist()
                for orig_idx, embedding in zip(non_empty_indices, embeddings_list):
                    batch_embeddings[orig_idx] = embedding

            all_embeddings.extend(batch_embeddings)

        return all_embeddings


# Global embedder instance
_embedder: Optional[HuBERTEmbedder] = None


def get_embedder() -> HuBERTEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = HuBERTEmbedder()
    return _embedder


def embed_batch(texts: List[str], preprocess: bool = False) -> List[List[float]]:
    """Generate embeddings for multiple texts using huBERT."""
    return get_embedder().embed_batch(texts)


class QdrantIndexer:
    """Handles Qdrant vector database indexing operations."""

    def __init__(self):
        """Initialize the Qdrant client."""
        # Check if using Qdrant Cloud or local
        if QDRANT_URL:
            # Qdrant Cloud connection
            logger.info(f"Connecting to Qdrant Cloud: {QDRANT_URL}")
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
            )
            logger.info("Connected to Qdrant Cloud successfully")
        else:
            # Local Qdrant connection
            self.client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
            )
            logger.info(f"Connected to local Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        self.vector_size = EMBEDDING_DIMENSION

    def create_collection(self, collection_name: str, recreate: bool = False) -> None:
        """
        Create a collection in Qdrant.

        Args:
            collection_name: Name of the collection to create.
            recreate: If True, delete existing collection and recreate.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    self.client.delete_collection(collection_name=collection_name)
                else:
                    logger.info(f"Collection '{collection_name}' already exists, skipping creation")
                    return

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_size,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def upsert_batch(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert a batch of vectors into a collection.

        Args:
            collection_name: Target collection name.
            ids: List of unique IDs for the points.
            vectors: List of embedding vectors.
            payloads: List of metadata payloads.
        """
        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, id_)),
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(ids, vectors, payloads)
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"name": collection_name, "status": "not_found"}


def load_dtc_codes(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load DTC codes from JSON file.

    Args:
        file_path: Path to the DTC codes JSON file.

    Returns:
        List of DTC code dictionaries.
    """
    logger.info(f"Loading DTC codes from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"DTC codes file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    logger.info(f"Loaded {len(codes)} DTC codes")
    return codes


def extract_unique_symptoms(dtc_codes: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Extract unique symptoms from DTC codes.

    Args:
        dtc_codes: List of DTC code dictionaries.

    Returns:
        Dictionary mapping symptom text to related DTC codes.
    """
    symptom_to_codes: Dict[str, Set[str]] = {}

    for dtc in dtc_codes:
        code = dtc.get("code", "")
        symptoms = dtc.get("symptoms", [])

        for symptom in symptoms:
            symptom_normalized = symptom.strip()
            if symptom_normalized:
                if symptom_normalized not in symptom_to_codes:
                    symptom_to_codes[symptom_normalized] = set()
                symptom_to_codes[symptom_normalized].add(code)

    logger.info(f"Extracted {len(symptom_to_codes)} unique symptoms")
    return symptom_to_codes


def index_dtc_codes(
    indexer: QdrantIndexer,
    dtc_codes: List[Dict[str, Any]],
    batch_size: int = 10,
    recreate: bool = False,
) -> int:
    """
    Index DTC codes into Qdrant.

    Args:
        indexer: QdrantIndexer instance.
        dtc_codes: List of DTC codes to index.
        batch_size: Number of codes to process per batch.
        recreate: Whether to recreate the collection.

    Returns:
        Number of indexed codes.
    """
    logger.info("Starting DTC code indexing...")

    # Create collection
    indexer.create_collection(DTC_COLLECTION, recreate=recreate)

    # Prepare data for indexing
    ids: List[str] = []
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for dtc in dtc_codes:
        code = dtc.get("code", "")
        description_hu = dtc.get("description_hu", "")

        if not code or not description_hu:
            continue

        ids.append(f"dtc_{code}")
        texts.append(description_hu)
        payloads.append({
            "code": code,
            "description_hu": description_hu,
            "description_en": dtc.get("description_en", ""),
            "category": dtc.get("category", "unknown"),
            "severity": dtc.get("severity", "unknown"),
            "system": dtc.get("system", ""),
            "symptoms": dtc.get("symptoms", []),
            "possible_causes": dtc.get("possible_causes", []),
            "diagnostic_steps": dtc.get("diagnostic_steps", []),
            "related_codes": dtc.get("related_codes", []),
            "is_generic": dtc.get("is_generic", True),
        })

    if not ids:
        logger.warning("No valid DTC codes to index")
        return 0

    # Generate embeddings with progress bar
    logger.info(f"Generating embeddings for {len(texts)} DTC descriptions...")
    all_embeddings: List[List[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding DTC codes"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embed_batch(batch_texts, preprocess=False)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant with progress bar
    logger.info(f"Upserting {len(ids)} vectors to Qdrant...")

    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing DTC codes"):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = all_embeddings[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]

        indexer.upsert_batch(
            collection_name=DTC_COLLECTION,
            ids=batch_ids,
            vectors=batch_vectors,
            payloads=batch_payloads,
        )

    logger.info(f"Successfully indexed {len(ids)} DTC codes")
    return len(ids)


def index_symptoms(
    indexer: QdrantIndexer,
    dtc_codes: List[Dict[str, Any]],
    batch_size: int = 10,
    recreate: bool = False,
) -> int:
    """
    Index symptoms into Qdrant.

    Args:
        indexer: QdrantIndexer instance.
        dtc_codes: List of DTC codes to extract symptoms from.
        batch_size: Number of symptoms to process per batch.
        recreate: Whether to recreate the collection.

    Returns:
        Number of indexed symptoms.
    """
    logger.info("Starting symptom indexing...")

    # Create collection
    indexer.create_collection(SYMPTOM_COLLECTION, recreate=recreate)

    # Extract unique symptoms
    symptom_to_codes = extract_unique_symptoms(dtc_codes)

    if not symptom_to_codes:
        logger.warning("No symptoms to index")
        return 0

    # Prepare data for indexing
    ids: List[str] = []
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for symptom_text, related_codes in symptom_to_codes.items():
        # Create unique ID from symptom text
        symptom_id = f"symptom_{hash(symptom_text) % 10**10}"
        ids.append(symptom_id)
        texts.append(symptom_text)
        payloads.append({
            "symptom_text": symptom_text,
            "related_dtc_codes": sorted(list(related_codes)),
            "related_codes_count": len(related_codes),
        })

    # Generate embeddings with progress bar
    logger.info(f"Generating embeddings for {len(texts)} symptoms...")
    all_embeddings: List[List[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding symptoms"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embed_batch(batch_texts, preprocess=False)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant with progress bar
    logger.info(f"Upserting {len(ids)} symptom vectors to Qdrant...")

    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing symptoms"):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = all_embeddings[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]

        indexer.upsert_batch(
            collection_name=SYMPTOM_COLLECTION,
            ids=batch_ids,
            vectors=batch_vectors,
            payloads=batch_payloads,
        )

    logger.info(f"Successfully indexed {len(ids)} symptoms")
    return len(ids)


def print_summary(indexer: QdrantIndexer) -> None:
    """Print summary of indexed collections."""
    print("\n" + "=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)

    for collection_name in [DTC_COLLECTION, SYMPTOM_COLLECTION, DTC_COLLECTION_LEGACY, SYMPTOM_COLLECTION_LEGACY]:
        info = indexer.get_collection_info(collection_name)
        print(f"\nCollection: {info['name']}")
        if info.get("status") == "not_found":
            print("  Status: Not created")
        else:
            print(f"  Status: {info.get('status', 'unknown')}")
            print(f"  Points count: {info.get('points_count', 0)}")
            print(f"  Vectors count: {info.get('vectors_count', 0)}")

    print("\n" + "=" * 60)


def main():
    """Main entry point for the indexer."""
    parser = argparse.ArgumentParser(
        description="Index DTC codes and symptoms into Qdrant vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/index_qdrant.py --dtc              # Index only DTC codes
    python scripts/index_qdrant.py --symptoms         # Index only symptoms
    python scripts/index_qdrant.py --all              # Index everything
    python scripts/index_qdrant.py --all --recreate   # Recreate collections and reindex
        """,
    )

    parser.add_argument(
        "--dtc",
        action="store_true",
        help="Index DTC codes into dtc_embeddings collection",
    )
    parser.add_argument(
        "--symptoms",
        action="store_true",
        help="Index symptoms into symptom_embeddings collection",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Index both DTC codes and symptoms",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate collections before indexing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for embedding generation (default: 10)",
    )
    parser.add_argument(
        "--dtc-file",
        type=str,
        default=None,
        help="Path to DTC codes JSON file (default: data/dtc_codes/generic_codes.json)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not (args.dtc or args.symptoms or args.all):
        parser.print_help()
        print("\nError: Please specify at least one indexing option: --dtc, --symptoms, or --all")
        sys.exit(1)

    # Determine what to index
    index_dtc = args.dtc or args.all
    index_symptoms_flag = args.symptoms or args.all

    # Set up paths
    if args.dtc_file:
        dtc_file_path = Path(args.dtc_file)
    else:
        dtc_file_path = project_root / "data" / "dtc_codes" / "generic_codes.json"

    try:
        # Initialize indexer
        logger.info("Initializing Qdrant indexer...")
        indexer = QdrantIndexer()

        # Load DTC codes
        dtc_codes = load_dtc_codes(dtc_file_path)

        total_indexed = 0

        # Index DTC codes if requested
        if index_dtc:
            count = index_dtc_codes(
                indexer=indexer,
                dtc_codes=dtc_codes,
                batch_size=args.batch_size,
                recreate=args.recreate,
            )
            total_indexed += count

        # Index symptoms if requested
        if index_symptoms_flag:
            count = index_symptoms(
                indexer=indexer,
                dtc_codes=dtc_codes,
                batch_size=args.batch_size,
                recreate=args.recreate,
            )
            total_indexed += count

        # Print summary
        print_summary(indexer)

        logger.info(f"Total vectors indexed: {total_indexed}")
        logger.info("Indexing completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running on localhost:6333")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


if __name__ == "__main__":
    main()
