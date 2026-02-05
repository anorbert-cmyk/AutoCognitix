#!/usr/bin/env python3
"""
Comprehensive Qdrant Vector Database Indexer for AutoCognitix.

This script re-indexes ALL data into Qdrant with Hungarian huBERT embeddings:
- 3,579+ DTC codes with Hungarian descriptions
- Symptoms from DTC data and Neo4j
- Components from Neo4j
- Repair procedures from Neo4j

Creates 4 separate collections (768-dim huBERT embeddings):
- dtc_embeddings_hu: DTC codes with Hungarian/English descriptions
- symptom_embeddings_hu: Vehicle symptoms
- component_embeddings_hu: Vehicle components
- repair_embeddings_hu: Repair procedures

Usage:
    python scripts/index_qdrant_full.py --all              # Index everything
    python scripts/index_qdrant_full.py --dtc              # Index DTC codes only
    python scripts/index_qdrant_full.py --symptoms         # Index symptoms only
    python scripts/index_qdrant_full.py --components       # Index components only
    python scripts/index_qdrant_full.py --repairs          # Index repairs only
    python scripts/index_qdrant_full.py --all --recreate   # Recreate and reindex all
    python scripts/index_qdrant_full.py --verify           # Verify with test queries

Requirements:
    - Qdrant running locally or Qdrant Cloud configured
    - huBERT model (SZTAKI-HLT/hubert-base-cc)
    - Neo4j running (for component/repair data)
"""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from transformers import AutoModel, AutoTokenizer

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration - can be overridden by environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL")  # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # For Qdrant Cloud
HUBERT_MODEL = os.getenv("HUBERT_MODEL", "SZTAKI-HLT/hubert-base-cc")
EMBEDDING_DIMENSION = 768  # huBERT output dimension

# Collection names - Hungarian versions with huBERT embeddings (768-dim)
DTC_COLLECTION = "dtc_embeddings_hu"
SYMPTOM_COLLECTION = "symptom_embeddings_hu"
COMPONENT_COLLECTION = "component_embeddings_hu"
REPAIR_COLLECTION = "repair_embeddings_hu"

# Batch sizes for optimal performance
EMBEDDING_BATCH_SIZE = 32  # Texts per embedding batch
UPSERT_BATCH_SIZE = 100  # Vectors per Qdrant upsert batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class HuBERTEmbedder:
    """Handles Hungarian text embedding using huBERT model with GPU/MPS support."""

    def __init__(self, model_name: str = HUBERT_MODEL):
        """Initialize the huBERT model."""
        self.model_name = model_name
        self.device = self._detect_device()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        logger.info(f"HuBERTEmbedder initialized with device: {self.device}")

    def _detect_device(self) -> torch.device:
        """Detect the best available device (CUDA > MPS > CPU)."""
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

    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Validate that embedding is valid (no NaN/Inf, correct dimension)."""
        if len(embedding) != EMBEDDING_DIMENSION:
            return False
        for val in embedding:
            if np.isnan(val) or np.isinf(val):
                return False
        return True

    def embed_batch(
        self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch

        Returns:
            List of 768-dimensional embedding vectors
        """
        self._load_model()

        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Handle empty texts in batch
            non_empty_indices = []
            non_empty_texts = []
            for idx, text in enumerate(batch_texts):
                if text and text.strip():
                    non_empty_indices.append(idx)
                    non_empty_texts.append(text.strip())

            # Initialize batch embeddings with zeros
            batch_embeddings = [[0.0] * EMBEDDING_DIMENSION for _ in range(len(batch_texts))]

            if non_empty_texts:
                try:
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

                    # Convert to list and validate
                    embeddings_list = embeddings.cpu().tolist()
                    for orig_idx, embedding in zip(non_empty_indices, embeddings_list):
                        if self._validate_embedding(embedding):
                            batch_embeddings[orig_idx] = embedding
                        else:
                            logger.warning(f"Invalid embedding generated for text at index {orig_idx}")

                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {e}")
                    # Keep zero vectors for failed batch

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        result = self.embed_batch([text])
        return result[0] if result else [0.0] * EMBEDDING_DIMENSION


# Global embedder instance
_embedder: Optional[HuBERTEmbedder] = None


def get_embedder() -> HuBERTEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = HuBERTEmbedder()
    return _embedder


class QdrantIndexer:
    """Handles Qdrant vector database indexing operations."""

    def __init__(self):
        """Initialize the Qdrant client."""
        if QDRANT_URL:
            # Qdrant Cloud configuration
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            logger.info(f"Connected to Qdrant Cloud: {QDRANT_URL}")
        else:
            # Local Qdrant configuration
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
            collection_name: Name of the collection to create
            recreate: If True, delete existing collection and recreate
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
            logger.info(f"Created collection: {collection_name} (dim={self.vector_size})")

        except Exception as e:
            logger.error(f"Error creating collection {collection_name}: {e}")
            raise

    def upsert_batch(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> int:
        """
        Upsert a batch of vectors into a collection.

        Args:
            collection_name: Target collection name
            ids: List of unique IDs for the points
            vectors: List of embedding vectors
            payloads: List of metadata payloads

        Returns:
            Number of successfully upserted points
        """
        # Filter out invalid vectors
        valid_points = []
        for id_, vector, payload in zip(ids, vectors, payloads):
            # Validate vector
            if len(vector) != EMBEDDING_DIMENSION:
                logger.warning(f"Skipping invalid vector for {id_}: wrong dimension")
                continue
            if any(np.isnan(v) or np.isinf(v) for v in vector):
                logger.warning(f"Skipping invalid vector for {id_}: contains NaN/Inf")
                continue
            # Check for zero vector
            if all(v == 0.0 for v in vector):
                logger.warning(f"Skipping zero vector for {id_}")
                continue

            # Sanitize payload - remove sensitive data and validate
            sanitized_payload = self._sanitize_payload(payload)

            valid_points.append(
                qdrant_models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, id_)),
                    vector=vector,
                    payload=sanitized_payload,
                )
            )

        if not valid_points:
            return 0

        self.client.upsert(
            collection_name=collection_name,
            points=valid_points,
        )

        return len(valid_points)

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize payload to remove sensitive data and validate values."""
        sanitized = {}
        # Sensitive field patterns to exclude
        sensitive_patterns = ["password", "secret", "token", "api_key", "credential"]

        for key, value in payload.items():
            # Skip sensitive fields
            if any(pattern in key.lower() for pattern in sensitive_patterns):
                continue

            # Handle different value types
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Ensure list elements are JSON-serializable
                sanitized[key] = [
                    v for v in value
                    if isinstance(v, (str, int, float, bool, type(None)))
                ]
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts
                sanitized[key] = self._sanitize_payload(value)

        return sanitized

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }

        if score_threshold:
            search_params["score_threshold"] = score_threshold

        results = self.client.search(**search_params)

        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in results
        ]

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
    """Load DTC codes from JSON file."""
    logger.info(f"Loading DTC codes from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"DTC codes file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    codes = data.get("codes", [])
    metadata = data.get("metadata", {})

    logger.info(f"Loaded {len(codes)} DTC codes")
    logger.info(f"  - Total codes: {metadata.get('total_codes', 'N/A')}")
    logger.info(f"  - Translated: {metadata.get('translated', 'N/A')}")

    return codes


def load_neo4j_data() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load symptoms, components, and repairs from Neo4j.

    Returns:
        Tuple of (symptoms, components, repairs)
    """
    symptoms = []
    components = []
    repairs = []

    try:
        from backend.app.db.neo4j_models import (
            SymptomNode,
            ComponentNode,
            RepairNode,
        )

        # Load symptoms
        logger.info("Loading symptoms from Neo4j...")
        for symptom in SymptomNode.nodes.all():
            symptoms.append({
                "name": symptom.name,
                "description": symptom.description,
                "description_hu": symptom.description_hu,
                "severity": symptom.severity,
            })
        logger.info(f"Loaded {len(symptoms)} symptoms from Neo4j")

        # Load components
        logger.info("Loading components from Neo4j...")
        for component in ComponentNode.nodes.all():
            components.append({
                "name": component.name,
                "name_hu": component.name_hu,
                "system": component.system,
                "part_number": component.part_number,
            })
        logger.info(f"Loaded {len(components)} components from Neo4j")

        # Load repairs
        logger.info("Loading repairs from Neo4j...")
        for repair in RepairNode.nodes.all():
            repairs.append({
                "name": repair.name,
                "description": repair.description,
                "description_hu": repair.description_hu,
                "difficulty": repair.difficulty,
                "estimated_time_minutes": repair.estimated_time_minutes,
                "estimated_cost_min": repair.estimated_cost_min,
                "estimated_cost_max": repair.estimated_cost_max,
            })
        logger.info(f"Loaded {len(repairs)} repairs from Neo4j")

    except Exception as e:
        logger.warning(f"Could not load Neo4j data: {e}")
        logger.info("Using fallback data from seed_database definitions")

        # Fallback to seed data if Neo4j is not available
        from scripts.seed_database import COMPONENTS, REPAIRS

        components = COMPONENTS
        repairs = REPAIRS

        logger.info(f"Using {len(components)} components and {len(repairs)} repairs from fallback data")

    return symptoms, components, repairs


def extract_symptoms_from_dtc(dtc_codes: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """
    Extract unique symptoms from DTC codes.

    Args:
        dtc_codes: List of DTC code dictionaries

    Returns:
        Dictionary mapping symptom text to related DTC codes
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

    logger.info(f"Extracted {len(symptom_to_codes)} unique symptoms from DTC codes")
    return symptom_to_codes


def index_dtc_codes(
    indexer: QdrantIndexer,
    dtc_codes: List[Dict[str, Any]],
    recreate: bool = False,
) -> int:
    """
    Index DTC codes into Qdrant with Hungarian huBERT embeddings.

    Args:
        indexer: QdrantIndexer instance
        dtc_codes: List of DTC codes to index
        recreate: Whether to recreate the collection

    Returns:
        Number of indexed codes
    """
    logger.info("=" * 60)
    logger.info("Starting DTC code indexing...")
    logger.info(f"Total codes to index: {len(dtc_codes)}")

    # Create collection
    indexer.create_collection(DTC_COLLECTION, recreate=recreate)

    # Prepare data for indexing
    ids: List[str] = []
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for dtc in dtc_codes:
        code = dtc.get("code", "")
        description_hu = dtc.get("description_hu", "")
        description_en = dtc.get("description_en", "")

        if not code:
            continue

        # Use Hungarian description if available, otherwise English
        text_to_embed = description_hu if description_hu else description_en
        if not text_to_embed:
            logger.debug(f"Skipping DTC {code}: no description available")
            continue

        ids.append(f"dtc_{code}")
        texts.append(text_to_embed)
        payloads.append({
            "code": code,
            "description_hu": description_hu,
            "description_en": description_en,
            "category": dtc.get("category", "unknown"),
            "severity": dtc.get("severity", "unknown"),
            "system": dtc.get("system", ""),
            "symptoms": dtc.get("symptoms", []),
            "possible_causes": dtc.get("possible_causes", []),
            "diagnostic_steps": dtc.get("diagnostic_steps", []),
            "related_codes": dtc.get("related_codes", []),
            "is_generic": dtc.get("is_generic", True),
            "sources": dtc.get("sources", []),
            "manufacturer": dtc.get("manufacturer"),
        })

    if not ids:
        logger.warning("No valid DTC codes to index")
        return 0

    # Generate embeddings with progress bar
    logger.info(f"Generating embeddings for {len(texts)} DTC descriptions...")
    embedder = get_embedder()

    all_embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding DTC codes"):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embedder.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant with progress bar
    logger.info(f"Upserting {len(ids)} vectors to Qdrant...")
    total_indexed = 0

    for i in tqdm(range(0, len(ids), UPSERT_BATCH_SIZE), desc="Indexing DTC codes"):
        batch_ids = ids[i:i + UPSERT_BATCH_SIZE]
        batch_vectors = all_embeddings[i:i + UPSERT_BATCH_SIZE]
        batch_payloads = payloads[i:i + UPSERT_BATCH_SIZE]

        count = indexer.upsert_batch(
            collection_name=DTC_COLLECTION,
            ids=batch_ids,
            vectors=batch_vectors,
            payloads=batch_payloads,
        )
        total_indexed += count

    logger.info(f"Successfully indexed {total_indexed} DTC codes into {DTC_COLLECTION}")
    return total_indexed


def index_symptoms(
    indexer: QdrantIndexer,
    dtc_codes: List[Dict[str, Any]],
    neo4j_symptoms: List[Dict[str, Any]],
    recreate: bool = False,
) -> int:
    """
    Index symptoms into Qdrant.

    Combines symptoms from DTC codes and Neo4j.

    Args:
        indexer: QdrantIndexer instance
        dtc_codes: List of DTC codes to extract symptoms from
        neo4j_symptoms: List of symptom dictionaries from Neo4j
        recreate: Whether to recreate the collection

    Returns:
        Number of indexed symptoms
    """
    logger.info("=" * 60)
    logger.info("Starting symptom indexing...")

    # Create collection
    indexer.create_collection(SYMPTOM_COLLECTION, recreate=recreate)

    # Extract symptoms from DTC codes
    symptom_to_codes = extract_symptoms_from_dtc(dtc_codes)

    # Prepare data for indexing
    ids: List[str] = []
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    # Add symptoms from DTC codes
    for symptom_text, related_codes in symptom_to_codes.items():
        symptom_id = f"symptom_dtc_{hash(symptom_text) % 10**10}"
        ids.append(symptom_id)
        texts.append(symptom_text)
        payloads.append({
            "symptom_text": symptom_text,
            "symptom_text_hu": symptom_text,  # Already in Hungarian
            "related_dtc_codes": sorted(list(related_codes)),
            "related_codes_count": len(related_codes),
            "source": "dtc_codes",
        })

    # Add symptoms from Neo4j (avoiding duplicates)
    existing_texts = set(texts)
    for symptom in neo4j_symptoms:
        name = symptom.get("name", "")
        if not name or name in existing_texts:
            continue

        symptom_id = f"symptom_neo4j_{hash(name) % 10**10}"
        ids.append(symptom_id)
        # Use Hungarian description if available
        text_to_embed = symptom.get("description_hu") or name
        texts.append(text_to_embed)
        payloads.append({
            "symptom_text": name,
            "symptom_text_hu": symptom.get("description_hu", name),
            "description": symptom.get("description"),
            "severity": symptom.get("severity", "medium"),
            "source": "neo4j",
        })
        existing_texts.add(name)

    if not ids:
        logger.warning("No symptoms to index")
        return 0

    logger.info(f"Total unique symptoms to index: {len(ids)}")

    # Generate embeddings
    logger.info("Generating embeddings for symptoms...")
    embedder = get_embedder()

    all_embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding symptoms"):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embedder.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant
    logger.info(f"Upserting {len(ids)} symptom vectors to Qdrant...")
    total_indexed = 0

    for i in tqdm(range(0, len(ids), UPSERT_BATCH_SIZE), desc="Indexing symptoms"):
        batch_ids = ids[i:i + UPSERT_BATCH_SIZE]
        batch_vectors = all_embeddings[i:i + UPSERT_BATCH_SIZE]
        batch_payloads = payloads[i:i + UPSERT_BATCH_SIZE]

        count = indexer.upsert_batch(
            collection_name=SYMPTOM_COLLECTION,
            ids=batch_ids,
            vectors=batch_vectors,
            payloads=batch_payloads,
        )
        total_indexed += count

    logger.info(f"Successfully indexed {total_indexed} symptoms into {SYMPTOM_COLLECTION}")
    return total_indexed


def index_components(
    indexer: QdrantIndexer,
    components: List[Dict[str, Any]],
    recreate: bool = False,
) -> int:
    """
    Index vehicle components into Qdrant.

    Args:
        indexer: QdrantIndexer instance
        components: List of component dictionaries
        recreate: Whether to recreate the collection

    Returns:
        Number of indexed components
    """
    logger.info("=" * 60)
    logger.info("Starting component indexing...")

    # Create collection
    indexer.create_collection(COMPONENT_COLLECTION, recreate=recreate)

    if not components:
        logger.warning("No components to index")
        return 0

    # Prepare data for indexing
    ids: List[str] = []
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for comp in components:
        name = comp.get("name", "")
        if not name:
            continue

        # Use Hungarian name if available, otherwise English
        name_hu = comp.get("name_hu", "")
        text_to_embed = name_hu if name_hu else name

        component_id = f"component_{hash(name) % 10**10}"
        ids.append(component_id)
        texts.append(text_to_embed)
        payloads.append({
            "name": name,
            "name_hu": name_hu,
            "system": comp.get("system", ""),
            "part_number": comp.get("part_number"),
        })

    logger.info(f"Components to index: {len(ids)}")

    # Generate embeddings
    logger.info("Generating embeddings for components...")
    embedder = get_embedder()

    all_embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding components"):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embedder.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant
    logger.info(f"Upserting {len(ids)} component vectors to Qdrant...")
    total_indexed = 0

    for i in tqdm(range(0, len(ids), UPSERT_BATCH_SIZE), desc="Indexing components"):
        batch_ids = ids[i:i + UPSERT_BATCH_SIZE]
        batch_vectors = all_embeddings[i:i + UPSERT_BATCH_SIZE]
        batch_payloads = payloads[i:i + UPSERT_BATCH_SIZE]

        count = indexer.upsert_batch(
            collection_name=COMPONENT_COLLECTION,
            ids=batch_ids,
            vectors=batch_vectors,
            payloads=batch_payloads,
        )
        total_indexed += count

    logger.info(f"Successfully indexed {total_indexed} components into {COMPONENT_COLLECTION}")
    return total_indexed


def index_repairs(
    indexer: QdrantIndexer,
    repairs: List[Dict[str, Any]],
    recreate: bool = False,
) -> int:
    """
    Index repair procedures into Qdrant.

    Args:
        indexer: QdrantIndexer instance
        repairs: List of repair dictionaries
        recreate: Whether to recreate the collection

    Returns:
        Number of indexed repairs
    """
    logger.info("=" * 60)
    logger.info("Starting repair procedure indexing...")

    # Create collection
    indexer.create_collection(REPAIR_COLLECTION, recreate=recreate)

    if not repairs:
        logger.warning("No repairs to index")
        return 0

    # Prepare data for indexing
    ids: List[str] = []
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for repair in repairs:
        name = repair.get("name", "")
        if not name:
            continue

        # Use Hungarian description/name if available
        description_hu = repair.get("description_hu") or repair.get("name_hu", "")
        text_to_embed = description_hu if description_hu else name

        repair_id = f"repair_{hash(name) % 10**10}"
        ids.append(repair_id)
        texts.append(text_to_embed)
        payloads.append({
            "name": name,
            "description": repair.get("description"),
            "description_hu": description_hu,
            "difficulty": repair.get("difficulty", "intermediate"),
            "estimated_time_minutes": repair.get("estimated_time_minutes"),
            "estimated_cost_min": repair.get("estimated_cost_min"),
            "estimated_cost_max": repair.get("estimated_cost_max"),
        })

    logger.info(f"Repairs to index: {len(ids)}")

    # Generate embeddings
    logger.info("Generating embeddings for repairs...")
    embedder = get_embedder()

    all_embeddings: List[List[float]] = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding repairs"):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_embeddings = embedder.embed_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant
    logger.info(f"Upserting {len(ids)} repair vectors to Qdrant...")
    total_indexed = 0

    for i in tqdm(range(0, len(ids), UPSERT_BATCH_SIZE), desc="Indexing repairs"):
        batch_ids = ids[i:i + UPSERT_BATCH_SIZE]
        batch_vectors = all_embeddings[i:i + UPSERT_BATCH_SIZE]
        batch_payloads = payloads[i:i + UPSERT_BATCH_SIZE]

        count = indexer.upsert_batch(
            collection_name=REPAIR_COLLECTION,
            ids=batch_ids,
            vectors=batch_vectors,
            payloads=batch_payloads,
        )
        total_indexed += count

    logger.info(f"Successfully indexed {total_indexed} repairs into {REPAIR_COLLECTION}")
    return total_indexed


def verify_semantic_search(indexer: QdrantIndexer) -> bool:
    """
    Verify semantic search functionality with test queries.

    Returns:
        True if verification passes, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Verifying semantic search functionality...")

    embedder = get_embedder()
    all_passed = True

    # Test queries in Hungarian
    test_cases = [
        {
            "collection": DTC_COLLECTION,
            "query": "A motor nem indul hidegen",
            "description": "Cold start engine issue query",
            "expected_relevance": ["P0", "indulás", "hideg", "motor"],
        },
        {
            "collection": DTC_COLLECTION,
            "query": "Gyenge gyorsulás autópályán",
            "description": "Weak acceleration query",
            "expected_relevance": ["P0", "gyorsulás", "teljesítmény"],
        },
        {
            "collection": SYMPTOM_COLLECTION,
            "query": "Motor vibrál alapjáraton",
            "description": "Engine vibration symptom query",
            "expected_relevance": ["vibrál", "alapjárat", "motor"],
        },
        {
            "collection": COMPONENT_COLLECTION,
            "query": "Levegő mérő szenzor",
            "description": "MAF sensor component query",
            "expected_relevance": ["MAF", "levegő", "szenzor"],
        },
        {
            "collection": REPAIR_COLLECTION,
            "query": "Gyertya csere",
            "description": "Spark plug replacement query",
            "expected_relevance": ["gyertya", "csere"],
        },
    ]

    for test in test_cases:
        collection = test["collection"]
        query = test["query"]
        description = test["description"]

        logger.info(f"\nTest: {description}")
        logger.info(f"Query: '{query}'")

        # Check if collection exists
        info = indexer.get_collection_info(collection)
        if info.get("status") == "not_found":
            logger.warning(f"  Collection {collection} not found - skipping test")
            continue

        # Generate query embedding
        query_embedding = embedder.embed_single(query)

        # Search
        results = indexer.search(
            collection_name=collection,
            query_vector=query_embedding,
            limit=5,
        )

        if not results:
            logger.warning(f"  No results returned for query!")
            all_passed = False
            continue

        logger.info(f"  Top {len(results)} results:")
        for i, result in enumerate(results[:3]):
            score = result["score"]
            payload = result["payload"]

            # Get display text based on collection type
            if collection == DTC_COLLECTION:
                display = f"{payload.get('code', 'N/A')}: {payload.get('description_hu', payload.get('description_en', 'N/A'))[:60]}"
            elif collection == SYMPTOM_COLLECTION:
                display = payload.get("symptom_text_hu", payload.get("symptom_text", "N/A"))[:60]
            elif collection == COMPONENT_COLLECTION:
                display = payload.get("name_hu", payload.get("name", "N/A"))[:60]
            elif collection == REPAIR_COLLECTION:
                display = payload.get("description_hu", payload.get("name", "N/A"))[:60]
            else:
                display = str(payload)[:60]

            logger.info(f"    {i+1}. [{score:.4f}] {display}")

        # Check if top result has reasonable relevance score
        top_score = results[0]["score"]
        if top_score < 0.3:
            logger.warning(f"  Low relevance score ({top_score:.4f}) - may indicate indexing issues")
            all_passed = False
        else:
            logger.info(f"  Relevance OK (top score: {top_score:.4f})")

    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("VERIFICATION PASSED - Semantic search is working correctly")
    else:
        logger.warning("VERIFICATION WARNINGS - Some tests had issues (see above)")

    return all_passed


def print_summary(indexer: QdrantIndexer) -> None:
    """Print summary of all indexed collections."""
    print("\n" + "=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)

    collections = [
        DTC_COLLECTION,
        SYMPTOM_COLLECTION,
        COMPONENT_COLLECTION,
        REPAIR_COLLECTION,
    ]

    total_vectors = 0
    for collection_name in collections:
        info = indexer.get_collection_info(collection_name)
        print(f"\nCollection: {info['name']}")
        if info.get("status") == "not_found":
            print("  Status: Not created")
        else:
            print(f"  Status: {info.get('status', 'unknown')}")
            print(f"  Points count: {info.get('points_count', 0)}")
            print(f"  Vectors count: {info.get('vectors_count', 0)}")
            total_vectors += info.get("vectors_count", 0) or 0

    print("\n" + "-" * 60)
    print(f"TOTAL VECTORS: {total_vectors}")
    print("=" * 60)


def main():
    """Main entry point for the comprehensive indexer."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Qdrant indexer for AutoCognitix with Hungarian huBERT embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/index_qdrant_full.py --all              # Index everything
    python scripts/index_qdrant_full.py --dtc              # Index only DTC codes
    python scripts/index_qdrant_full.py --symptoms         # Index only symptoms
    python scripts/index_qdrant_full.py --components       # Index only components
    python scripts/index_qdrant_full.py --repairs          # Index only repairs
    python scripts/index_qdrant_full.py --all --recreate   # Recreate collections and reindex
    python scripts/index_qdrant_full.py --verify           # Run verification tests only
    python scripts/index_qdrant_full.py --all --verify     # Index all and verify
        """,
    )

    parser.add_argument("--dtc", action="store_true", help="Index DTC codes")
    parser.add_argument("--symptoms", action="store_true", help="Index symptoms")
    parser.add_argument("--components", action="store_true", help="Index components")
    parser.add_argument("--repairs", action="store_true", help="Index repairs")
    parser.add_argument("--all", action="store_true", help="Index all data types")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collections")
    parser.add_argument("--verify", action="store_true", help="Run verification tests")
    parser.add_argument(
        "--dtc-file",
        type=str,
        default=None,
        help="Path to DTC codes JSON file (default: data/dtc_codes/all_codes_merged.json)",
    )

    args = parser.parse_args()

    # Determine what to index
    index_all = args.all
    index_dtc = args.dtc or index_all
    index_symptoms = args.symptoms or index_all
    index_components = args.components or index_all
    index_repairs = args.repairs or index_all

    # If nothing specified and not just verifying, show help
    if not (index_dtc or index_symptoms or index_components or index_repairs or args.verify):
        parser.print_help()
        print("\nError: Please specify at least one indexing option or --verify")
        sys.exit(1)

    # Set up paths
    if args.dtc_file:
        dtc_file_path = Path(args.dtc_file)
    else:
        dtc_file_path = PROJECT_ROOT / "data" / "dtc_codes" / "all_codes_merged.json"

    try:
        # Initialize indexer
        logger.info("Initializing Qdrant indexer...")
        indexer = QdrantIndexer()

        # Track totals
        totals = {
            "dtc": 0,
            "symptoms": 0,
            "components": 0,
            "repairs": 0,
        }

        # Load data if needed
        dtc_codes = []
        neo4j_symptoms = []
        components = []
        repairs = []

        if index_dtc or index_symptoms:
            dtc_codes = load_dtc_codes(dtc_file_path)

        if index_symptoms or index_components or index_repairs:
            neo4j_symptoms, components, repairs = load_neo4j_data()

        # Index each type as requested
        if index_dtc:
            totals["dtc"] = index_dtc_codes(indexer, dtc_codes, recreate=args.recreate)

        if index_symptoms:
            totals["symptoms"] = index_symptoms(indexer, dtc_codes, neo4j_symptoms, recreate=args.recreate)

        if index_components:
            totals["components"] = index_components(indexer, components, recreate=args.recreate)

        if index_repairs:
            totals["repairs"] = index_repairs(indexer, repairs, recreate=args.recreate)

        # Print summary
        print_summary(indexer)

        # Log totals
        total_indexed = sum(totals.values())
        if total_indexed > 0:
            logger.info(f"\nTotal vectors indexed this run:")
            logger.info(f"  - DTC codes: {totals['dtc']}")
            logger.info(f"  - Symptoms: {totals['symptoms']}")
            logger.info(f"  - Components: {totals['components']}")
            logger.info(f"  - Repairs: {totals['repairs']}")
            logger.info(f"  - TOTAL: {total_indexed}")

        # Run verification if requested
        if args.verify:
            verify_semantic_search(indexer)

        logger.info("\nIndexing completed successfully!")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ConnectionError as e:
        logger.error(f"Could not connect to Qdrant: {e}")
        logger.error("Make sure Qdrant is running or QDRANT_URL is set correctly")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
