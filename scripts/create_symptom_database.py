#!/usr/bin/env python3
"""
Symptom Database Creator for AutoCognitix

This script creates a comprehensive symptom database by:
1. Loading symptoms from the Hungarian symptom JSON file
2. Creating Symptom nodes in Neo4j with relationships to DTC codes
3. Indexing symptoms in Qdrant for semantic search using huBERT embeddings

Usage:
    python scripts/create_symptom_database.py --neo4j      # Create Neo4j nodes only
    python scripts/create_symptom_database.py --qdrant     # Index in Qdrant only
    python scripts/create_symptom_database.py --all        # Both Neo4j and Qdrant
    python scripts/create_symptom_database.py --all --recreate  # Recreate collections

Requirements:
    - Neo4j running and configured
    - Qdrant running on localhost:6333
    - huBERT model for Hungarian embeddings
"""

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL", None)  # For cloud deployment
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
HUBERT_MODEL = os.getenv("HUBERT_MODEL", "SZTAKI-HLT/hubert-base-cc")
EMBEDDING_DIMENSION = 768

# Collection names
SYMPTOM_COLLECTION = "symptom_embeddings_hu"

# Data paths
SYMPTOMS_FILE = PROJECT_ROOT / "data" / "symptoms" / "symptoms_hu.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Service (standalone for script usage)
# =============================================================================

class HuBERTEmbedder:
    """Handles Hungarian text embedding using huBERT model."""

    def __init__(self, model_name: str = HUBERT_MODEL):
        """Initialize the huBERT model."""
        self.model_name = model_name
        self.device = self._detect_device()
        self._tokenizer = None
        self._model = None
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

        from transformers import AutoModel, AutoTokenizer

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

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        self._load_model()

        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIMENSION

        encoded = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            model_output = self._model(**encoded)

        embedding = self._mean_pooling(model_output, encoded["attention_mask"])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.squeeze().cpu().tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generate embeddings for multiple texts with batch processing."""
        self._load_model()

        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

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
                encoded = self._tokenizer(
                    non_empty_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    model_output = self._model(**encoded)

                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

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


# =============================================================================
# Data Loading
# =============================================================================

def load_symptoms(file_path: Path = SYMPTOMS_FILE) -> Dict[str, Any]:
    """
    Load symptoms from JSON file.

    Returns:
        Dictionary containing metadata, categories, and symptoms.
    """
    logger.info(f"Loading symptoms from: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Symptoms file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    symptoms = data.get("symptoms", [])
    categories = data.get("categories", [])
    metadata = data.get("metadata", {})

    logger.info(f"Loaded {len(symptoms)} symptoms in {len(categories)} categories")
    return data


def load_dtc_codes() -> List[Dict[str, Any]]:
    """Load DTC codes from the merged JSON file for relationship creation."""
    dtc_file = PROJECT_ROOT / "data" / "dtc_codes" / "all_codes_merged.json"

    if not dtc_file.exists():
        logger.warning(f"DTC codes file not found: {dtc_file}")
        return []

    with open(dtc_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("codes", [])


# =============================================================================
# Neo4j Integration
# =============================================================================

def create_neo4j_symptom_nodes(symptoms_data: Dict[str, Any]) -> Dict[str, int]:
    """
    Create Symptom nodes in Neo4j with full metadata.

    Args:
        symptoms_data: Dictionary containing symptoms and categories.

    Returns:
        Dictionary with counts of created nodes and relationships.
    """
    from backend.app.db.neo4j_models import DTCNode, SymptomNode

    symptoms = symptoms_data.get("symptoms", [])
    categories = {cat["id"]: cat for cat in symptoms_data.get("categories", [])}

    counts = {
        "symptoms_created": 0,
        "symptoms_updated": 0,
        "dtc_relationships": 0,
        "related_symptom_relationships": 0,
    }

    logger.info("Creating Neo4j Symptom nodes...")

    # First pass: create all symptom nodes
    symptom_nodes: Dict[str, SymptomNode] = {}

    for symptom in tqdm(symptoms, desc="Creating Symptom nodes"):
        symptom_id = symptom["id"]
        symptom_name = symptom["name_hu"]

        # Check if symptom already exists (by name or symptom_id)
        existing = SymptomNode.nodes.get_or_none(name=symptom_name)
        if not existing:
            existing = SymptomNode.nodes.get_or_none(symptom_id=symptom_id)

        if existing:
            # Update existing node with all fields
            existing.symptom_id = symptom_id
            existing.name_en = symptom.get("name_en", "")
            existing.description = symptom.get("description_en", "")
            existing.description_hu = symptom.get("description_hu", "")
            existing.category = symptom.get("category", "")
            existing.severity = symptom.get("severity", "medium")
            existing.keywords = symptom.get("keywords", [])
            existing.possible_causes = symptom.get("possible_causes", [])
            existing.diagnostic_steps = symptom.get("diagnostic_steps", [])
            existing.save()
            counts["symptoms_updated"] += 1
            symptom_nodes[symptom_id] = existing
        else:
            # Create new node with all fields
            symptom_node = SymptomNode(
                symptom_id=symptom_id,
                name=symptom_name,
                name_en=symptom.get("name_en", ""),
                description=symptom.get("description_en", ""),
                description_hu=symptom.get("description_hu", ""),
                category=symptom.get("category", ""),
                severity=symptom.get("severity", "medium"),
                keywords=symptom.get("keywords", []),
                possible_causes=symptom.get("possible_causes", []),
                diagnostic_steps=symptom.get("diagnostic_steps", []),
            ).save()
            counts["symptoms_created"] += 1
            symptom_nodes[symptom_id] = symptom_node

    logger.info("Creating DTC relationships...")

    # Second pass: create relationships to DTC codes
    for symptom in tqdm(symptoms, desc="Creating DTC relationships"):
        symptom_id = symptom["id"]
        symptom_node = symptom_nodes.get(symptom_id)
        if not symptom_node:
            continue

        related_codes = symptom.get("related_dtc_codes", [])
        for dtc_code in related_codes:
            dtc_node = DTCNode.nodes.get_or_none(code=dtc_code)
            if dtc_node:
                # Check if relationship already exists
                if not dtc_node.causes.is_connected(symptom_node):
                    # Create relationship with confidence based on symptom data
                    dtc_node.causes.connect(symptom_node, {
                        "confidence": 0.75,
                        "data_source": "symptom_database"
                    })
                    counts["dtc_relationships"] += 1

    logger.info("Creating related symptom relationships...")

    # Third pass: create relationships between symptoms in the same category
    # (symptoms in the same category that share DTC codes are related)
    category_symptoms: Dict[str, List[str]] = {}
    for symptom in symptoms:
        category = symptom.get("category", "")
        if category:
            if category not in category_symptoms:
                category_symptoms[category] = []
            category_symptoms[category].append(symptom["id"])

    # Link symptoms that share the same DTC codes
    dtc_to_symptoms: Dict[str, List[str]] = {}
    for symptom in symptoms:
        for dtc_code in symptom.get("related_dtc_codes", []):
            if dtc_code not in dtc_to_symptoms:
                dtc_to_symptoms[dtc_code] = []
            dtc_to_symptoms[dtc_code].append(symptom["id"])

    # Create relationships for symptoms sharing 2+ DTC codes
    for symptom in symptoms:
        symptom_id = symptom["id"]
        symptom_node = symptom_nodes.get(symptom_id)
        if not symptom_node:
            continue

        related_symptoms_set: Set[str] = set()
        for dtc_code in symptom.get("related_dtc_codes", []):
            for other_id in dtc_to_symptoms.get(dtc_code, []):
                if other_id != symptom_id:
                    related_symptoms_set.add(other_id)

        # Create relationships for symptoms with strong DTC overlap
        for other_id in related_symptoms_set:
            other_node = symptom_nodes.get(other_id)
            if other_node and not symptom_node.related_symptoms.is_connected(other_node):
                # Only link if they share 2+ DTC codes
                shared_dtcs = set(symptom.get("related_dtc_codes", [])) & set(
                    next((s.get("related_dtc_codes", []) for s in symptoms if s["id"] == other_id), [])
                )
                if len(shared_dtcs) >= 2:
                    symptom_node.related_symptoms.connect(other_node)
                    counts["related_symptom_relationships"] += 1

    return counts


def create_neo4j_category_stats(symptoms_data: Dict[str, Any]) -> Dict[str, int]:
    """
    Calculate category statistics for reporting.

    Args:
        symptoms_data: Dictionary containing symptoms and categories.

    Returns:
        Dictionary with symptom counts per category.
    """
    symptoms = symptoms_data.get("symptoms", [])
    category_counts: Dict[str, int] = {}

    for symptom in symptoms:
        category = symptom.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

    return category_counts


# =============================================================================
# Qdrant Integration
# =============================================================================

def create_qdrant_client():
    """Create Qdrant client with appropriate connection settings."""
    from qdrant_client import QdrantClient

    if QDRANT_URL and QDRANT_API_KEY:
        # Cloud deployment
        logger.info(f"Connecting to Qdrant Cloud: {QDRANT_URL}")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        # Local deployment
        logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_symptom_collection(client, recreate: bool = False) -> None:
    """
    Create the symptom embeddings collection in Qdrant.

    Args:
        client: QdrantClient instance.
        recreate: If True, delete and recreate the collection.
    """
    from qdrant_client.http import models as qdrant_models

    try:
        collections = client.get_collections().collections
        exists = any(c.name == SYMPTOM_COLLECTION for c in collections)

        if exists:
            if recreate:
                logger.info(f"Deleting existing collection: {SYMPTOM_COLLECTION}")
                client.delete_collection(collection_name=SYMPTOM_COLLECTION)
            else:
                logger.info(f"Collection '{SYMPTOM_COLLECTION}' already exists, skipping creation")
                return

        client.create_collection(
            collection_name=SYMPTOM_COLLECTION,
            vectors_config=qdrant_models.VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {SYMPTOM_COLLECTION}")

    except Exception as e:
        logger.error(f"Error creating collection {SYMPTOM_COLLECTION}: {e}")
        raise


def index_symptoms_in_qdrant(
    symptoms_data: Dict[str, Any],
    recreate: bool = False,
    batch_size: int = 10,
) -> int:
    """
    Index symptoms in Qdrant for semantic search.

    Args:
        symptoms_data: Dictionary containing symptoms.
        recreate: Whether to recreate the collection.
        batch_size: Number of symptoms to process per batch.

    Returns:
        Number of indexed symptoms.
    """
    from qdrant_client.http import models as qdrant_models

    client = create_qdrant_client()
    create_symptom_collection(client, recreate=recreate)

    symptoms = symptoms_data.get("symptoms", [])
    categories = {cat["id"]: cat for cat in symptoms_data.get("categories", [])}
    embedder = get_embedder()

    if not symptoms:
        logger.warning("No symptoms to index")
        return 0

    # Prepare data for indexing
    ids: List[str] = []
    texts: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for symptom in symptoms:
        symptom_id = symptom["id"]

        # Create rich text for embedding (Hungarian description + name + keywords)
        name_hu = symptom.get("name_hu", "")
        desc_hu = symptom.get("description_hu", "")
        keywords = symptom.get("keywords", [])
        possible_causes = symptom.get("possible_causes", [])

        # Combine text for better semantic matching
        text_parts = [name_hu, desc_hu]
        text_parts.extend(keywords)
        text_parts.extend(possible_causes[:3])  # Add top 3 causes
        embedding_text = " ".join(filter(None, text_parts))

        ids.append(f"symptom_{symptom_id}")
        texts.append(embedding_text)

        # Get category info
        category_id = symptom.get("category", "unknown")
        category_info = categories.get(category_id, {})

        payloads.append({
            "symptom_id": symptom_id,
            "name_hu": name_hu,
            "name_en": symptom.get("name_en", ""),
            "description_hu": desc_hu,
            "description_en": symptom.get("description_en", ""),
            "category": category_id,
            "category_name_hu": category_info.get("name_hu", ""),
            "category_name_en": category_info.get("name_en", ""),
            "severity": symptom.get("severity", "medium"),
            "related_dtc_codes": symptom.get("related_dtc_codes", []),
            "possible_causes": symptom.get("possible_causes", []),
            "diagnostic_steps": symptom.get("diagnostic_steps", []),
            "keywords": symptom.get("keywords", []),
        })

    # Generate embeddings with progress bar
    logger.info(f"Generating embeddings for {len(texts)} symptoms...")
    all_embeddings: List[List[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding symptoms"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embedder.embed_batch(batch_texts, batch_size=batch_size)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant with progress bar
    logger.info(f"Upserting {len(ids)} symptom vectors to Qdrant...")

    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing symptoms"):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = all_embeddings[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]

        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, id_)),
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
        ]

        client.upsert(
            collection_name=SYMPTOM_COLLECTION,
            points=points,
        )

    logger.info(f"Successfully indexed {len(ids)} symptoms")
    return len(ids)


def search_symptoms(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for symptoms using semantic similarity.

    Args:
        query: Hungarian text query describing symptoms.
        top_k: Number of results to return.

    Returns:
        List of matching symptoms with scores.
    """
    client = create_qdrant_client()
    embedder = get_embedder()

    # Generate query embedding
    query_embedding = embedder.embed_text(query)

    # Search in Qdrant
    results = client.search(
        collection_name=SYMPTOM_COLLECTION,
        query_vector=query_embedding,
        limit=top_k,
    )

    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            "score": result.score,
            "symptom_id": result.payload.get("symptom_id"),
            "name_hu": result.payload.get("name_hu"),
            "name_en": result.payload.get("name_en"),
            "description_hu": result.payload.get("description_hu"),
            "category": result.payload.get("category"),
            "severity": result.payload.get("severity"),
            "related_dtc_codes": result.payload.get("related_dtc_codes"),
            "possible_causes": result.payload.get("possible_causes"),
        })

    return formatted_results


# =============================================================================
# Reporting
# =============================================================================

def print_summary(
    symptoms_data: Dict[str, Any],
    neo4j_counts: Optional[Dict[str, int]] = None,
    qdrant_count: Optional[int] = None,
) -> None:
    """Print summary of the symptom database creation."""
    symptoms = symptoms_data.get("symptoms", [])
    categories = symptoms_data.get("categories", [])

    print("\n" + "=" * 70)
    print("SYMPTOM DATABASE CREATION SUMMARY")
    print("=" * 70)

    print(f"\nTotal symptoms: {len(symptoms)}")
    print(f"Categories: {len(categories)}")

    # Category breakdown
    print("\nSymptoms by category:")
    category_stats = create_neo4j_category_stats(symptoms_data)
    for cat in categories:
        count = category_stats.get(cat["id"], 0)
        print(f"  {cat['name_hu']:30} ({cat['name_en']:25}): {count:3}")

    # Severity breakdown
    severity_counts: Dict[str, int] = {}
    for symptom in symptoms:
        severity = symptom.get("severity", "unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1

    print("\nSymptoms by severity:")
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        print(f"  {severity:10}: {count:3}")

    # DTC code coverage
    all_dtc_codes: Set[str] = set()
    for symptom in symptoms:
        all_dtc_codes.update(symptom.get("related_dtc_codes", []))
    print(f"\nUnique DTC codes referenced: {len(all_dtc_codes)}")

    # Neo4j summary
    if neo4j_counts:
        print("\nNeo4j Database:")
        print(f"  Symptoms created: {neo4j_counts.get('symptoms_created', 0)}")
        print(f"  Symptoms updated: {neo4j_counts.get('symptoms_updated', 0)}")
        print(f"  DTC relationships: {neo4j_counts.get('dtc_relationships', 0)}")

    # Qdrant summary
    if qdrant_count is not None:
        print(f"\nQdrant Vector Database:")
        print(f"  Symptoms indexed: {qdrant_count}")

    print("\n" + "=" * 70)


def demo_search() -> None:
    """Demonstrate semantic symptom search."""
    print("\n" + "=" * 70)
    print("SYMPTOM SEARCH DEMO")
    print("=" * 70)

    test_queries = [
        "A motor nem indul be hideg időben",
        "Furcsa zaj a fékből",
        "Klíma nem működik rendesen",
        "Autó egyik oldalra húz",
        "Motor vibrál alapjáraton",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        try:
            results = search_symptoms(query, top_k=3)
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['name_hu']} (score: {result['score']:.3f})")
                print(f"     Category: {result['category']}, Severity: {result['severity']}")
                if result['related_dtc_codes']:
                    print(f"     Related DTCs: {', '.join(result['related_dtc_codes'][:3])}")
        except Exception as e:
            print(f"  Error: {e}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create symptom database for AutoCognitix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/create_symptom_database.py --neo4j         # Create Neo4j nodes only
    python scripts/create_symptom_database.py --qdrant        # Index in Qdrant only
    python scripts/create_symptom_database.py --all           # Both Neo4j and Qdrant
    python scripts/create_symptom_database.py --all --recreate  # Recreate and reindex
    python scripts/create_symptom_database.py --demo          # Run search demo
        """,
    )

    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Create Symptom nodes in Neo4j",
    )
    parser.add_argument(
        "--qdrant",
        action="store_true",
        help="Index symptoms in Qdrant vector database",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create both Neo4j nodes and Qdrant index",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collection before indexing",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run semantic search demo after indexing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for embedding generation (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not (args.neo4j or args.qdrant or args.all or args.demo):
        parser.print_help()
        print("\nError: Please specify at least one option: --neo4j, --qdrant, --all, or --demo")
        sys.exit(1)

    # Load symptoms data
    try:
        symptoms_data = load_symptoms()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    neo4j_counts = None
    qdrant_count = None

    try:
        # Create Neo4j nodes
        if args.neo4j or args.all:
            logger.info("Starting Neo4j symptom node creation...")
            neo4j_counts = create_neo4j_symptom_nodes(symptoms_data)
            logger.info("Neo4j symptom nodes created successfully")

        # Index in Qdrant
        if args.qdrant or args.all:
            logger.info("Starting Qdrant symptom indexing...")
            qdrant_count = index_symptoms_in_qdrant(
                symptoms_data,
                recreate=args.recreate,
                batch_size=args.batch_size,
            )
            logger.info("Qdrant symptom indexing completed successfully")

        # Print summary
        print_summary(symptoms_data, neo4j_counts, qdrant_count)

        # Run demo if requested
        if args.demo:
            demo_search()

        logger.info("Symptom database creation completed!")

    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        logger.error("Make sure Neo4j and/or Qdrant services are running")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
