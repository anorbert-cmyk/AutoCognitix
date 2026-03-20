"""
Qdrant vector database client and utilities with comprehensive error handling.

This module provides a service class for interacting with Qdrant vector database,
supporting both local and cloud deployments with Hungarian error messages.
"""

import asyncio
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.core.config import settings
from app.core.exceptions import (
    QdrantConnectionException,
    QdrantException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    # Embedding model version tracking
    EMBEDDING_MODEL_VERSION = "hubert-base-cc-v1"

    # Expected vector dimension for validation
    EXPECTED_DIMENSION = 768

    # Collection names - Hungarian versions with huBERT embeddings (768-dim)
    DTC_COLLECTION = "dtc_embeddings_hu"
    SYMPTOM_COLLECTION = "symptom_embeddings_hu"
    COMPONENT_COLLECTION = "component_embeddings_hu"
    REPAIR_COLLECTION = "repair_embeddings_hu"
    ISSUE_COLLECTION = "known_issue_embeddings_hu"

    # Storage alert threshold (vectors per collection)
    STORAGE_WARN_THRESHOLD = 50000

    # Legacy collection names (English, for backwards compatibility)
    DTC_COLLECTION_LEGACY = "dtc_embeddings"
    SYMPTOM_COLLECTION_LEGACY = "symptom_embeddings"
    ISSUE_COLLECTION_LEGACY = "known_issue_embeddings"

    def __init__(self):
        """Initialize Qdrant client."""
        # Support both local Qdrant and Qdrant Cloud
        if settings.QDRANT_URL:
            # Qdrant Cloud configuration
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
            logger.info(f"Connected to Qdrant Cloud: {settings.QDRANT_URL}")
        else:
            # Local Qdrant configuration
            self.client = QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                prefer_grpc=True,
            )
            logger.info(f"Connected to local Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        self.vector_size = settings.EMBEDDING_DIMENSION

    async def initialize_collections(self) -> None:
        """Initialize all required collections."""
        collections = [
            (self.DTC_COLLECTION, "DTC code embeddings (Hungarian huBERT, 768-dim)"),
            (self.SYMPTOM_COLLECTION, "Symptom text embeddings (Hungarian huBERT, 768-dim)"),
            (self.COMPONENT_COLLECTION, "Vehicle component embeddings (Hungarian huBERT, 768-dim)"),
            (self.REPAIR_COLLECTION, "Repair procedure embeddings (Hungarian huBERT, 768-dim)"),
            (self.ISSUE_COLLECTION, "Known issue embeddings (Hungarian huBERT, 768-dim)"),
        ]

        for collection_name, description in collections:
            await self._create_collection_if_not_exists(collection_name)
            logger.debug(f"  - {collection_name}: {description}")

        logger.info(f"Qdrant collections initialized ({len(collections)} collections)")

    async def _create_collection_if_not_exists(self, collection_name: str) -> None:
        """Create a collection if it doesn't exist."""
        try:
            collections_response = await asyncio.to_thread(self.client.get_collections)
            collections = collections_response.collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.vector_size,
                        distance=qdrant_models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created collection: {collection_name}")
            else:
                logger.info(f"Collection already exists: {collection_name}")

        except ConnectionError as e:
            logger.error(
                f"Qdrant connection error while creating collection {collection_name}",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
            raise QdrantConnectionException(
                message="Nem sikerult csatlakozni a Qdrant adatbazishoz.",
                original_error=e,
            )
        except Exception as e:
            logger.error(
                f"Error creating collection {collection_name}",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
            raise QdrantException(
                message="Qdrant vektor adatbazis hiba.",
                details={"collection": collection_name},
                original_error=e,
            )

    async def upsert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        """
        Upsert vectors into a collection.

        Args:
            collection_name: Name of the collection
            ids: List of point IDs
            vectors: List of embedding vectors
            payloads: Optional list of metadata payloads
        """
        # Validate vector dimensions
        for i, vec in enumerate(vectors):
            if len(vec) != self.EXPECTED_DIMENSION:
                raise ValueError(
                    f"Vector dimension mismatch at index {i}: "
                    f"expected {self.EXPECTED_DIMENSION}, got {len(vec)}"
                )

        # Inject embedding model version into each payload
        resolved_payloads = list(payloads) if payloads else [{} for _ in ids]
        for payload in resolved_payloads:
            payload["_embedding_model_version"] = self.EMBEDDING_MODEL_VERSION

        points = [
            qdrant_models.PointStruct(
                id=id_,
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(ids, vectors, resolved_payloads)
        ]

        await asyncio.to_thread(
            self.client.upsert,
            collection_name=collection_name,
            points=points,
        )

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[dict] = None,
        score_threshold: Optional[float] = None,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_conditions: Optional Qdrant filter conditions
            score_threshold: Minimum similarity score
            model_version: Filter by embedding model version (defaults to current version)

        Returns:
            List of search results with scores and payloads
        """
        # Build the must filter list
        must_conditions: List[qdrant_models.FieldCondition] = []

        # Only filter by version when explicitly requested (avoids hiding pre-existing vectors)
        if model_version is not None:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="_embedding_model_version",
                    match=qdrant_models.MatchValue(value=model_version),
                )
            )

        # Add user-supplied filter conditions
        if filter_conditions:
            for key, value in filter_conditions.items():
                must_conditions.append(
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                )

        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "query_filter": qdrant_models.Filter(must=must_conditions),
        }

        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold

        try:
            results = await asyncio.to_thread(lambda: self.client.search(**search_params))

            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                }
                for result in results
            ]
        except ConnectionError as e:
            logger.error(
                f"Qdrant connection error during search in {collection_name}",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
            raise QdrantConnectionException(
                message="Nem sikerult csatlakozni a Qdrant adatbazishoz.",
                original_error=e,
            )
        except Exception as e:
            logger.error(
                f"Qdrant search error in {collection_name}",
                extra={"error_type": type(e).__name__, "error_message": str(e)},
            )
            raise QdrantException(
                message="Vektor kereses sikertelen.",
                details={"collection": collection_name},
                original_error=e,
            )

    async def search_dtc(
        self,
        query_vector: List[float],
        limit: int = 10,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for similar DTC codes.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            category: Filter by DTC category
            severity: Filter by severity level
            model_version: Filter by embedding model version (defaults to current version)

        Returns:
            List of matching DTC codes with similarity scores
        """
        filter_conditions = {}
        if category:
            filter_conditions["category"] = category
        if severity:
            filter_conditions["severity"] = severity

        return await self.search(
            collection_name=self.DTC_COLLECTION,
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions if filter_conditions else None,
            model_version=model_version,
        )

    async def search_similar_symptoms(
        self,
        query_vector: List[float],
        limit: int = 10,
        vehicle_make: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for similar symptom descriptions.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            vehicle_make: Filter by vehicle make
            model_version: Filter by embedding model version (defaults to current version)

        Returns:
            List of similar symptoms with scores
        """
        filter_conditions = {}
        if vehicle_make:
            filter_conditions["vehicle_make"] = vehicle_make

        return await self.search(
            collection_name=self.SYMPTOM_COLLECTION,
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions if filter_conditions else None,
            model_version=model_version,
        )

    async def search_components(
        self,
        query_vector: List[float],
        limit: int = 10,
        system: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for similar vehicle components.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            system: Filter by vehicle system (engine, transmission, etc.)
            model_version: Filter by embedding model version (defaults to current version)

        Returns:
            List of similar components with scores
        """
        filter_conditions = {}
        if system:
            filter_conditions["system"] = system

        return await self.search(
            collection_name=self.COMPONENT_COLLECTION,
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions if filter_conditions else None,
            model_version=model_version,
        )

    async def search_repairs(
        self,
        query_vector: List[float],
        limit: int = 10,
        difficulty: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Search for similar repair procedures.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            difficulty: Filter by difficulty level (beginner, intermediate, advanced, professional)
            model_version: Filter by embedding model version (defaults to current version)

        Returns:
            List of similar repairs with scores
        """
        filter_conditions = {}
        if difficulty:
            filter_conditions["difficulty"] = difficulty

        return await self.search(
            collection_name=self.REPAIR_COLLECTION,
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions if filter_conditions else None,
            model_version=model_version,
        )

    async def check_storage_alerts(self) -> List[dict]:
        """Check if any collection is approaching storage limits.

        Returns:
            List of alert dicts for collections exceeding STORAGE_WARN_THRESHOLD.
        """
        alerts: List[dict] = []
        stats = await self.get_storage_stats()
        for collection, info in stats.items():
            if isinstance(info, dict) and "error" not in info:
                count = info.get("points_count", 0)
                if count > self.STORAGE_WARN_THRESHOLD:
                    alerts.append(
                        {
                            "collection": collection,
                            "count": count,
                            "threshold": self.STORAGE_WARN_THRESHOLD,
                            "severity": "warning",
                            "message": (
                                f"Collection {collection} has {count} vectors "
                                f"(threshold: {self.STORAGE_WARN_THRESHOLD})"
                            ),
                        }
                    )
        return alerts

    async def delete_by_user(self, user_id: str) -> int:
        """
        Delete all vectors associated with a user (GDPR Article 17).

        Args:
            user_id: The user ID whose vectors should be deleted

        Returns:
            Number of collections processed
        """
        collections_processed = 0
        for collection in [
            self.DTC_COLLECTION,
            self.SYMPTOM_COLLECTION,
            self.COMPONENT_COLLECTION,
            self.REPAIR_COLLECTION,
            self.ISSUE_COLLECTION,
        ]:
            try:
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name=collection,
                    points_selector=qdrant_models.FilterSelector(
                        filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="user_id",
                                    match=qdrant_models.MatchValue(value=user_id),
                                )
                            ]
                        )
                    ),
                )
                collections_processed += 1
            except Exception as e:
                logger.warning(f"Failed to delete user vectors from {collection}: {e}")
        return collections_processed

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        await asyncio.to_thread(self.client.delete_collection, collection_name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        info = await asyncio.to_thread(self.client.get_collection, collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": getattr(
                info, "indexed_vectors_count", getattr(info, "vectors_count", 0)
            ),
            "points_count": info.points_count,
            "status": info.status,
        }

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for all collections."""
        all_collections = [
            self.DTC_COLLECTION,
            self.SYMPTOM_COLLECTION,
            self.COMPONENT_COLLECTION,
            self.REPAIR_COLLECTION,
            self.ISSUE_COLLECTION,
        ]
        stats: Dict[str, Any] = {}
        for collection in all_collections:
            try:
                info = await self.get_collection_info(collection)
                if info:
                    stats[collection] = info
            except Exception:
                stats[collection] = {"error": "unavailable"}
        return stats


# Global instance
qdrant_client = QdrantService()


async def get_qdrant_service() -> QdrantService:
    """Get the global Qdrant service instance."""
    return qdrant_client
