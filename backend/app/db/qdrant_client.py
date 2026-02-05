"""
Qdrant vector database client and utilities with comprehensive error handling.

This module provides a service class for interacting with Qdrant vector database,
supporting both local and cloud deployments with Hungarian error messages.
"""

from typing import Any

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

    # Collection names - Hungarian versions with huBERT embeddings (768-dim)
    DTC_COLLECTION = "dtc_embeddings_hu"
    SYMPTOM_COLLECTION = "symptom_embeddings_hu"
    COMPONENT_COLLECTION = "component_embeddings_hu"
    REPAIR_COLLECTION = "repair_embeddings_hu"
    ISSUE_COLLECTION = "known_issue_embeddings_hu"

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
            collections = self.client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                self.client.create_collection(
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
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict] | None = None,
    ) -> None:
        """
        Upsert vectors into a collection.

        Args:
            collection_name: Name of the collection
            ids: List of point IDs
            vectors: List of embedding vectors
            payloads: Optional list of metadata payloads
        """
        points = [
            qdrant_models.PointStruct(
                id=id_,
                vector=vector,
                payload=payload if payloads else {},
            )
            for id_, vector, payload in zip(
                ids, vectors, payloads or [{}] * len(ids), strict=False
            )
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    async def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: dict | None = None,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        Search for similar vectors.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_conditions: Optional Qdrant filter conditions
            score_threshold: Minimum similarity score

        Returns:
            List of search results with scores and payloads
        """
        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True,
        }

        if filter_conditions:
            search_params["query_filter"] = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key=key,
                        match=qdrant_models.MatchValue(value=value),
                    )
                    for key, value in filter_conditions.items()
                ]
            )

        if score_threshold:
            search_params["score_threshold"] = score_threshold

        try:
            results = self.client.search(**search_params)

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
        query_vector: list[float],
        limit: int = 10,
        category: str | None = None,
        severity: str | None = None,
    ) -> list[dict]:
        """
        Search for similar DTC codes.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            category: Filter by DTC category
            severity: Filter by severity level

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
        )

    async def search_similar_symptoms(
        self,
        query_vector: list[float],
        limit: int = 10,
        vehicle_make: str | None = None,
    ) -> list[dict]:
        """
        Search for similar symptom descriptions.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            vehicle_make: Filter by vehicle make

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
        )

    async def search_components(
        self,
        query_vector: list[float],
        limit: int = 10,
        system: str | None = None,
    ) -> list[dict]:
        """
        Search for similar vehicle components.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            system: Filter by vehicle system (engine, transmission, etc.)

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
        )

    async def search_repairs(
        self,
        query_vector: list[float],
        limit: int = 10,
        difficulty: str | None = None,
    ) -> list[dict]:
        """
        Search for similar repair procedures.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            difficulty: Filter by difficulty level (beginner, intermediate, advanced, professional)

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
        )

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        self.client.delete_collection(collection_name=collection_name)
        logger.info(f"Deleted collection: {collection_name}")

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get information about a collection."""
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
        }


# Global instance
qdrant_client = QdrantService()
