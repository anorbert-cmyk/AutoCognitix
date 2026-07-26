"""
Qdrant vector database client and utilities with comprehensive error handling.

This module provides a service class for interacting with Qdrant vector database,
supporting both local and cloud deployments with Hungarian error messages.
"""

import math
import threading
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.core.config import settings
from app.core.exceptions import (
    QdrantConnectionException,
    QdrantException,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# The payload ``type`` discriminators that physically exist in the unified
# collection. Every vector lives in ONE collection and is told apart by this
# key, so the discriminator is now the only thing a caller can get wrong - and
# getting it wrong has the same signature as the collection-drift bug did: a
# type-correct query that matches nothing and returns [] with no error. It is
# therefore a CLOSED set, validated at the gate. Notably ``"symptom"`` is NOT
# in it: no vector was ever indexed with that type, and the retrieval leg that
# asked for it silently returned nothing.
UNIFIED_PAYLOAD_TYPES: FrozenSet[str] = frozenset({"dtc", "complaint", "recall"})

# Pre-unification per-type collections. THE ONLY PLACE THESE NAMES MAY APPEAR.
#
# Nothing reads or writes them: every search targets
# ``settings.QDRANT_UNIFIED_COLLECTION``. They are listed here for exactly two
# purposes - the GDPR erasure sweep (an instance seeded before the unification
# can still hold user-tagged points) and storage visibility, so an operator can
# see that they are still on the cluster.
#
# They are NOT empty on the production cluster: ``dtc_embeddings_hu`` holds
# ~2,323 points and ``symptom_embeddings_hu`` ~117, written by historical
# importer runs. That is a partial, stale copy of a 6,814-code corpus, which is
# worse than an empty one: querying it returns plausible-looking but incomplete
# results. Dropping the collections is a DATA decision reserved for a human -
# this module must never delete them, only stop creating and stop querying them.
_LEGACY_COLLECTIONS: Tuple[str, ...] = (
    "dtc_embeddings_hu",
    "symptom_embeddings_hu",
    "component_embeddings_hu",
    "repair_embeddings_hu",
    "known_issue_embeddings_hu",
)

# Minimum L2 norm a query vector must have to be worth searching with.
# Cosine distance normalizes the query vector, so a (near-)zero-norm vector
# degenerates to "every dot product is 0.0": it does not return BAD results, it
# returns MEANINGLESS ones - silently. That exact failure hid a broken
# production embedding path for months, so it is now a hard error at the gate.
MIN_QUERY_VECTOR_NORM = 1e-6


def _validate_query_vector(query_vector: List[float], collection_name: str) -> None:
    """
    Reject a degenerate query vector before it reaches Qdrant.

    A zero (or near-zero) vector is type-correct and dimension-correct, so every
    downstream check passes it - but under cosine distance every score collapses
    to 0.0 and any ``score_threshold`` turns the result into an empty list. That
    looks exactly like "no matches" and is indistinguishable from a healthy
    query, which is why the broken embedding path stayed invisible for months.

    Raises ``ValueError`` rather than a ``QdrantException``: nothing is wrong
    with Qdrant, the caller handed us an invalid vector. Every call site already
    wraps searches in a broad ``except Exception`` that degrades to the lexical /
    graph path, so this fails loudly in the logs without 500-ing an endpoint.

    Args:
        query_vector: The vector about to be searched with.
        collection_name: Target collection (for the error/log message only).

    Raises:
        ValueError: If the vector is empty, non-finite, or has norm < 1e-6.
    """
    if not query_vector:
        logger.error(
            "Rejected EMPTY query vector for collection %s - refusing to run a "
            "meaningless similarity search.",
            collection_name,
        )
        raise ValueError(
            f"Empty query vector for collection '{collection_name}'; "
            "refusing to run a similarity search."
        )

    norm = math.sqrt(math.fsum(float(v) * float(v) for v in query_vector))

    if not math.isfinite(norm) or norm < MIN_QUERY_VECTOR_NORM:
        logger.error(
            "Rejected degenerate query vector for collection %s (norm=%r, dim=%d). "
            "A zero-norm vector matches nothing under cosine distance - this "
            "usually means the embedding backend is unavailable.",
            collection_name,
            norm,
            len(query_vector),
        )
        raise ValueError(
            f"Degenerate query vector (norm={norm!r}) for collection "
            f"'{collection_name}'; refusing to run a similarity search."
        )


class QdrantService:
    """Service for interacting with Qdrant vector database."""

    # Embedding model version tracking
    EMBEDDING_MODEL_VERSION = "hubert-base-cc-v1"

    # Expected vector dimension for validation
    EXPECTED_DIMENSION = 768

    # Storage alert threshold (vectors per collection)
    STORAGE_WARN_THRESHOLD = 50000

    # NOTE: there are deliberately NO per-collection name constants on this
    # class. Every read and write goes to ``settings.QDRANT_UNIFIED_COLLECTION``
    # and no public method takes a collection name, so a caller cannot address
    # the wrong collection. Exporting ``DTC_COLLECTION = "dtc_embeddings_hu"``
    # is what let three separate call sites point at an all-but-empty store and
    # get an empty result instead of an error.

    def __init__(self):
        """Initialize Qdrant async client."""
        # Support both local Qdrant and Qdrant Cloud
        if settings.QDRANT_URL:
            # Qdrant Cloud configuration
            self.client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
            logger.info(f"Connected to Qdrant Cloud: {settings.QDRANT_URL}")
        else:
            # Local Qdrant configuration
            self.client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                prefer_grpc=True,
            )
            logger.info(f"Connected to local Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        self.vector_size = settings.EMBEDDING_DIMENSION

    async def initialize_collections(self) -> None:
        """Ensure the ONE collection the application uses exists.

        Previously this created the five per-type ``*_hu`` collections on every
        boot and did NOT create the unified one - it initialised everything
        except the store that actually holds the vectors.

        That was not merely useless, it manufactured the trap that hid the
        drift bug: a search against a collection that does not exist is a loud
        404 from Qdrant, while a search against an existing-but-(near-)empty
        one returns ``[]``, which is a legitimate answer for a search engine.
        By pre-creating those collections on every boot, the application
        guaranteed that a mis-addressed search would fail silently.

        Existing legacy collections are only reported, never created and never
        deleted - see :data:`_LEGACY_COLLECTIONS`.
        """
        await self._create_collection_if_not_exists(settings.QDRANT_UNIFIED_COLLECTION)

        surviving = await self._legacy_collections_present()
        if surviving:
            logger.warning(
                "Legacy Qdrant collections still present on the cluster: %s. Nothing "
                "reads or writes them; dropping them is a human data decision.",
                ", ".join(surviving),
            )

        logger.info(
            "Qdrant initialized: unified collection '%s' ready (768-dim huBERT, "
            "type-discriminated payloads)",
            settings.QDRANT_UNIFIED_COLLECTION,
        )

    async def _create_collection_if_not_exists(self, collection_name: str) -> None:
        """Create a collection if it doesn't exist."""
        try:
            collections_response = await self.client.get_collections()
            collections = collections_response.collections
            exists = any(c.name == collection_name for c in collections)

            if not exists:
                await self.client.create_collection(
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
        ids: List[str],
        vectors: List[List[float]],
        payloads: Optional[List[dict]] = None,
    ) -> None:
        """
        Upsert vectors into the unified collection.

        Takes no collection name for the same reason :meth:`search` does not:
        a hand-typed destination on the write side is how the stale partial
        copies in ``dtc_embeddings_hu`` / ``symptom_embeddings_hu`` came to
        exist in the first place.

        Args:
            ids: List of point IDs
            vectors: List of embedding vectors
            payloads: Optional list of metadata payloads. Each should carry a
                ``type`` key from :data:`UNIFIED_PAYLOAD_TYPES` so the point is
                discoverable by :meth:`search_unified`.
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

        await self.client.upsert(
            collection_name=settings.QDRANT_UNIFIED_COLLECTION,
            points=points,
        )

    async def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[dict] = None,
        score_threshold: Optional[float] = None,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Run a filtered similarity search against the one vector collection.

        There is intentionally no ``collection_name`` parameter. Every huBERT
        vector lives in ``settings.QDRANT_UNIFIED_COLLECTION`` and entities are
        told apart by a payload ``type`` discriminator, so a collection name is
        not a decision any caller gets to make. Making it an argument is what
        allowed three independent call sites to address an all-but-empty
        collection: each got ``[]``, which is a valid search result, so the
        flagship Hungarian semantic search was dead for months without a single
        error. Prefer :meth:`search_unified`, which also injects the type filter.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_conditions: Optional exact-match payload filters
            score_threshold: Minimum similarity score
            model_version: Filter by embedding model version (None = no filter)

        Returns:
            List of search results with scores and payloads

        Raises:
            ValueError: If ``query_vector`` is empty or (near-)zero-norm. This is
                the last line of defence against a degenerate vector reaching a
                similarity search - see :data:`MIN_QUERY_VECTOR_NORM`.
        """
        collection_name = settings.QDRANT_UNIFIED_COLLECTION
        _validate_query_vector(query_vector, collection_name)

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

        search_params: Dict[str, Any] = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "query_filter": qdrant_models.Filter(must=must_conditions),  # type: ignore[arg-type]
        }

        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold

        try:
            results = await self.client.search(**search_params)

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

    async def search_unified(
        self,
        query_vector: List[float],
        type_: str,
        limit: int = 10,
        extra_filters: Optional[Dict[str, str]] = None,
        score_threshold: Optional[float] = None,
        model_version: Optional[str] = None,
    ) -> List[dict]:
        """
        Search the unified collection, discriminating results by payload ``type``.

        All huBERT vectors (DTC/complaint/recall) are indexed into a single
        collection (``settings.QDRANT_UNIFIED_COLLECTION``, default
        ``autocognitix``) with a type-discriminated payload. This is the front
        door for every semantic search in the application: the collection is not
        a parameter, and the type discriminator is validated against a closed
        set, so neither half of the target can be silently wrong.

        Args:
            query_vector: Query embedding vector
            type_: Payload discriminator, one of :data:`UNIFIED_PAYLOAD_TYPES`
            limit: Maximum number of results
            extra_filters: Optional additional exact-match payload filters
            score_threshold: Minimum similarity score
            model_version: Filter by embedding model version (None = no filter)

        Returns:
            List of matching hits (id/score/payload) for the requested type.

        Raises:
            ValueError: If ``type_`` is not a discriminator that exists in the
                collection. An unknown type would match zero points and return
                ``[]`` - the exact silent-empty failure this module exists to
                make impossible.
        """
        if type_ not in UNIFIED_PAYLOAD_TYPES:
            raise ValueError(
                f"Unknown payload type {type_!r} for collection "
                f"'{settings.QDRANT_UNIFIED_COLLECTION}'; known types: "
                f"{sorted(UNIFIED_PAYLOAD_TYPES)}. An unknown type matches nothing "
                "and would look like an empty result set."
            )

        # Build the optional payload filters first, then set the ``type``
        # discriminator LAST so a caller-supplied ``type`` in ``extra_filters`` can
        # never clobber it. NOTE: the unified ``autocognitix`` collection carries no
        # ``_embedding_model_version`` payload, so callers must not pass ``model_version``.
        filter_conditions: Dict[str, str] = {k: v for k, v in (extra_filters or {}).items() if v}
        filter_conditions["type"] = type_

        return await self.search(
            query_vector=query_vector,
            limit=limit,
            filter_conditions=filter_conditions,
            score_threshold=score_threshold,
            model_version=model_version,
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

        Thin convenience wrapper over :meth:`search_unified` with
        ``type_="dtc"``. Hits carry ``code`` in their payload; callers enrich
        them to full records from PostgreSQL.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            category: Filter by DTC category
            severity: Filter by severity level
            model_version: Filter by embedding model version (defaults to current version)

        Returns:
            List of matching DTC codes with similarity scores
        """
        extra_filters: Dict[str, str] = {}
        if category:
            extra_filters["category"] = category
        if severity:
            extra_filters["severity"] = severity

        return await self.search_unified(
            query_vector=query_vector,
            type_="dtc",
            limit=limit,
            extra_filters=extra_filters or None,
            model_version=model_version,
        )

    # NOTE: ``search_similar_symptoms`` / ``search_components`` /
    # ``search_repairs`` were removed. Each targeted one of the per-type
    # collections in :data:`_LEGACY_COLLECTIONS` and had no production caller -
    # only their own unit tests and a few pre-configured mock attributes. They
    # could not have worked: ``component_embeddings_hu`` and
    # ``repair_embeddings_hu`` hold no points at all, and
    # ``symptom_embeddings_hu`` holds ~117 stale ones. Their replacement is
    # ``search_unified(type_=...)``.

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

    async def _legacy_collections_present(self) -> List[str]:
        """Which of the pre-unification per-type collections actually exist.

        On the production cluster several of them still do, holding stale
        partial data (see :data:`_LEGACY_COLLECTIONS`). Nothing queries them,
        but the GDPR sweep must still visit the ones that are there, and
        deleting from a MISSING collection is a 404 from Qdrant rather than a
        real erasure failure. Probing first is what lets :meth:`delete_by_user`
        treat every remaining failure as fatal instead of having to swallow the
        benign case.

        A probe failure is logged and treated as "none present": the unified
        delete right after it is the one that decides the outcome, and it will
        surface the same outage far less ambiguously.
        """
        try:
            existing = {c.name for c in (await self.client.get_collections()).collections}
        except Exception as e:
            logger.warning(
                "Could not enumerate Qdrant collections",
                extra={"error_type": type(e).__name__},
            )
            return []
        return [name for name in _LEGACY_COLLECTIONS if name in existing]

    async def delete_by_user(self, user_id: str) -> int:
        """
        Delete all vectors associated with a user (GDPR Article 17).

        Targets ``settings.QDRANT_UNIFIED_COLLECTION`` first - that is where
        every write path has put vectors since the unification, so it is the
        only collection that can actually hold the user's data. Before this,
        the sweep iterated ONLY the legacy per-type collections, all of which
        are documented as never populated, which meant erasure touched nothing
        at all while reporting success.

        The legacy sweep is kept, but only for collections that are actually
        present: an instance seeded before the unification can still hold
        user-tagged points in them, and dropping the sweep would strand those
        forever with no way to notice. This is the ONLY code path that still
        addresses those collections, and it deletes points BY USER ID - it
        never drops a collection.

        Failures are NOT swallowed. A ``logger.warning`` inside the loop used to
        leave ``cleanup_errors`` empty in ``DELETE /api/v1/auth/me``, so a failed
        purge was reported to the data subject as a completed erasure - the one
        outcome Article 17 does not permit. Any failure now propagates, and that
        endpoint's ``except Exception`` turns it into a partial-deletion error
        the caller can retry.

        Args:
            user_id: The user ID whose vectors should be deleted

        Returns:
            Number of collections a delete was successfully executed against.

        Raises:
            QdrantException: If ANY targeted collection could not be purged.
        """
        selector = qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="user_id",
                        match=qdrant_models.MatchValue(value=user_id),
                    )
                ]
            )
        )

        targets = [settings.QDRANT_UNIFIED_COLLECTION]
        targets += [c for c in await self._legacy_collections_present() if c not in targets]

        collections_processed = 0
        failed: List[str] = []
        for collection in targets:
            try:
                await self.client.delete(
                    collection_name=collection,
                    points_selector=selector,
                )
                collections_processed += 1
            except Exception as e:
                failed.append(collection)
                logger.error(
                    "GDPR erasure failed for a Qdrant collection",
                    extra={"collection": collection, "error_type": type(e).__name__},
                )

        if failed:
            raise QdrantException(
                message="A felhasznaloi vektorok torlese nem sikerult minden gyujtemenyben.",
                details={
                    "failed_collections": failed,
                    "deleted_collections": collections_processed,
                },
            )

        return collections_processed

    # NOTE: ``delete_collection(name)`` was removed. It had no production
    # caller, and it was the only code path by which the application could drop
    # a Qdrant collection wholesale - including the legacy ones that turned out
    # to hold thousands of real points. Dropping a collection is a data
    # decision for a human at the Qdrant console, not an API this service
    # should expose. ``delete_by_user`` remains: it deletes POINTS by user id.

    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get read-only information about a collection.

        This one does take a name: it is introspection, not retrieval. Naming
        the wrong collection here surfaces as an error or an obviously wrong
        count, never as a silently empty search result.
        """
        info = await self.client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vectors_count": getattr(
                info, "indexed_vectors_count", getattr(info, "vectors_count", 0)
            ),
            "points_count": info.points_count,
            "status": info.status,
        }

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the vector store.

        Always reports ``settings.QDRANT_UNIFIED_COLLECTION`` FIRST - it is the
        only collection the application reads or writes, and until now it was
        the one collection these stats did not cover. Enumerating just the five
        legacy names meant the real store (~60k vectors, above
        :data:`STORAGE_WARN_THRESHOLD`) was completely unmonitored while the
        alerting looked healthy.

        Legacy collections are included only when they physically exist, so an
        operator can see the stale points that are still on the cluster and
        decide whether to drop them. Their absence from this dict is the signal
        that the cleanup is done.
        """
        all_collections = [settings.QDRANT_UNIFIED_COLLECTION]
        all_collections += [
            name for name in await self._legacy_collections_present() if name not in all_collections
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


# Lazy global instance -initialised on first access so that importing
# this module does not open a network connection (important for tests/CI
# where Qdrant may not be running).
_qdrant_instance: Optional[QdrantService] = None
_qdrant_lock = threading.Lock()


def _get_qdrant_instance() -> QdrantService:
    """Return (and lazily create) the global QdrantService singleton."""
    global _qdrant_instance
    if _qdrant_instance is None:
        with _qdrant_lock:
            if _qdrant_instance is None:
                _qdrant_instance = QdrantService()
    return _qdrant_instance


class _LazyQdrantProxy:
    """Transparent proxy that defers QdrantService creation until first use."""

    def __getattr__(self, name: str):
        return getattr(_get_qdrant_instance(), name)


# Importable module-level name -behaves like QdrantService but is lazy.
qdrant_client: QdrantService = _LazyQdrantProxy()  # type: ignore[assignment]


async def get_qdrant_service() -> QdrantService:
    """Get the global Qdrant service instance."""
    return _get_qdrant_instance()
