"""
Regression tests for the adversarial review of the RAG + embedding stack.

Each class pins ONE defect found by that review. Every test in this module was
verified to FAIL against the pre-fix code, so reintroducing any of these bugs
breaks the suite instead of silently degrading a diagnosis:

* B1 - Reciprocal Rank Fusion overwrote the cosine similarity of the items it
  ranked. Those objects are the ones ``retrieve_from_qdrant`` caches and hands
  back by reference, and ``calculate_confidence`` reads their ``score`` as a
  similarity, so the confidence shown to the user collapsed from ~50% to ~1% on
  every diagnosis with DTC hits, and the corrupted values stayed in the cache.
* B2 - the embedding singleton published ``_initialized = True`` before it had
  assigned ``_backend_name``, so a second thread could get a half-built
  instance (``AttributeError``, which bypasses the EmbeddingUnavailableError
  handler and degrades semantic search silently).
* B3 - the ONNX backend pooled with ``encoded["attention_mask"]`` taken from the
  dict it had just filtered down to the graph's declared inputs, so a graph
  exported without the mask input raised ``KeyError: 'attention_mask'``.
* B4 - the Qdrant retrieval cache key omitted ``top_k``/``score_threshold``, so a
  ``top_k=3`` chat lookup served its 3 items to a later ``top_k=10`` diagnosis.
* B5 - the "similar cases" leg read ``description`` from complaint payloads that
  the current indexer writes WITHOUT any narrative, producing blank numbered
  lines in the Hungarian prompt and inflating the confidence.
* B6 - blank inputs produced ``[0.0] * 768``, which was returned as if real and
  written into the ``v2`` Redis namespace the version bump exists to keep clean.

Everything here is offline: no Qdrant, no Redis, no model download.
"""

import threading
from functools import partial
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import EmbeddingUnavailableError

# =============================================================================
# Shared fixtures / doubles
# =============================================================================

# Payload written by scripts/index_qdrant_hubert.py: narrative under "description".
COMPLAINT_HITS_WITH_NARRATIVE = [
    {
        "id": 2,
        "score": 0.71,
        "payload": {
            "type": "complaint",
            "odi_id": "11554321",
            "make": "VOLKSWAGEN",
            "model": "GOLF",
            "year": "2018",
            "component": "ENGINE",
            "description": "Engine misfires and shakes at idle",
        },
    }
]

# Payload written by scripts/sync_qdrant_sprint9.py: the narrative is embedded
# but deliberately NOT stored, so these hits carry no readable symptom text.
COMPLAINT_HITS_WITHOUT_NARRATIVE = [
    {
        "id": 3,
        "score": 0.78,
        "payload": {
            "type": "complaint",
            "source": "nhtsa_flat",
            "odi_number": "11554322",
            "make": "VOLKSWAGEN",
            "model": "GOLF",
            "model_year": 2018,
            "component": "ENGINE",
            "crash": False,
            "fire": False,
            "injuries": 0,
            "deaths": 0,
        },
    }
]


def _dtc_hits(scores: List[float]) -> List[Dict[str, Any]]:
    """Unified-collection DTC hits with explicit cosine similarities."""
    return [
        {
            "id": idx,
            "score": score,
            "payload": {
                "type": "dtc",
                "code": f"P030{idx}",
                "description": f"{idx}. henger egeskimaradas",
                "category": "powertrain",
            },
        }
        for idx, score in enumerate(scores, 1)
    ]


def _unified_qdrant_mock(hits_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None):
    """QdrantService double whose ``search_unified`` is the REAL one.

    Only the low-level ``search`` is stubbed - and it honours ``limit``, so a
    ``top_k`` that never reaches Qdrant is observable.
    """
    from app.db.qdrant_client import QdrantService

    mock = MagicMock()

    async def _search(**kwargs):
        payload_type = (kwargs.get("filter_conditions") or {}).get("type")
        hits = list((hits_by_type or {}).get(payload_type, []))
        limit = kwargs.get("limit")
        return hits[:limit] if limit else hits

    mock.search = AsyncMock(side_effect=_search)
    mock.search_unified = partial(QdrantService.search_unified, mock)
    return mock


@pytest.fixture
def rag_service():
    """RAGService singleton with a clean cache, restored after the test."""
    from app.services.rag_service import RAGService

    service = RAGService()
    original_qdrant = service._qdrant
    service._cache.clear()
    service.set_db_session(None)
    yield service
    service._qdrant = original_qdrant
    service._cache.clear()


def _offline_rag_patches():
    """Patch the embedding / spaCy / Neo4j legs so assembly runs offline."""
    return (
        patch("app.services.rag_service.embed_text_async", new=AsyncMock(return_value=[0.1] * 768)),
        patch("app.services.rag_service.preprocess_hungarian", side_effect=lambda text: text),
        patch("app.services.rag_service.get_diagnostic_path", new=AsyncMock(return_value={})),
    )


async def _assemble(rag_service, symptoms: str = "rangat a motor alapjaraton"):
    from app.services.rag_service import VehicleInfo

    embed_patch, preprocess_patch, graph_patch = _offline_rag_patches()
    with embed_patch, preprocess_patch, graph_patch:
        return await rag_service.assemble_context(
            vehicle_info=VehicleInfo(make="Volkswagen", model="Golf", year=2018),
            dtc_codes=["P0301"],
            symptoms=symptoms,
        )


@pytest.fixture
def reset_embedding_singleton():
    """Reset the embedding singleton before and after the test."""
    import app.services.embedding_service as mod

    def _reset():
        mod.HungarianEmbeddingService._instance = None
        mod.HungarianEmbeddingService._initialized = False
        mod._embedding_service = None

    _reset()
    yield mod
    _reset()


# =============================================================================
# B1 - RRF must not overwrite the similarity the confidence score reads
# =============================================================================


class TestRRFDoesNotCorruptSimilarity:
    """Ranking re-orders items; it must not re-score the retrieved objects."""

    @pytest.mark.asyncio
    async def test_confidence_magnitude_reflects_cosine_similarity_after_rrf(self, rag_service):
        """PINS THE MAGNITUDE of the user-facing confidence.

        Two DTC hits at cosine 0.85 / 0.83: factor 1 contributes
        avg(0.84) * 0.3 = 0.252 of the 0.7 counted weight (factors 1, 2 and 4
        are in play), i.e. 36.0%. With RRF writing 1/(60+rank) back into those
        items the SAME diagnosis reported 0.7% - a ~35 point understatement on
        every diagnosis that has DTC hits.
        """
        rag_service._qdrant = _unified_qdrant_mock({"dtc": _dtc_hits([0.85, 0.83])})

        context = await _assemble(rag_service)

        assert [item.score for item in context.dtc_items] == [0.85, 0.83]

        level, score = rag_service.calculate_confidence(context, ["P0301"])

        assert score == pytest.approx(0.360, abs=1e-3)
        assert str(level) == "low"

    @pytest.mark.asyncio
    async def test_rrf_does_not_mutate_the_cached_retrieval_items(self, rag_service):
        """The cache hands back the same objects RRF ranked - they must be intact."""
        rag_service._qdrant = _unified_qdrant_mock({"dtc": _dtc_hits([0.85, 0.83])})

        embed_patch, _, _ = _offline_rag_patches()
        with embed_patch:
            first = await rag_service.retrieve_from_qdrant(
                query="P0301", type_="dtc", top_k=10, preprocess=False
            )

            fused = rag_service._ranker.reciprocal_rank_fusion([first])

            second = await rag_service.retrieve_from_qdrant(
                query="P0301", type_="dtc", top_k=10, preprocess=False
            )

        # Served from cache (one Qdrant round-trip), with similarities intact.
        assert rag_service._qdrant.search.await_count == 1
        assert [item.score for item in second] == [0.85, 0.83]
        assert [item.score for item in first] == [0.85, 0.83]

        # The fused list is a separate set of objects carrying the RRF score.
        assert fused[0] is not first[0]
        assert fused[0].score == pytest.approx(1 / 61)

    def test_rrf_ordering_is_unchanged_by_the_copying(self):
        """An item present in both lists still outranks a single-list item."""
        from app.services.rag_service import HybridRanker, RetrievalSource, RetrievedItem

        ranker = HybridRanker(k=60)
        list_a = [
            RetrievedItem(content={"id": "a"}, source=RetrievalSource.QDRANT_DTC, score=0.90),
            RetrievedItem(content={"id": "b"}, source=RetrievalSource.QDRANT_DTC, score=0.80),
        ]
        list_b = [
            RetrievedItem(content={"id": "b"}, source=RetrievalSource.POSTGRES_TEXT, score=0.95),
            RetrievedItem(content={"id": "c"}, source=RetrievalSource.POSTGRES_TEXT, score=0.70),
        ]

        fused = ranker.reciprocal_rank_fusion([list_a, list_b])

        assert [item.content["id"] for item in fused] == ["b", "a", "c"]
        # ... and the inputs keep their similarities.
        assert [item.score for item in list_a] == [0.90, 0.80]
        assert [item.score for item in list_b] == [0.95, 0.70]

    def test_postgres_direct_matches_survive_ranking(self):
        """Factor 2 counts ``score >= 1.0`` text hits; RRF used to erase them."""
        from app.services.rag_service import (
            RAGContext,
            RAGService,
            RetrievalSource,
            RetrievedItem,
        )

        service = RAGService()
        context = RAGContext(
            dtc_items=[
                RetrievedItem(
                    content={"code": "P0301"}, source=RetrievalSource.QDRANT_DTC, score=0.8
                )
            ],
            text_items=[
                RetrievedItem(
                    content={"code": "P0301"}, source=RetrievalSource.POSTGRES_TEXT, score=1.0
                )
            ],
        )

        service._ranker.reciprocal_rank_fusion(
            [context.dtc_items, context.text_items], weights=[1.0, 0.8]
        )

        assert context.text_items[0].score == 1.0
        _level, score = service.calculate_confidence(context, ["P0301"])
        # 0.8 * 0.3 (factor 1) + 1.0 * 0.2 (factor 2) over 0.5 counted weight.
        assert score == pytest.approx(0.88, abs=1e-3)

    def test_normalize_scores_also_leaves_its_inputs_alone(self):
        """Same ranker contract: no in-place re-scoring of cached objects."""
        from app.services.rag_service import HybridRanker, RetrievalSource, RetrievedItem

        items = [
            RetrievedItem(content={"id": 1}, source=RetrievalSource.QDRANT_DTC, score=0.9),
            RetrievedItem(content={"id": 2}, source=RetrievalSource.QDRANT_DTC, score=0.5),
        ]

        normalized = HybridRanker().normalize_scores(items)

        assert [item.score for item in normalized] == [1.0, 0.0]
        assert [item.score for item in items] == [0.9, 0.5]


# =============================================================================
# B2 - the embedding singleton must never publish a half-built instance
# =============================================================================


class TestEmbeddingSingletonInitializationIsAtomic:
    """``_initialized`` is the single publish point, and it is published last."""

    def test_concurrent_get_never_sees_a_half_initialized_instance(self, reset_embedding_singleton):
        """Deterministic race: thread B asks for the service while thread A is
        still inside ``__init__``, then reads the attribute ``_require_backend``
        depends on. Before the fix this raised
        ``AttributeError: ... has no attribute '_backend_name'``.
        """
        mod = reset_embedding_singleton

        inside_init = threading.Event()
        may_finish_init = threading.Event()

        def slow_select_backend(self):
            inside_init.set()
            # Held here until the racing thread has had its chance.
            may_finish_init.wait(timeout=10)
            return "onnx"

        errors: List[BaseException] = []
        observed: List[Any] = []

        def racer():
            try:
                service = mod.get_embedding_service()
                # Both readers of the attribute that used not to exist yet - and
                # they must see the RESOLVED value, not just "no AttributeError".
                observed.append(service.backend_name)
                observed.append(service._require_backend())
            except BaseException as exc:
                errors.append(exc)

        with patch.object(
            mod.HungarianEmbeddingService, "_select_backend_name", slow_select_backend
        ):
            first = threading.Thread(target=lambda: observed.append(mod.get_embedding_service()))
            first.start()
            assert inside_init.wait(timeout=10), "initializer never started"

            second = threading.Thread(target=racer)
            second.start()
            # The racer is now either blocked on the init lock or (pre-fix)
            # already using the published instance.
            second.join(timeout=0.5)

            may_finish_init.set()
            first.join(timeout=10)
            second.join(timeout=10)

        assert not first.is_alive() and not second.is_alive()
        assert errors == [], f"racing thread saw {errors!r}"
        # The racer saw the resolved backend, not a half-built instance.
        assert observed[-2:] == ["onnx", "onnx"]
        assert mod.get_embedding_service()._initialized is True

    def test_initialized_is_the_last_attribute_assigned_by_init(self):
        """Guard the ordering itself: nothing may be assigned after the publish."""
        import ast
        import inspect
        import textwrap

        from app.services.embedding_service import HungarianEmbeddingService

        tree = ast.parse(textwrap.dedent(inspect.getsource(HungarianEmbeddingService.__init__)))
        assignments = sorted(
            (node.lineno, node.targets[0].attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        )

        assert assignments, "__init__ assigns nothing?"
        assert assignments[-1][1] == "_initialized", (
            f"attributes are assigned AFTER the publish point ({assignments}) - "
            "a concurrent caller can observe the instance without them"
        )

    def test_backend_name_has_a_class_level_default(self):
        """Even an unpublished instance must fail with the HANDLED error."""
        from app.services.embedding_service import HungarianEmbeddingService

        raw = object.__new__(HungarianEmbeddingService)
        assert raw._backend_name is None
        with pytest.raises(EmbeddingUnavailableError):
            raw._require_backend()


# =============================================================================
# B3 - ONNX pooling must not depend on the graph declaring the mask input
# =============================================================================


class _FoldedGraphSession:
    """ONNX session double for a graph whose only input is ``input_ids``.

    ``torch.onnx.export(do_constant_folding=True)`` traced on an all-ones mask
    folds the mask input away - exactly the graph the input filter exists for.
    """

    def __init__(self, hidden: np.ndarray) -> None:
        self._hidden = hidden
        self.last_feeds: Optional[Dict[str, np.ndarray]] = None

    def run(self, output_names, feeds):
        self.last_feeds = feeds
        return [self._hidden]


class _Encoding:
    def __init__(self, ids, attention_mask, type_ids) -> None:
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


class _PaddingTokenizer:
    """Two sequences of different length, so padding (and the mask) matters."""

    ENCODINGS = [
        _Encoding([101, 11, 12, 102], [1, 1, 1, 1], [0, 0, 0, 0]),
        _Encoding([101, 21, 102, 0], [1, 1, 1, 0], [0, 0, 0, 0]),
    ]

    def encode_batch(self, texts):
        assert len(texts) == len(self.ENCODINGS)
        return self.ENCODINGS


class TestOnnxPoolingWithFoldedMaskInput:
    def _backend(self):
        from app.services.embedding_service import _OnnxEmbeddingBackend

        # Distinct per-token values so an all-ones mask pools differently.
        hidden = np.arange(2 * 4 * 768, dtype=np.float32).reshape(2, 4, 768) * 0.001
        backend = _OnnxEmbeddingBackend.__new__(_OnnxEmbeddingBackend)
        backend._input_names = {"input_ids"}
        backend._tokenizer = _PaddingTokenizer()
        backend._session = _FoldedGraphSession(hidden)
        return backend, hidden

    def test_embed_succeeds_when_the_graph_declares_only_input_ids(self):
        """Used to raise ``KeyError: 'attention_mask'``."""
        from app.services.embedding_service import _mean_pool_l2_numpy

        backend, hidden = self._backend()

        vectors = backend.embed(["elso szoveg", "masodik"])

        # The feed dict still respects the graph's declared inputs.
        assert set(backend._session.last_feeds) == {"input_ids"}

        mask = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=np.int64)
        expected = _mean_pool_l2_numpy(hidden, mask)

        assert len(vectors) == 2
        assert len(vectors[0]) == 768
        np.testing.assert_allclose(np.array(vectors), expected, rtol=1e-6, atol=1e-6)

        # L2-normalized...
        norms = np.linalg.norm(np.array(vectors), axis=1)
        np.testing.assert_allclose(norms, np.ones(2), rtol=1e-5, atol=1e-5)

        # ...and genuinely mask-weighted: padding must not be pooled in.
        all_ones = _mean_pool_l2_numpy(hidden, np.ones((2, 4), dtype=np.int64))
        assert not np.allclose(np.array(vectors)[1], all_ones[1])

    def test_encode_still_returns_only_declared_graph_inputs(self):
        """The filtering behaviour for the ONNX feed is preserved."""
        backend, _ = self._backend()

        assert set(backend.encode(["a", "b"])) == {"input_ids"}
        assert set(backend.tokenize(["a", "b"])) == {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }


# =============================================================================
# B4 - the retrieval cache key must cover top_k and score_threshold
# =============================================================================


class TestRetrievalCacheKeyCoversResultShape:
    @pytest.mark.asyncio
    async def test_top_k_is_part_of_the_cache_key(self, rag_service):
        """A ``top_k=3`` lookup must not serve its 3 items to a ``top_k=10`` one.

        chat_service asks for ``top_k=3`` on a bare DTC code; within the 300 s TTL
        the diagnosis pipeline asks for ``top_k=10`` with the same reduced query.
        """
        rag_service._qdrant = _unified_qdrant_mock({"dtc": _dtc_hits([0.9] * 10)})

        embed_patch, _, _ = _offline_rag_patches()
        with embed_patch:
            few = await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc", top_k=3)
            many = await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc", top_k=10)

        assert len(few) == 3
        assert len(many) == 10
        assert rag_service._qdrant.search.await_count == 2

    @pytest.mark.asyncio
    async def test_score_threshold_is_part_of_the_cache_key(self, rag_service):
        """A stricter threshold must re-query instead of reusing loose results."""
        rag_service._qdrant = _unified_qdrant_mock({"dtc": _dtc_hits([0.9, 0.8])})

        embed_patch, _, _ = _offline_rag_patches()
        with embed_patch:
            await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc", score_threshold=0.5)
            await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc", score_threshold=0.85)

        thresholds = [
            call.kwargs.get("score_threshold") for call in rag_service._qdrant.search.call_args_list
        ]
        assert thresholds == [0.5, 0.85]

    @pytest.mark.asyncio
    async def test_identical_request_is_still_cached(self, rag_service):
        """The key got wider, not useless: a repeated request stays a cache hit."""
        rag_service._qdrant = _unified_qdrant_mock({"dtc": _dtc_hits([0.9, 0.8])})

        embed_patch, _, _ = _offline_rag_patches()
        with embed_patch:
            await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc", top_k=10)
            await rag_service.retrieve_from_qdrant(query="P0301", type_="dtc", top_k=10)

        assert rag_service._qdrant.search.await_count == 1


# =============================================================================
# B5 - the similar-cases leg must read the payload shape that actually exists
# =============================================================================


class TestComplaintPayloadNarrative:
    @pytest.mark.asyncio
    async def test_narrative_is_read_from_the_indexed_payload(self, rag_service):
        """The ``description`` shape (scripts/index_qdrant_hubert.py) is used."""
        rag_service._qdrant = _unified_qdrant_mock(
            {"dtc": _dtc_hits([0.85]), "complaint": COMPLAINT_HITS_WITH_NARRATIVE}
        )

        context = await _assemble(rag_service)

        assert "Engine misfires and shakes at idle" in context.symptom_context

    @pytest.mark.asyncio
    async def test_narrative_less_complaints_do_not_emit_blank_prompt_lines(self, rag_service):
        """The ``sync_qdrant_sprint9`` shape stores no narrative at all.

        Those hits used to render as "1.  (hasonlosag: 78%)" - five blank lines
        in the Hungarian prompt - and suppressed the honest empty-state branch.
        """
        rag_service._qdrant = _unified_qdrant_mock(
            {"dtc": _dtc_hits([0.85]), "complaint": COMPLAINT_HITS_WITHOUT_NARRATIVE}
        )

        context = await _assemble(rag_service)

        assert context.symptom_items, "the hit itself is still retrieved"
        assert context.symptom_context == "Nincs hasonlo eset az adatbazisban."
        assert "hasonlosag" not in context.symptom_context

    def test_contentless_symptom_hits_do_not_inflate_confidence(self):
        """Factor 3 must not score evidence the diagnosis never saw."""
        from app.services.rag_service import (
            RAGContext,
            RAGService,
            RetrievalSource,
            RetrievedItem,
        )

        service = RAGService()
        dtc_item = RetrievedItem(
            content={"code": "P0301"}, source=RetrievalSource.QDRANT_DTC, score=0.9
        )

        with_contentless_hit = RAGContext(
            dtc_items=[dtc_item],
            symptom_items=[
                RetrievedItem(
                    content=dict(COMPLAINT_HITS_WITHOUT_NARRATIVE[0]["payload"]),
                    source=RetrievalSource.QDRANT_COMPLAINT,
                    score=0.95,
                )
            ],
        )
        without_any_hit = RAGContext(dtc_items=[dtc_item])

        _level_a, score_a = service.calculate_confidence(with_contentless_hit, ["P0301"])
        _level_b, score_b = service.calculate_confidence(without_any_hit, ["P0301"])

        assert score_a == pytest.approx(score_b)
        assert score_a == pytest.approx(0.54, abs=1e-3)

    def test_narrative_bearing_hits_still_count_towards_confidence(self):
        """The fix must not silence real evidence."""
        from app.services.rag_service import (
            RAGContext,
            RAGService,
            RetrievalSource,
            RetrievedItem,
        )

        service = RAGService()
        context = RAGContext(
            dtc_items=[
                RetrievedItem(
                    content={"code": "P0301"}, source=RetrievalSource.QDRANT_DTC, score=0.9
                )
            ],
            symptom_items=[
                RetrievedItem(
                    content=dict(COMPLAINT_HITS_WITH_NARRATIVE[0]["payload"]),
                    source=RetrievalSource.QDRANT_COMPLAINT,
                    score=0.95,
                )
            ],
        )

        _level, score = service.calculate_confidence(context, ["P0301"])

        # 0.9 * 0.3 + 0.95 * 0.2 over 0.7 counted weight.
        assert score == pytest.approx(0.6571, abs=1e-3)


# =============================================================================
# B6 - a degenerate vector is never produced, returned as real, or cached
# =============================================================================


class _RecordingCache:
    def __init__(self) -> None:
        self.written: List[str] = []

    async def get_embedding(self, key):
        return None

    async def get_embeddings_batch(self, keys):
        return [None] * len(keys)

    async def set_embedding(self, key, embedding):
        self.written.append(key)
        assert any(embedding), f"zero vector written to the cache under {key!r}"


def _onnx_service(mod):
    """A service instance whose backend is resolved without touching disk."""
    with patch.object(mod.HungarianEmbeddingService, "_select_backend_name", lambda self: "onnx"):
        return mod.HungarianEmbeddingService()


class TestZeroVectorsAreNeverCachedOrReturnedAsReal:
    @pytest.mark.asyncio
    async def test_blank_entry_is_not_cached_and_is_reported_as_none(
        self, reset_embedding_singleton
    ):
        """``embed_batch_async`` used to write ``[0.0] * 768`` into the v2 namespace."""
        mod = reset_embedding_singleton
        service = _onnx_service(mod)
        cache = _RecordingCache()

        with (
            patch.object(
                service,
                "embed_batch",
                return_value=[[0.0] * 768, [0.1] * 768],
            ),
            patch("app.db.redis_cache.get_cache_service", new=AsyncMock(return_value=cache)),
        ):
            results = await service.embed_batch_async(["   ", "rangat a motor"])

        assert results[0] is None
        assert results[1] == [0.1] * 768
        assert len(cache.written) == 1
        assert cache.written[0].endswith("rangat a motor")

    @pytest.mark.asyncio
    async def test_embed_text_async_does_not_cache_a_zero_vector(self, reset_embedding_singleton):
        mod = reset_embedding_singleton
        service = _onnx_service(mod)
        cache = _RecordingCache()

        with (
            patch.object(service, "embed_text", return_value=[0.0] * 768),
            patch("app.db.redis_cache.get_cache_service", new=AsyncMock(return_value=cache)),
        ):
            vector = await service.embed_text_async("   ")

        assert vector == [0.0] * 768  # documented empty-input result
        assert cache.written == []

    def test_missing_vector_for_real_text_raises_instead_of_returning_zeros(
        self, reset_embedding_singleton
    ):
        """A short backend response used to leave the pre-seeded zero vector."""
        mod = reset_embedding_singleton
        service = _onnx_service(mod)

        with (
            patch.object(service, "_embed_batch_internal", return_value=None),
            pytest.raises(EmbeddingUnavailableError),
        ):
            service.embed_batch(["valodi magyar szoveg"], use_cache=False)

    def test_empty_input_still_returns_the_documented_zero_vector(self, reset_embedding_singleton):
        """Indexers rely on it: an empty record must not get a random point."""
        from app.core.config import settings

        mod = reset_embedding_singleton
        service = _onnx_service(mod)

        with patch.object(service, "_embed_batch_internal", return_value=None):
            vectors = service.embed_batch(["", "   "], use_cache=False)

        assert vectors == [[0.0] * settings.EMBEDDING_DIMENSION] * 2
