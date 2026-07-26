"""
Embedding backend contract, safety guards and frozen-space regression tests.

Context (docs/EMBEDDING_ARCHITECTURE_DECISION.md): production ran without torch,
so ``embed_text`` silently returned ``[0.0] * 768``. A zero vector matches
nothing under cosine distance, so Hungarian semantic search returned ``[]`` for
months without a single error being logged. The fix has three parts and this
module tests all three:

1. R1 - a real inference backend (ONNX Runtime fp32, torch-free) that stays in
   the SAME embedding space as the ~54k already-indexed Qdrant vectors.
2. R3 - the safety guards: never a zero vector, a norm guard at the Qdrant
   gate, and a health probe.
3. The frozen fixtures that make a future silent shift of the embedding space
   fail the suite instead of quietly degrading search quality.

Everything here runs offline: no network, no Qdrant, no Redis, no model
download. The tests that genuinely need a loaded 440 MB model are explicitly
skipped (and say so) rather than faked.
"""

import ast
import importlib
import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import EmbeddingUnavailableError

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
REFERENCE_VECTORS_PATH = FIXTURE_DIR / "hubert_reference_vectors.json"
POOLING_REFERENCE_PATH = FIXTURE_DIR / "pooling_reference.json"


@pytest.fixture(autouse=True)
def _reset_embedding_singleton():
    """Reset the singleton so each test resolves its own backend."""
    import app.services.embedding_service as mod

    mod.HungarianEmbeddingService._instance = None
    mod.HungarianEmbeddingService._initialized = False
    mod._embedding_service = None
    yield
    mod.HungarianEmbeddingService._instance = None
    mod.HungarianEmbeddingService._initialized = False
    mod._embedding_service = None


# ---------------------------------------------------------------------------
# Fake ONNX Runtime + tokenizer (deterministic, dependency-free)
# ---------------------------------------------------------------------------


class _FakeSessionOptions:
    def __init__(self) -> None:
        self.intra_op_num_threads: Optional[int] = None
        self.inter_op_num_threads: Optional[int] = None
        self.graph_optimization_level: Any = None


class _FakeGraphOptimizationLevel:
    ORT_ENABLE_ALL = "ORT_ENABLE_ALL"


class _FakeNamedInput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeInferenceSession:
    """Deterministic stand-in for ort.InferenceSession."""

    last_instance: Optional["_FakeInferenceSession"] = None

    def __init__(self, model_path, session_options=None, providers=None) -> None:
        self.model_path = model_path
        self.session_options = session_options
        self.providers = providers
        self.last_feeds: Optional[Dict[str, np.ndarray]] = None
        _FakeInferenceSession.last_instance = self

    def get_inputs(self):
        return [
            _FakeNamedInput("input_ids"),
            _FakeNamedInput("attention_mask"),
            _FakeNamedInput("token_type_ids"),
        ]

    def run(self, output_names, feeds):
        self.last_feeds = feeds
        ids = feeds["input_ids"].astype(np.float32)
        batch, seq = ids.shape
        # A deterministic, id-dependent "hidden state" so the same text always
        # produces the same vector and different texts produce different ones.
        d = np.arange(1, 769, dtype=np.float32)[None, None, :]
        hidden = np.sin(ids[:, :, None] * 0.017 + d * 0.001).astype(np.float32)
        assert hidden.shape == (batch, seq, 768)
        return [hidden]


class _FakeEncoding:
    def __init__(self, ids, attention_mask, type_ids) -> None:
        self.ids = ids
        self.attention_mask = attention_mask
        self.type_ids = type_ids


class _FakeBertWordPieceTokenizer:
    """Stand-in for tokenizers.BertWordPieceTokenizer, records its config."""

    last_instance: Optional["_FakeBertWordPieceTokenizer"] = None

    def __init__(self, vocab_path, lowercase=True) -> None:
        self.vocab_path = vocab_path
        self.lowercase = lowercase
        self.truncation_max_length: Optional[int] = None
        self.padding_enabled = False
        _FakeBertWordPieceTokenizer.last_instance = self

    def enable_truncation(self, max_length: int) -> None:
        self.truncation_max_length = max_length

    def enable_padding(self) -> None:
        self.padding_enabled = True

    def encode(self, text: str) -> _FakeEncoding:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: List[str]) -> List[_FakeEncoding]:
        raw = [
            [101, *[ord(c) % 30000 for c in t[: self.truncation_max_length or 512]], 102]
            for t in texts
        ]
        width = max(len(r) for r in raw)
        out = []
        for r in raw:
            pad = width - len(r)
            out.append(
                _FakeEncoding(
                    ids=r + [0] * pad,
                    attention_mask=[1] * len(r) + [0] * pad,
                    type_ids=[0] * width,
                )
            )
        return out


@pytest.fixture
def onnx_service(tmp_path):
    """A HungarianEmbeddingService wired to the fake ONNX stack."""
    import app.services.embedding_service as mod

    model_file = tmp_path / "hubert_fp32.onnx"
    vocab_file = tmp_path / "vocab.txt"
    model_file.write_bytes(b"not-a-real-onnx-graph")
    vocab_file.write_text("[PAD]\n[CLS]\n[SEP]\n", encoding="utf-8")

    fake_ort = MagicMock()
    fake_ort.SessionOptions = _FakeSessionOptions
    fake_ort.GraphOptimizationLevel = _FakeGraphOptimizationLevel
    fake_ort.InferenceSession = _FakeInferenceSession

    with (
        patch.object(mod, "ONNX_RUNTIME_AVAILABLE", True),
        patch.object(mod, "ort", fake_ort),
        patch.object(mod, "BertWordPieceTokenizer", _FakeBertWordPieceTokenizer),
        patch.object(settings, "EMBEDDING_BACKEND", "onnx"),
        patch.object(settings, "HUBERT_ONNX_PATH", str(model_file)),
        patch.object(settings, "HUBERT_VOCAB_PATH", str(vocab_file)),
        patch.object(settings, "EMBEDDING_ORT_THREADS", 2),
    ):
        svc = mod.HungarianEmbeddingService()
        svc.disable_cache()
        yield svc


# ---------------------------------------------------------------------------
# 1. Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def test_onnx_selected_when_artifacts_present(self, onnx_service):
        assert onnx_service.backend_name == "onnx"

    def test_disabled_setting_yields_no_backend(self):
        import app.services.embedding_service as mod

        with patch.object(settings, "EMBEDDING_BACKEND", "disabled"):
            svc = mod.HungarianEmbeddingService()
            assert svc.backend_name is None

    def test_onnx_requested_but_missing_yields_no_backend(self, tmp_path):
        import app.services.embedding_service as mod

        with (
            patch.object(settings, "EMBEDDING_BACKEND", "onnx"),
            patch.object(settings, "HUBERT_ONNX_PATH", str(tmp_path / "nope.onnx")),
            patch.object(settings, "HUBERT_VOCAB_PATH", str(tmp_path / "nope.txt")),
        ):
            svc = mod.HungarianEmbeddingService()
            assert svc.backend_name is None

    def test_torch_requested_but_missing_yields_no_backend(self):
        import app.services.embedding_service as mod

        with (
            patch.object(settings, "EMBEDDING_BACKEND", "torch"),
            patch.object(mod, "TORCH_AVAILABLE", False),
        ):
            svc = mod.HungarianEmbeddingService()
            assert svc.backend_name is None

    def test_auto_prefers_onnx_over_torch(self, tmp_path):
        """Production must not accidentally load torch when ONNX is available."""
        import app.services.embedding_service as mod

        model_file = tmp_path / "m.onnx"
        vocab_file = tmp_path / "v.txt"
        model_file.write_bytes(b"x")
        vocab_file.write_text("x", encoding="utf-8")

        with (
            patch.object(settings, "EMBEDDING_BACKEND", "auto"),
            patch.object(settings, "HUBERT_ONNX_PATH", str(model_file)),
            patch.object(settings, "HUBERT_VOCAB_PATH", str(vocab_file)),
            patch.object(mod, "ONNX_RUNTIME_AVAILABLE", True),
            patch.object(mod, "TORCH_AVAILABLE", True),
        ):
            svc = mod.HungarianEmbeddingService()
            assert svc.backend_name == "onnx"


# ---------------------------------------------------------------------------
# 2. THE core regression: never a zero vector
# ---------------------------------------------------------------------------


class TestNeverAZeroVector:
    def test_embed_text_raises_when_no_backend(self):
        import app.services.embedding_service as mod

        with patch.object(settings, "EMBEDDING_BACKEND", "disabled"):
            svc = mod.HungarianEmbeddingService()
            svc.disable_cache()
            with pytest.raises(EmbeddingUnavailableError):
                svc.embed_text("rangat a motor")

    def test_embed_batch_raises_when_no_backend(self):
        import app.services.embedding_service as mod

        with patch.object(settings, "EMBEDDING_BACKEND", "disabled"):
            svc = mod.HungarianEmbeddingService()
            svc.disable_cache()
            with pytest.raises(EmbeddingUnavailableError):
                svc.embed_batch(["a", "b"])

    @pytest.mark.asyncio
    async def test_embed_text_async_raises_when_no_backend(self):
        import app.services.embedding_service as mod

        with patch.object(settings, "EMBEDDING_BACKEND", "disabled"):
            svc = mod.HungarianEmbeddingService()
            svc.disable_cache()
            with pytest.raises(EmbeddingUnavailableError):
                await svc.embed_text_async("rangat a motor", use_cache=False)

    def test_error_is_not_swallowed_into_a_vector(self):
        """The exception must carry actionable diagnostics, not be generic."""
        import app.services.embedding_service as mod

        with patch.object(settings, "EMBEDDING_BACKEND", "disabled"):
            svc = mod.HungarianEmbeddingService()
            svc.disable_cache()
            with pytest.raises(EmbeddingUnavailableError) as exc:
                svc.embed_text("x")

        assert "zero vector" in str(exc.value).lower()
        assert exc.value.details["embedding_backend_setting"] == "disabled"
        assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# 3. The embedding contract
# ---------------------------------------------------------------------------


class TestEmbeddingContract:
    def test_returns_768_unit_norm_vector(self, onnx_service):
        vector = onnx_service.embed_text("rangat a motor gyorsitaskor")
        assert len(vector) == 768
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5

    def test_deterministic_for_same_input(self, onnx_service):
        a = onnx_service.embed_text("Kek fust jon a kipufogobol")
        b = onnx_service.embed_text("Kek fust jon a kipufogobol")
        assert a == b

    def test_different_texts_give_different_vectors(self, onnx_service):
        a = onnx_service.embed_text("rangat a motor")
        b = onnx_service.embed_text("nem fekez rendesen")
        assert a != b

    def test_batch_matches_single(self, onnx_service):
        """A text must embed IDENTICALLY alone and inside a batch.

        This is the property the ONNX path can lose silently.
        ``_OnnxEmbeddingBackend.__init__`` calls ``enable_padding()``, so batching
        a short text with a long one pads the short row with [PAD] tokens up to
        the long row's length, while ``embed_text`` runs that same short text
        with no padding at all. The ONLY thing that keeps the two results equal
        is mask-weighted mean pooling: ``_mean_pool_l2_numpy`` zeroes the padded
        positions in the numerator AND excludes them from the token count.

        Drop the mask - a plain ``.mean(axis=1)``, or letting ``attention_mask``
        fall out of the tokenizer output (which is exactly why ``tokenize()``
        returns it independently of ``graph_inputs()``) - and a vector starts
        depending on which texts it happened to be batched with. It would still
        be 768-dim, still unit-norm, still look completely healthy: a silent
        embedding-quality regression of precisely the class this suite exists to
        catch. The previous version of this test asserted only shape and norm and
        never compared batch to single, so it would have passed straight through
        that regression.
        """
        short = "fek"
        long_text = "rangat a motor gyorsitaskor es kek fust jon a kipufogobol " * 4
        texts = [short, long_text]

        batch = onnx_service.embed_batch(texts, use_cache=False)
        assert len(batch) == 2

        # Guard against a VACUOUS assertion: the property only has teeth if this
        # batch genuinely padded the short row. Two same-length texts would make
        # the comparison below trivially true and prove nothing.
        backend = onnx_service._load_onnx_backend()
        batched_mask = backend.tokenize(texts)["attention_mask"][0].tolist()
        alone_mask = backend.tokenize([short])["attention_mask"][0].tolist()
        assert 0 in batched_mask, (
            "the short text was NOT padded in this batch - this test would prove nothing"
        )
        assert 0 not in alone_mask, "the short text must be unpadded when embedded alone"

        for text, batched in zip(texts, batch):
            single = onnx_service.embed_text(text)
            assert len(batched) == 768
            assert abs(float(np.linalg.norm(batched)) - 1.0) < 1e-5
            np.testing.assert_allclose(
                np.asarray(batched, dtype=np.float64),
                np.asarray(single, dtype=np.float64),
                rtol=0,
                atol=1e-6,
                err_msg=(
                    f"{text!r} embeds differently in a batch than alone - "
                    "mask-weighted pooling is no longer neutralising [PAD] tokens."
                ),
            )

    def test_batch_matches_single_would_fail_without_mask_weighted_pooling(self, onnx_service):
        """Proves the test above actually BITES.

        A regression test that cannot fail is not a test - that lesson cost this
        project a ``skipif`` that skipped in every environment. Here the pooling
        is swapped for a mask-ignoring mean and the batch/single equality MUST
        break; if it does not, the assertion above is vacuous and the real guard
        is gone.
        """
        import app.services.embedding_service as mod

        def _mask_ignoring_pool(last_hidden, attention_mask):
            pooled = last_hidden.astype(np.float32).mean(axis=1)
            norms = np.clip(np.linalg.norm(pooled, ord=2, axis=1, keepdims=True), 1e-12, None)
            return np.asarray(pooled / norms, dtype=np.float32)

        short = "fek"
        long_text = "rangat a motor gyorsitaskor es kek fust jon a kipufogobol " * 4

        with patch.object(mod, "_mean_pool_l2_numpy", _mask_ignoring_pool):
            batched_short = onnx_service.embed_batch([short, long_text], use_cache=False)[0]
            single_short = onnx_service.embed_text(short)

        assert not np.allclose(
            np.asarray(batched_short, dtype=np.float64),
            np.asarray(single_short, dtype=np.float64),
            atol=1e-6,
        ), (
            "mask-ignoring pooling produced the SAME vector batched and alone - "
            "test_batch_matches_single cannot detect a pooling regression."
        )

    def test_empty_text_still_returns_zero_vector(self, onnx_service):
        """Empty input is defensible on the INDEXING side; the Qdrant norm
        guard is what stops it reaching a similarity search."""
        assert onnx_service.embed_text("") == [0.0] * 768

    def test_tokenizer_is_configured_case_sensitively(self, onnx_service):
        """do_lower_case is false for hubert-base-cc: lowercase=True would
        silently degrade every embedding."""
        onnx_service.embed_text("Motor")
        tokenizer = _FakeBertWordPieceTokenizer.last_instance
        assert tokenizer is not None
        assert tokenizer.lowercase is False
        assert tokenizer.truncation_max_length == 512
        assert tokenizer.padding_enabled is True

    def test_ort_threads_are_capped(self, onnx_service):
        onnx_service.embed_text("Motor")
        session = _FakeInferenceSession.last_instance
        assert session is not None
        assert session.session_options.intra_op_num_threads == 2
        assert session.session_options.inter_op_num_threads == 1
        assert session.providers == ["CPUExecutionProvider"]

    def test_preprocess_hungarian_is_unchanged(self, onnx_service):
        """Without spaCy this is the identity function - and it MUST stay one.
        Turning it into real lemmatization would put queries in a different
        text space than the indexed vectors (silent quality loss + reindex)."""
        import app.services.embedding_service as mod

        with patch.object(mod, "SPACY_AVAILABLE", False):
            assert onnx_service.preprocess_hungarian("  Motor hiba  ") == "Motor hiba"
            assert onnx_service.preprocess_hungarian("") == ""


# ---------------------------------------------------------------------------
# 4. FROZEN embedding space
# ---------------------------------------------------------------------------


def _load_reference_fixture() -> Dict[str, Any]:
    with REFERENCE_VECTORS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


class TestFrozenReferenceVectors:
    """The highest-value tests here: they pin the embedding SPACE itself.

    The vectors were produced by the torch reference path (transformers 4.37.2 +
    torch 2.2.0+cpu at the pinned revision) - i.e. the very path that produced
    the ~54k vectors sitting in Qdrant today.
    """

    def test_fixture_is_wellformed_768_dim_unit_norm(self):
        data = _load_reference_fixture()
        assert data["dimension"] == 768
        assert len(data["vectors"]) >= 5
        for entry in data["vectors"]:
            vector = np.asarray(entry["vector"], dtype=np.float64)
            assert vector.shape == (768,), entry["text"]
            assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5, entry["text"]

    def test_fixture_matches_the_configured_model_and_revision(self):
        """If someone bumps HUBERT_MODEL/HUBERT_REVISION without regenerating
        this fixture AND reindexing Qdrant, the embedding space silently shifts
        under 54k existing vectors. That must not be a quiet change."""
        data = _load_reference_fixture()
        assert data["model"] == settings.HUBERT_MODEL
        assert data["revision"] == settings.HUBERT_REVISION
        assert data["revision"] != "main", (
            "HUBERT_REVISION must be pinned to a commit SHA - 'main' lets "
            "HuggingFace move the weights under the indexed vectors."
        )

    def test_fixture_covers_uppercase_and_accented_hungarian(self):
        """lowercase=False is mandatory for this model. A fixture without
        uppercase/accented Hungarian could not catch a lowercase regression."""
        data = _load_reference_fixture()
        texts = [entry["text"] for entry in data["vectors"]]
        assert any(t != t.lower() for t in texts), "no uppercase text in fixture"
        assert any(any(ch in t for ch in "áéíóöőúüűÁÉÍÓÖŐÚÜŰ") for t in texts)
        assert any(entry["n_tokens"] > 30 for entry in data["vectors"]), "no long text"

    @pytest.mark.skipif(
        not (
            Path(settings.HUBERT_ONNX_PATH).is_file() and Path(settings.HUBERT_VOCAB_PATH).is_file()
        ),
        reason=(
            "Needs the real exported huBERT graph (present in the production "
            "image, absent in the lightweight CI env)."
        ),
    )
    def test_frozen_reference_vectors_match_backend(self):
        """A real backend must reproduce the frozen vectors to cosine ~ 1.0.

        This is what turns "we swapped the inference backend" from a leap of
        faith into a checked property.
        """
        import app.services.embedding_service as mod

        data = _load_reference_fixture()
        svc = mod.HungarianEmbeddingService()
        svc.disable_cache()

        worst = 1.0
        for entry in data["vectors"]:
            produced = np.asarray(svc.embed_text(entry["text"]), dtype=np.float64)
            expected = np.asarray(entry["vector"], dtype=np.float64)
            cosine = float(produced @ expected)
            worst = min(worst, cosine)
            assert cosine > 0.9999, (
                f"Embedding space drifted for {entry['text']!r}: cosine={cosine}. "
                "Either the backend/model changed, or the pooling/normalization "
                "was altered. Qdrant would need a full reindex."
            )
        assert worst > 0.9999


class TestFrozenPooling:
    """Pins the pooling+L2 math that lives OUTSIDE the model graph.

    This is precisely why the ONNX backend can be dropped in without touching a
    single indexed vector: swapping the transformer forward pass changes nothing
    as long as this stays identical. Runs with zero ML dependencies.
    """

    def test_numpy_pooling_matches_frozen_reference(self):
        import app.services.embedding_service as mod

        with POOLING_REFERENCE_PATH.open(encoding="utf-8") as f:
            data = json.load(f)

        assert data["cases"], "pooling reference fixture is empty"
        for case in data["cases"]:
            hidden = np.asarray(case["last_hidden_state"], dtype=np.float32)
            mask = np.asarray(case["attention_mask"], dtype=np.int64)
            expected = np.asarray(case["expected"], dtype=np.float64)
            produced = np.asarray(mod._mean_pool_l2_numpy(hidden, mask), dtype=np.float64)
            assert produced.shape == expected.shape
            assert np.allclose(produced, expected, atol=1e-6), case["attention_mask"]

    def test_fully_masked_row_does_not_divide_by_zero(self):
        """The clamp(min=1e-9) / clamp(min=1e-12) guards are load-bearing."""
        import app.services.embedding_service as mod

        hidden = np.zeros((1, 4, 8), dtype=np.float32)
        mask = np.zeros((1, 4), dtype=np.int64)
        result = mod._mean_pool_l2_numpy(hidden, mask)
        assert np.isfinite(result).all()

    def test_output_is_l2_normalized(self):
        import app.services.embedding_service as mod

        rng = np.random.default_rng(7)
        hidden = rng.standard_normal((2, 6, 32)).astype(np.float32)
        mask = np.ones((2, 6), dtype=np.int64)
        result = mod._mean_pool_l2_numpy(hidden, mask)
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. THE transformers guard
# ---------------------------------------------------------------------------


class _ImportBlocker:
    """meta_path finder that makes importing the blocked roots explode."""

    def __init__(self, blocked_roots):
        self.blocked_roots = set(blocked_roots)

    def find_spec(self, fullname, path=None, target=None):
        # Implicit None == "not my import, ask the next finder".
        if fullname.split(".")[0] in self.blocked_roots:
            raise ImportError(f"Import of {fullname!r} is FORBIDDEN on the ONNX runtime embed path")


class TestNoTransformersOnTheOnnxPath:
    """Importing ``transformers`` drags torch in (+367 MB RSS measured), which
    makes the ONNX path use MORE memory than plain torch. That would silently
    undo the whole architecture, so it is enforced both dynamically and
    statically.
    """

    def test_onnx_embed_works_with_torch_and_transformers_import_blocked(self, onnx_service):
        blocker = _ImportBlocker({"torch", "transformers"})
        saved = {
            name: mod
            for name, mod in list(sys.modules.items())
            if name.split(".")[0] in {"torch", "transformers"}
        }
        for name in saved:
            del sys.modules[name]

        sys.meta_path.insert(0, blocker)
        try:
            vector = onnx_service.embed_text("ÁRAMSZÜNET! Ékezetes, NAGYBETŰS teszt.")
            batch = onnx_service.embed_batch(["fék", "motor"], use_cache=False)

            assert len(vector) == 768
            assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
            assert len(batch) == 2

            # Nothing on the embed path may have (re-)imported them.
            assert "transformers" not in sys.modules
            assert "torch" not in sys.modules
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved)

    def test_onnx_backend_source_never_references_torch_or_transformers(self):
        """Static check, immune to docstrings and to whether the packages
        happen to be installed in the test environment."""
        import app.services.embedding_service as mod

        source = textwrap.dedent(inspect.getsource(mod._OnnxEmbeddingBackend))
        tree = ast.parse(source)

        forbidden = {"torch", "transformers", "AutoModel", "AutoTokenizer"}
        seen = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                seen.add(node.id)
            elif isinstance(node, ast.Attribute):
                seen.add(node.attr)
            elif isinstance(node, ast.Import):
                seen.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                seen.add(node.module.split(".")[0])
                seen.update(alias.name for alias in node.names)

        offending = forbidden & seen
        assert not offending, (
            f"_OnnxEmbeddingBackend references {sorted(offending)} - importing "
            "transformers pulls torch back into the production image."
        )

    def test_prod_requirements_do_not_ship_torch_or_transformers(self):
        requirements = (Path(__file__).resolve().parents[2] / "requirements.prod.txt").read_text(
            encoding="utf-8"
        )
        installed = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for package in ("torch", "transformers", "sentence-transformers", "spacy", "huspacy"):
            assert not any(line.lower().startswith(package) for line in installed), (
                f"{package} must not be a production dependency"
            )
        assert any(line.startswith("onnxruntime") for line in installed)
        assert any(line.startswith("tokenizers") for line in installed)


# ---------------------------------------------------------------------------
# 6. Qdrant norm guard
# ---------------------------------------------------------------------------


class TestQdrantQueryVectorGuard:
    def test_zero_vector_is_rejected(self):
        from app.db.qdrant_client import _validate_query_vector

        with pytest.raises(ValueError, match="Degenerate query vector"):
            _validate_query_vector([0.0] * 768, "autocognitix")

    def test_near_zero_vector_is_rejected(self):
        from app.db.qdrant_client import _validate_query_vector

        with pytest.raises(ValueError, match="Degenerate query vector"):
            _validate_query_vector([1e-12] * 768, "autocognitix")

    def test_empty_vector_is_rejected(self):
        from app.db.qdrant_client import _validate_query_vector

        with pytest.raises(ValueError, match="Empty query vector"):
            _validate_query_vector([], "autocognitix")

    def test_nan_vector_is_rejected(self):
        from app.db.qdrant_client import _validate_query_vector

        with pytest.raises(ValueError):
            _validate_query_vector([float("nan")] * 768, "autocognitix")

    def test_unit_vector_passes(self):
        from app.db.qdrant_client import _validate_query_vector

        vector = [0.0] * 768
        vector[0] = 1.0
        _validate_query_vector(vector, "autocognitix")  # must not raise

    @pytest.mark.asyncio
    async def test_search_rejects_zero_vector_before_touching_qdrant(self):
        """The guard must fire at the gate - no request may reach Qdrant."""
        from app.db.qdrant_client import QdrantService

        service = QdrantService.__new__(QdrantService)
        service.client = MagicMock()
        service.client.search = AsyncMock()

        with pytest.raises(ValueError):
            await service.search(query_vector=[0.0] * 768)
        service.client.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_unified_also_rejects_zero_vector(self):
        from app.db.qdrant_client import QdrantService

        service = QdrantService.__new__(QdrantService)
        service.client = MagicMock()
        service.client.search = AsyncMock()

        with pytest.raises(ValueError):
            await service.search_unified(query_vector=[0.0] * 768, type_="dtc")
        service.client.search.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Health self-test probe
# ---------------------------------------------------------------------------


class TestEmbeddingSelfTest:
    def test_reports_ok_for_a_working_backend(self, onnx_service):
        probe = onnx_service.self_test()
        assert probe["status"] == "ok"
        assert probe["backend"] == "onnx"
        assert abs(probe["self_test_norm"] - 1.0) < 1e-4
        assert probe["self_test_dimension"] == 768
        assert probe["revision"] == settings.HUBERT_REVISION

    def test_reports_unavailable_without_a_backend(self):
        import app.services.embedding_service as mod

        with patch.object(settings, "EMBEDDING_BACKEND", "disabled"):
            svc = mod.HungarianEmbeddingService()
            probe = svc.self_test()

        assert probe["status"] == "unavailable"
        assert probe["backend"] is None
        assert probe["error"]

    def test_reports_degraded_for_a_zero_vector(self, onnx_service):
        """If any future path ever DOES emit a zero vector, the health endpoint
        must call it out rather than report a healthy service."""
        with patch.object(onnx_service, "embed_text", return_value=[0.0] * 768):
            probe = onnx_service.self_test()

        assert probe["status"] == "degraded"
        assert probe["self_test_norm"] == 0.0

    def test_reports_degraded_for_wrong_dimension(self, onnx_service):
        with patch.object(onnx_service, "embed_text", return_value=[1.0, 0.0]):
            probe = onnx_service.self_test()
        assert probe["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_maps_ok_to_healthy(self):
        import app.api.v1.endpoints.health as health_mod

        probe = {"status": "ok", "backend": "onnx", "self_test_norm": 1.0}
        with patch("app.services.embedding_service.embedding_self_test", return_value=dict(probe)):
            result = await health_mod.check_embedding_health()

        assert result.status == "healthy"
        assert result.details["backend"] == "onnx"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_health_check_maps_unavailable_to_degraded(self):
        """An embedding outage is a degradation, not a full service outage:
        lexical + Neo4j diagnosis still works."""
        import app.api.v1.endpoints.health as health_mod

        probe = {"status": "unavailable", "backend": None, "error": "no backend"}
        with patch("app.services.embedding_service.embedding_self_test", return_value=dict(probe)):
            result = await health_mod.check_embedding_health()

        assert result.status == "degraded"
        assert result.error == "no backend"

    @pytest.mark.asyncio
    async def test_health_check_survives_an_exploding_probe(self):
        import app.api.v1.endpoints.health as health_mod

        with patch(
            "app.services.embedding_service.embedding_self_test",
            side_effect=RuntimeError("boom"),
        ):
            result = await health_mod.check_embedding_health()

        assert result.status == "unhealthy"
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_a_slow_probe_degrades_only_itself(self):
        """A cold model load must never spend the SHARED health-check budget.

        ``detailed_health_check`` gathers all five probes under one timeout, and
        that timeout's handler marks postgres, neo4j, qdrant AND redis
        unhealthy. If the embedding probe could trip it, one slow model load
        would report a total datastore outage. Its own smaller budget makes that
        impossible - and a timeout here is "degraded", because lexical and graph
        diagnosis keep working.
        """
        import asyncio

        import app.api.v1.endpoints.health as health_mod

        async def _never_finishes():
            await asyncio.sleep(3600)

        with (
            patch.object(health_mod, "EMBEDDING_HEALTH_TIMEOUT_SECONDS", 0.01),
            patch.object(health_mod, "check_embedding_health", _never_finishes),
        ):
            result = await health_mod._check_embedding_health_bounded()

        assert result.name == "Embedding"
        assert result.status == "degraded"
        assert "timed out" in (result.error or "")

    def test_the_embedding_budget_is_strictly_below_the_shared_one(self):
        """A budget that is not smaller is no isolation at all."""
        import re

        import app.api.v1.endpoints.health as health_mod

        source = inspect.getsource(health_mod.detailed_health_check)
        shared = [float(m) for m in re.findall(r"timeout=([0-9]+(?:\.[0-9]+)?)", source)]
        assert shared, "could not find the shared gather timeout to compare against"
        assert health_mod.EMBEDDING_HEALTH_TIMEOUT_SECONDS < min(shared)


# ---------------------------------------------------------------------------
# 8. Cache-key versioning
# ---------------------------------------------------------------------------


class TestEmbeddingCacheVersioning:
    def test_namespace_carries_version_and_backend(self, onnx_service):
        import app.services.embedding_service as mod

        key = onnx_service._cache_namespace("rangat a motor")
        assert key.startswith(f"{mod.EMBEDDING_CACHE_VERSION}|onnx|")
        assert key.endswith("rangat a motor")

    def test_namespace_differs_from_the_raw_text(self, onnx_service):
        """This is what makes pre-fix cached ZERO vectors unreachable on deploy
        without any manual `SCAN`+`DEL` of embed:* and without waiting out the
        1 hour TTL."""
        assert onnx_service._cache_namespace("x") != "x"

    def test_namespace_differs_per_backend(self, onnx_service):
        onnx_key = onnx_service._cache_namespace("x")
        onnx_service._backend_name = "torch"
        assert onnx_service._cache_namespace("x") != onnx_key

    def test_sync_embed_batch_has_no_dead_cache_scaffolding(self):
        """REGRESSION: ``embed_batch``'s ``use_cache`` guarded nothing at all.

        The block looked like a cache lookup - it seeded a ``cached_embeddings``
        list, branched on whether an event loop was running, and had an
        ``except ImportError`` fallback - but all four branches assigned the
        identical "embed every text" list, ``cached_embeddings`` was never
        written to, and the ``except ImportError`` was unreachable (the only
        calls in the ``try`` were ``asyncio.get_running_loop()`` and a list
        comprehension, and asyncio is imported at module scope). ``use_cache``
        therefore had zero effect while the docstring advertised "Redis cache
        integration for repeated texts".

        This pins the shape of the fix: no resurrected placeholder cache in the
        sync path.
        """
        import app.services.embedding_service as mod

        source = textwrap.dedent(inspect.getsource(mod.HungarianEmbeddingService.embed_batch))
        body = ast.parse(source).body[0]
        assert isinstance(body, ast.FunctionDef)

        assigned = {
            target.id
            for node in ast.walk(body)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "cached_embeddings" not in assigned, (
            "a cache placeholder that is never populated is back in embed_batch"
        )
        assert "texts_to_embed" not in assigned, (
            "the four identical no-op branches are back in embed_batch"
        )

    def test_use_cache_true_is_announced_not_silently_ignored(self, onnx_service, caplog):
        """An unkeepable promise must be audible.

        The sync path cannot reach the async Redis client, so ``use_cache=True``
        cannot be honoured. Accepting it and quietly doing nothing is the exact
        silent-no-op pattern that hid the zero-vector outage; the caller has to
        be told.
        """
        import logging

        import app.services.embedding_service as mod

        mod.HungarianEmbeddingService._warned_sync_cache = False
        onnx_service.enable_cache()

        with caplog.at_level(logging.WARNING, logger=mod.__name__):
            onnx_service.embed_batch(["rangat a motor"], use_cache=True)

        assert any("use_cache=True" in r.message for r in caplog.records), (
            "embed_batch(use_cache=True) silently ignored the request"
        )
        assert any("embed_batch_async" in r.message for r in caplog.records), (
            "the warning must name the API that DOES cache"
        )

    def test_the_sync_cache_warning_is_one_shot(self, onnx_service, caplog):
        """A per-call warning inside a 50K-text indexing loop is its own outage."""
        import logging

        import app.services.embedding_service as mod

        mod.HungarianEmbeddingService._warned_sync_cache = False
        onnx_service.enable_cache()

        with caplog.at_level(logging.WARNING, logger=mod.__name__):
            for _ in range(5):
                onnx_service.embed_batch(["rangat a motor"], use_cache=True)

        assert sum("use_cache=True" in r.message for r in caplog.records) == 1

    def test_use_cache_false_says_nothing(self, onnx_service, caplog):
        import logging

        import app.services.embedding_service as mod

        mod.HungarianEmbeddingService._warned_sync_cache = False
        onnx_service.enable_cache()

        with caplog.at_level(logging.WARNING, logger=mod.__name__):
            onnx_service.embed_batch(["rangat a motor"], use_cache=False)

        assert not any("use_cache=True" in r.message for r in caplog.records)

    def test_use_cache_does_not_change_the_vectors(self, onnx_service):
        """Whatever the flag says, the embeddings must be identical."""
        import app.services.embedding_service as mod

        mod.HungarianEmbeddingService._warned_sync_cache = False
        texts = ["rangat a motor", "kek fust"]
        assert onnx_service.embed_batch(texts, use_cache=False) == onnx_service.embed_batch(
            texts, use_cache=True
        )

    def test_sync_embed_batch_defaults_to_uncached(self):
        """The signature must not claim a cache the sync path cannot provide.

        The parameter itself is KEPT (not deleted) because ``scripts/
        index_qdrant_hubert.py`` passes ``use_cache=False`` by keyword; removing
        it would break that offline indexer with a TypeError.
        """
        import app.services.embedding_service as mod

        default = (
            inspect.signature(mod.HungarianEmbeddingService.embed_batch)
            .parameters["use_cache"]
            .default
        )
        assert default is False

    @pytest.mark.asyncio
    async def test_async_cache_lookup_uses_the_versioned_key(self, onnx_service):
        onnx_service.enable_cache()
        cache = AsyncMock()
        cache.get_embedding.return_value = None
        cache.set_embedding.return_value = True

        async def _get_cache():
            return cache

        with patch.dict(
            sys.modules,
            {"app.db.redis_cache": MagicMock(get_cache_service=_get_cache)},
        ):
            await onnx_service.embed_text_async("rangat a motor", use_cache=True)

        expected = onnx_service._cache_namespace("rangat a motor")
        cache.get_embedding.assert_awaited_once_with(expected)
        cache.set_embedding.assert_awaited_once()
        assert cache.set_embedding.await_args.args[0] == expected


# ---------------------------------------------------------------------------
# 9. Caller degradation (no 500s)
# ---------------------------------------------------------------------------


class TestCallersDegradeWithout500:
    @pytest.mark.asyncio
    async def test_rag_retrieval_returns_empty_instead_of_raising(self):
        """rag_service must absorb EmbeddingUnavailableError and fall back to
        its keyword/graph path - a broken embedding must not 500 /diagnosis."""
        from app.services.rag_service import RAGService

        service = RAGService.__new__(RAGService)
        service._cache = MagicMock()
        service._cache.get.return_value = None
        service._qdrant = MagicMock()
        service._qdrant.search_unified = AsyncMock()

        with patch(
            "app.services.rag_service.embed_text_async",
            side_effect=EmbeddingUnavailableError(),
        ):
            results = await service.retrieve_from_qdrant(query="rangat a motor", type_="dtc")

        assert results == []
        service._qdrant.search_unified.assert_not_called()

    @pytest.mark.asyncio
    async def test_qdrant_norm_guard_is_absorbed_by_rag(self):
        """Even if a degenerate vector somehow gets produced, the ValueError
        raised at the Qdrant gate must not escape to the endpoint."""
        from app.services.rag_service import RAGService

        service = RAGService.__new__(RAGService)
        service._cache = MagicMock()
        service._cache.get.return_value = None
        service._cache.set = MagicMock()
        service._qdrant = MagicMock()
        service._qdrant.search_unified = AsyncMock(
            side_effect=ValueError("Degenerate query vector")
        )

        with patch(
            "app.services.rag_service.embed_text_async",
            return_value=[0.0] * 768,
        ):
            results = await service.retrieve_from_qdrant(query="rangat a motor", type_="dtc")

        assert results == []


# ---------------------------------------------------------------------------
# 10. Public interface stability
# ---------------------------------------------------------------------------


class TestPublicInterfaceUnchanged:
    def test_signatures_are_unchanged(self):
        """Many call sites depend on these - the backend swap must be invisible."""
        mod = importlib.import_module("app.services.embedding_service")

        assert list(
            inspect.signature(mod.HungarianEmbeddingService.embed_text_async).parameters
        ) == ["self", "text", "preprocess", "use_cache"]
        assert list(
            inspect.signature(mod.HungarianEmbeddingService.embed_batch_async).parameters
        ) == ["self", "texts", "preprocess", "batch_size", "use_cache"]
        assert list(inspect.signature(mod.HungarianEmbeddingService.embed_text).parameters) == [
            "self",
            "text",
            "preprocess",
        ]
        assert list(inspect.signature(mod.HungarianEmbeddingService.embed_batch).parameters) == [
            "self",
            "texts",
            "preprocess",
            "batch_size",
            "use_cache",
        ]
        assert list(inspect.signature(mod.embed_text_async).parameters) == [
            "text",
            "preprocess",
            "use_cache",
        ]

    def test_max_sequence_length_is_512(self):
        """Shared by both backends AND by the offline indexer. Changing it
        changes the embedding space."""
        import app.services.embedding_service as mod

        assert mod.MAX_SEQUENCE_LENGTH == 512
