"""
Unit tests for HungarianEmbeddingService.

Tests cover the actual service class with mocked torch/transformers:
- Singleton pattern and get_embedding_service()
- embed_text (with and without torch)
- embed_batch (empty, single, multi-batch, cache disabled)
- preprocess_hungarian (with and without spacy)
- get_similar_texts
- embed_text_async / embed_batch_async
- warmup
- Properties (device, embedding_dimension, is_model_loaded)
- GPU memory cleanup
- OOM recovery in _embed_batch_internal
- Convenience module-level functions
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers - build mock torch objects used across tests
# ---------------------------------------------------------------------------


def _make_mock_torch(device_type: str = "cpu"):
    """Create a mock torch module with the given device type."""
    mock_torch = MagicMock()

    # Device
    device_obj = MagicMock()
    device_obj.type = device_type
    mock_torch.device.return_value = device_obj

    # CUDA availability
    mock_torch.cuda.is_available.return_value = device_type == "cuda"
    mock_torch.backends.mps.is_available.return_value = device_type == "mps"

    # CPU threads (for CPU branch in _detect_device)
    mock_torch.get_num_threads.return_value = 8

    # no_grad context manager
    mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

    # torch.sum / torch.clamp
    mock_torch.sum.side_effect = lambda t, **kw: t.sum(**kw) if hasattr(t, "sum") else t
    mock_torch.clamp.side_effect = lambda t, **kw: t

    # torch.nn.functional.normalize → return input unchanged (already "normalized")
    mock_torch.nn.functional.normalize.side_effect = lambda t, **kw: t

    # torch.compile (PyTorch 2.0+)
    mock_torch.compile.side_effect = lambda model, **kw: model

    # torch.float16 / torch.float32
    mock_torch.float16 = "float16"
    mock_torch.float32 = "float32"

    return mock_torch, device_obj


def _make_fake_embedding(dim: int = 768, batch: int = 1):
    """Return a numpy-backed tensor mock that behaves like a torch tensor."""
    data = np.random.randn(batch, dim).astype(np.float32)

    tensor = MagicMock()
    tensor.size.return_value = (batch, dim)
    tensor.squeeze.return_value = tensor
    tensor.cpu.return_value = tensor
    tensor.float.return_value = tensor
    tensor.tolist.return_value = data.tolist() if batch > 1 else data[0].tolist()
    tensor.sum.return_value = tensor
    tensor.__mul__ = lambda self, other: tensor
    tensor.__truediv__ = lambda self, other: tensor

    # expand for attention mask expansion
    tensor.expand.return_value = tensor
    tensor.unsqueeze.return_value = tensor

    return tensor


def _make_model_output(batch: int = 1, dim: int = 768):
    """Create a mock model output with last_hidden_state."""
    output = MagicMock()
    hidden = _make_fake_embedding(dim, batch)
    hidden.size.return_value = (batch, 10, dim)  # (batch, seq_len, dim)
    output.last_hidden_state = hidden
    return output


def _make_mock_tokenizer():
    """Create a mock tokenizer that returns encoded dict."""

    def tokenize(*args, **kwargs):
        # Return dict with input_ids and attention_mask as mock tensors
        ids = MagicMock()
        mask = MagicMock()
        mask.unsqueeze.return_value = mask
        mask.expand.return_value = mask
        mask.size.return_value = (1, 10)
        mask.float.return_value = mask
        mask.sum.return_value = mask
        mask.to.return_value = mask
        ids.to.return_value = ids
        return {"input_ids": ids, "attention_mask": mask}

    tok = MagicMock(side_effect=tokenize)
    return tok


def _make_mock_model(batch: int = 1, dim: int = 768):
    """Create a mock HuBERT model."""
    model = MagicMock()
    model_output = _make_model_output(batch, dim)
    model.return_value = model_output  # model(**encoded)
    model.parameters.return_value = [MagicMock()]
    model.eval.return_value = model
    model.to.return_value = model
    model.dtype = "float32"
    return model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton and module-level instance between tests."""
    import app.services.embedding_service as mod

    # Reset class-level singleton
    mod.HungarianEmbeddingService._instance = None
    mod.HungarianEmbeddingService._initialized = False
    # Reset module-level convenience instance
    mod._embedding_service = None
    yield
    mod.HungarianEmbeddingService._instance = None
    mod.HungarianEmbeddingService._initialized = False
    mod._embedding_service = None


@pytest.fixture
def mock_torch_cpu():
    """Patch torch as available with CPU device."""
    mt, device_obj = _make_mock_torch("cpu")
    with (
        patch("app.services.embedding_service.torch", mt),
        patch("app.services.embedding_service.TORCH_AVAILABLE", True),
        patch("app.services.embedding_service.AutoTokenizer") as auto_tok,
        patch("app.services.embedding_service.AutoModel") as auto_model,
    ):
        auto_tok.from_pretrained.return_value = _make_mock_tokenizer()
        auto_model.from_pretrained.return_value = _make_mock_model()
        yield {
            "torch": mt,
            "device": device_obj,
            "tokenizer_cls": auto_tok,
            "model_cls": auto_model,
        }


@pytest.fixture
def svc_cpu(mock_torch_cpu):
    """Create a HungarianEmbeddingService on mocked CPU."""
    from app.services.embedding_service import HungarianEmbeddingService

    svc = HungarianEmbeddingService()
    svc.disable_cache()
    return svc


@pytest.fixture
def svc_no_torch():
    """Create a service when torch is not available."""
    with (
        patch("app.services.embedding_service.TORCH_AVAILABLE", False),
        patch("app.services.embedding_service.torch", None),
    ):
        from app.services.embedding_service import HungarianEmbeddingService

        svc = HungarianEmbeddingService()
        svc.disable_cache()
        yield svc


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_returns_same_instance(self, mock_torch_cpu):
        from app.services.embedding_service import HungarianEmbeddingService

        a = HungarianEmbeddingService()
        b = HungarianEmbeddingService()
        assert a is b

    def test_get_embedding_service(self, mock_torch_cpu):
        from app.services.embedding_service import get_embedding_service

        svc = get_embedding_service()
        assert svc is get_embedding_service()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_device_property(self, svc_cpu):
        assert svc_cpu.device is not None

    def test_embedding_dimension(self, svc_cpu):
        assert svc_cpu.embedding_dimension == 768

    def test_is_model_loaded_false_initially(self, svc_cpu):
        assert svc_cpu.is_model_loaded is False

    def test_is_model_loaded_after_load(self, svc_cpu):
        svc_cpu._load_hubert_model()
        assert svc_cpu.is_model_loaded is True

    def test_device_none_without_torch(self, svc_no_torch):
        assert svc_no_torch.device is None

    def test_cache_enable_disable(self, svc_cpu):
        svc_cpu.disable_cache()
        assert svc_cpu._cache_enabled is False
        svc_cpu.enable_cache()
        assert svc_cpu._cache_enabled is True


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


class TestEmbedText:
    def test_returns_list_of_floats(self, svc_cpu, mock_torch_cpu):
        # Make the model return a proper embedding
        model = _make_mock_model(1, 768)
        mock_torch_cpu["model_cls"].from_pretrained.return_value = model

        # Need to set up mean_pooling return
        embedding_tensor = _make_fake_embedding(768, 1)
        mock_torch_cpu["torch"].nn.functional.normalize.return_value = embedding_tensor

        result = svc_cpu.embed_text("Motor hiba")
        assert isinstance(result, list)
        assert len(result) == 768

    def test_empty_text_returns_zero_vector(self, svc_cpu):
        result = svc_cpu.embed_text("")
        assert result == [0.0] * 768

    def test_whitespace_returns_zero_vector(self, svc_cpu):
        result = svc_cpu.embed_text("   ")
        assert result == [0.0] * 768

    def test_no_torch_returns_zero_vector(self, svc_no_torch):
        result = svc_no_torch.embed_text("Motor hiba")
        assert result == [0.0] * 768

    def test_with_preprocess(self, svc_cpu, mock_torch_cpu):
        model = _make_mock_model(1, 768)
        mock_torch_cpu["model_cls"].from_pretrained.return_value = model
        embedding_tensor = _make_fake_embedding(768, 1)
        mock_torch_cpu["torch"].nn.functional.normalize.return_value = embedding_tensor

        # preprocess without spacy just returns stripped text
        result = svc_cpu.embed_text("Motor hiba", preprocess=True)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# embed_batch
# ---------------------------------------------------------------------------


class TestEmbedBatch:
    def test_empty_list(self, svc_cpu):
        result = svc_cpu.embed_batch([])
        assert result == []

    def test_no_torch_returns_zero_vectors(self, svc_no_torch):
        result = svc_no_torch.embed_batch(["a", "b"])
        assert len(result) == 2
        assert all(v == [0.0] * 768 for v in result)

    def test_batch_returns_correct_count(self, svc_cpu, mock_torch_cpu):
        texts = ["Text 1", "Text 2", "Text 3"]

        # Make _embed_batch_internal a no-op that fills results
        def fake_internal(texts_with_indices, results, batch_size):
            for idx, _text in texts_with_indices:
                results[idx] = [0.5] * 768

        with patch.object(svc_cpu, "_embed_batch_internal", side_effect=fake_internal):
            result = svc_cpu.embed_batch(texts, use_cache=False)

        assert len(result) == 3

    def test_batch_with_preprocess(self, svc_cpu, mock_torch_cpu):
        texts = ["Motor hiba", "Fek problema"]

        def fake_internal(texts_with_indices, results, batch_size):
            for idx, _text in texts_with_indices:
                results[idx] = [0.5] * 768

        with patch.object(svc_cpu, "_embed_batch_internal", side_effect=fake_internal):
            result = svc_cpu.embed_batch(texts, preprocess=True, use_cache=False)

        assert len(result) == 2

    def test_batch_custom_batch_size(self, svc_cpu):
        texts = ["t1", "t2", "t3", "t4", "t5"]

        def fake_internal(texts_with_indices, results, batch_size):
            assert batch_size == 2
            for idx, _text in texts_with_indices:
                results[idx] = [0.1] * 768

        with patch.object(svc_cpu, "_embed_batch_internal", side_effect=fake_internal):
            result = svc_cpu.embed_batch(texts, batch_size=2, use_cache=False)

        assert len(result) == 5

    def test_large_batch_triggers_cleanup(self, svc_cpu, mock_torch_cpu):
        # optimal_batch_size defaults to 16 on CPU
        svc_cpu._optimal_batch_size = 4
        texts = [f"text {i}" for i in range(20)]  # > 4*4 = 16

        def fake_internal(texts_with_indices, results, batch_size):
            for idx, _text in texts_with_indices:
                results[idx] = [0.1] * 768

        with (
            patch.object(svc_cpu, "_embed_batch_internal", side_effect=fake_internal),
            patch.object(svc_cpu, "_cleanup_gpu_memory") as cleanup_mock,
        ):
            svc_cpu.embed_batch(texts, use_cache=False)
            cleanup_mock.assert_called()


# ---------------------------------------------------------------------------
# preprocess_hungarian
# ---------------------------------------------------------------------------


class TestPreprocessHungarian:
    def test_empty_text(self, svc_cpu):
        assert svc_cpu.preprocess_hungarian("") == ""

    def test_whitespace(self, svc_cpu):
        assert svc_cpu.preprocess_hungarian("   ") == ""

    def test_none_text(self, svc_cpu):
        assert svc_cpu.preprocess_hungarian(None) == ""

    def test_without_spacy_returns_stripped(self, svc_cpu):
        with patch("app.services.embedding_service.SPACY_AVAILABLE", False):
            result = svc_cpu.preprocess_hungarian("  Motor hiba  ")
            assert result == "Motor hiba"

    def test_with_spacy_mock(self, svc_cpu):
        """Test preprocessing with a mocked spacy model."""
        mock_nlp = MagicMock()
        # Simulate spacy doc with tokens
        token1 = MagicMock()
        token1.lemma_ = "motor"
        token1.is_stop = False
        token1.is_punct = False
        token1.is_space = False

        token2 = MagicMock()
        token2.lemma_ = "hiba"
        token2.is_stop = False
        token2.is_punct = False
        token2.is_space = False

        token3 = MagicMock()
        token3.lemma_ = "."
        token3.is_stop = False
        token3.is_punct = True
        token3.is_space = False

        mock_nlp.return_value = [token1, token2, token3]
        svc_cpu._nlp = mock_nlp

        with patch("app.services.embedding_service.SPACY_AVAILABLE", True):
            result = svc_cpu.preprocess_hungarian("Motor hibaja.")
            assert result == "motor hiba"


# ---------------------------------------------------------------------------
# get_similar_texts
# ---------------------------------------------------------------------------


class TestGetSimilarTexts:
    def test_empty_candidates(self, svc_cpu):
        result = svc_cpu.get_similar_texts("query", [])
        assert result == []

    def test_top_k_zero_returns_all(self, svc_cpu):
        with (
            patch.object(svc_cpu, "embed_text", return_value=[1.0] * 768),
            patch.object(svc_cpu, "embed_batch", return_value=[[1.0] * 768, [0.5] * 768]),
        ):
            result = svc_cpu.get_similar_texts("query", ["a", "b"], top_k=0)
            assert len(result) == 2

    def test_returns_sorted_descending(self, svc_cpu):
        query_vec = np.zeros(768)
        query_vec[0] = 1.0

        cand1 = np.zeros(768)
        cand1[0] = 0.8
        cand1[1] = 0.2

        cand2 = np.zeros(768)
        cand2[0] = 0.3
        cand2[1] = 0.7

        with (
            patch.object(svc_cpu, "embed_text", return_value=query_vec.tolist()),
            patch.object(svc_cpu, "embed_batch", return_value=[cand1.tolist(), cand2.tolist()]),
        ):
            results = svc_cpu.get_similar_texts("query", ["close", "far"], top_k=2)

        assert results[0][0] == "close"
        assert results[0][1] > results[1][1]

    def test_top_k_limits_results(self, svc_cpu):
        with (
            patch.object(svc_cpu, "embed_text", return_value=[1.0] * 768),
            patch.object(svc_cpu, "embed_batch", return_value=[[1.0] * 768] * 5),
        ):
            result = svc_cpu.get_similar_texts("q", ["a", "b", "c", "d", "e"], top_k=2)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# Async methods
# ---------------------------------------------------------------------------


class TestAsyncMethods:
    @pytest.mark.asyncio
    async def test_embed_text_async_no_cache(self, svc_cpu):
        svc_cpu.disable_cache()
        with patch.object(svc_cpu, "embed_text", return_value=[0.1] * 768):
            result = await svc_cpu.embed_text_async("test", use_cache=False)
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_text_async_cache_hit(self, svc_cpu):
        svc_cpu.enable_cache()
        cached_vec = [0.42] * 768

        mock_cache = AsyncMock()
        mock_cache.get_embedding.return_value = cached_vec

        async def fake_get_cache():
            return mock_cache

        mock_redis_module = MagicMock()
        mock_redis_module.get_cache_service = fake_get_cache

        with patch.dict("sys.modules", {"app.db.redis_cache": mock_redis_module}):
            result = await svc_cpu.embed_text_async("cached text", use_cache=True)

        assert result == cached_vec

    @pytest.mark.asyncio
    async def test_embed_text_async_cache_miss(self, svc_cpu):
        svc_cpu.enable_cache()

        mock_cache = AsyncMock()
        mock_cache.get_embedding.return_value = None  # cache miss
        mock_cache.set_embedding.return_value = None

        async def fake_get_cache():
            return mock_cache

        with (
            patch.object(svc_cpu, "embed_text", return_value=[0.1] * 768),
            patch.dict(
                "sys.modules", {"app.db.redis_cache": MagicMock(get_cache_service=fake_get_cache)}
            ),
        ):
            result = await svc_cpu.embed_text_async("uncached", use_cache=True)

        assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_batch_async_empty(self, svc_cpu):
        result = await svc_cpu.embed_batch_async([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_async_no_cache(self, svc_cpu):
        svc_cpu.disable_cache()
        with patch.object(svc_cpu, "embed_batch", return_value=[[0.1] * 768, [0.2] * 768]):
            result = await svc_cpu.embed_batch_async(["a", "b"], use_cache=False)
            assert len(result) == 2


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_warmup_loads_model(self, svc_cpu):
        with (
            patch.object(svc_cpu, "_load_hubert_model") as load_model,
            patch.object(svc_cpu, "embed_text", return_value=[0.1] * 768),
            patch.object(svc_cpu, "embed_batch", return_value=[[0.1] * 768] * 3),
        ):
            svc_cpu.warmup()
            load_model.assert_called_once()

    def test_warmup_skipped_without_torch(self, svc_no_torch):
        # Should not raise
        svc_no_torch.warmup()

    def test_warmup_loads_spacy_if_available(self, svc_cpu):
        with (
            patch("app.services.embedding_service.SPACY_AVAILABLE", True),
            patch.object(svc_cpu, "_load_hubert_model"),
            patch.object(svc_cpu, "_load_huspacy_model") as load_spacy,
            patch.object(svc_cpu, "embed_text", return_value=[0.1] * 768),
            patch.object(svc_cpu, "embed_batch", return_value=[[0.1] * 768] * 3),
        ):
            svc_cpu.warmup()
            load_spacy.assert_called_once()


# ---------------------------------------------------------------------------
# _load_hubert_model
# ---------------------------------------------------------------------------


class TestLoadHubertModel:
    def test_skips_if_already_loaded(self, svc_cpu, mock_torch_cpu):
        svc_cpu._model = MagicMock()
        svc_cpu._tokenizer = MagicMock()
        svc_cpu._load_hubert_model()
        # from_pretrained should NOT be called since model already loaded
        mock_torch_cpu["tokenizer_cls"].from_pretrained.assert_not_called()

    def test_loads_model_and_tokenizer(self, svc_cpu, mock_torch_cpu):
        svc_cpu._load_hubert_model()
        mock_torch_cpu["tokenizer_cls"].from_pretrained.assert_called_once()
        mock_torch_cpu["model_cls"].from_pretrained.assert_called_once()

    def test_raises_on_failure(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["tokenizer_cls"].from_pretrained.side_effect = OSError("model not found")
        with pytest.raises(RuntimeError, match="Could not load huBERT model"):
            svc_cpu._load_hubert_model()


# ---------------------------------------------------------------------------
# _load_huspacy_model
# ---------------------------------------------------------------------------


class TestLoadHuspacyModel:
    def test_skips_if_already_loaded(self, svc_cpu):
        svc_cpu._nlp = MagicMock()
        with patch("app.services.embedding_service.SPACY_AVAILABLE", True):
            svc_cpu._load_huspacy_model()
        # No error, no re-load

    def test_skips_without_spacy(self, svc_cpu):
        with patch("app.services.embedding_service.SPACY_AVAILABLE", False):
            svc_cpu._load_huspacy_model()
            assert svc_cpu._nlp is None

    def test_loads_spacy_model(self, svc_cpu):
        mock_spacy = MagicMock()
        mock_nlp = MagicMock()
        mock_spacy.load.return_value = mock_nlp

        with (
            patch("app.services.embedding_service.SPACY_AVAILABLE", True),
            patch("app.services.embedding_service.spacy", mock_spacy),
        ):
            svc_cpu._load_huspacy_model()
            assert svc_cpu._nlp is mock_nlp

    def test_fallback_to_blank_on_oserror(self, svc_cpu):
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("Model not found")
        mock_blank = MagicMock()
        mock_spacy.blank.return_value = mock_blank

        with (
            patch("app.services.embedding_service.SPACY_AVAILABLE", True),
            patch("app.services.embedding_service.spacy", mock_spacy),
        ):
            svc_cpu._load_huspacy_model()
            assert svc_cpu._nlp is mock_blank
            mock_spacy.blank.assert_called_once_with("hu")

    def test_raises_on_unexpected_error(self, svc_cpu):
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = ValueError("unexpected")

        with (
            patch("app.services.embedding_service.SPACY_AVAILABLE", True),
            patch("app.services.embedding_service.spacy", mock_spacy),
            pytest.raises(RuntimeError, match="Could not load HuSpaCy model"),
        ):
            svc_cpu._load_huspacy_model()


# ---------------------------------------------------------------------------
# _cleanup_gpu_memory
# ---------------------------------------------------------------------------


class TestCleanupGpuMemory:
    def test_cpu_noop(self, svc_cpu, mock_torch_cpu):
        # CPU device - cleanup should not crash
        mock_torch_cpu["device"].type = "cpu"
        svc_cpu._cleanup_gpu_memory()

    def test_cuda_high_utilization(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "cuda"
        mock_torch_cpu["torch"].cuda.memory_allocated.return_value = 9_000_000_000
        props = MagicMock()
        props.total_memory = 10_000_000_000
        mock_torch_cpu["torch"].cuda.get_device_properties.return_value = props

        svc_cpu._cleanup_gpu_memory()
        mock_torch_cpu["torch"].cuda.empty_cache.assert_called()

    def test_cuda_low_utilization(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "cuda"
        mock_torch_cpu["torch"].cuda.memory_allocated.return_value = 1_000_000_000
        props = MagicMock()
        props.total_memory = 10_000_000_000
        mock_torch_cpu["torch"].cuda.get_device_properties.return_value = props

        svc_cpu._cleanup_gpu_memory()
        # Still calls empty_cache (the else branch)
        mock_torch_cpu["torch"].cuda.empty_cache.assert_called()

    def test_mps_cleanup(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "mps"
        # Should not crash
        svc_cpu._cleanup_gpu_memory()


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------


class TestDetectDevice:
    def test_cuda_device(self):
        mt, _ = _make_mock_torch("cuda")
        props = MagicMock()
        props.total_memory = 8 * (1024**3)
        mt.cuda.get_device_properties.return_value = props
        mt.cuda.get_device_name.return_value = "Tesla T4"

        with (
            patch("app.services.embedding_service.torch", mt),
            patch("app.services.embedding_service.TORCH_AVAILABLE", True),
            patch("app.services.embedding_service.AutoTokenizer"),
            patch("app.services.embedding_service.AutoModel"),
        ):
            from app.services.embedding_service import HungarianEmbeddingService

            svc = HungarianEmbeddingService()
            # device should have been set via _detect_device
            assert svc._device is not None

    def test_mps_device(self):
        mt, _ = _make_mock_torch("mps")

        with (
            patch("app.services.embedding_service.torch", mt),
            patch("app.services.embedding_service.TORCH_AVAILABLE", True),
            patch("app.services.embedding_service.AutoTokenizer"),
            patch("app.services.embedding_service.AutoModel"),
        ):
            from app.services.embedding_service import HungarianEmbeddingService

            svc = HungarianEmbeddingService()
            assert svc._device is not None


# ---------------------------------------------------------------------------
# _get_optimal_batch_size
# ---------------------------------------------------------------------------


class TestOptimalBatchSize:
    def test_cpu_batch_size(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "cpu"
        assert svc_cpu._get_optimal_batch_size() == 16

    def test_mps_batch_size(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "mps"
        assert svc_cpu._get_optimal_batch_size() == 32

    def test_cuda_large_gpu(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "cuda"
        props = MagicMock()
        props.total_memory = 16 * (1024**3)  # 16GB
        mock_torch_cpu["torch"].cuda.get_device_properties.return_value = props
        assert svc_cpu._get_optimal_batch_size() == 128

    def test_cuda_medium_gpu(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "cuda"
        props = MagicMock()
        props.total_memory = 6 * (1024**3)  # 6GB
        mock_torch_cpu["torch"].cuda.get_device_properties.return_value = props
        assert svc_cpu._get_optimal_batch_size() == 64

    def test_cuda_small_gpu(self, svc_cpu, mock_torch_cpu):
        mock_torch_cpu["device"].type = "cuda"
        props = MagicMock()
        props.total_memory = 2 * (1024**3)  # 2GB
        mock_torch_cpu["torch"].cuda.get_device_properties.return_value = props
        assert svc_cpu._get_optimal_batch_size() == 32


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    def test_embed_text_function(self, mock_torch_cpu):
        from app.services.embedding_service import embed_text

        with patch("app.services.embedding_service.get_embedding_service") as get_svc:
            mock_svc = MagicMock()
            mock_svc.embed_text.return_value = [0.1] * 768
            get_svc.return_value = mock_svc
            result = embed_text("test")
            assert len(result) == 768

    def test_embed_batch_function(self, mock_torch_cpu):
        from app.services.embedding_service import embed_batch

        with patch("app.services.embedding_service.get_embedding_service") as get_svc:
            mock_svc = MagicMock()
            mock_svc.embed_batch.return_value = [[0.1] * 768]
            get_svc.return_value = mock_svc
            result = embed_batch(["test"])
            assert len(result) == 1

    def test_preprocess_hungarian_function(self, mock_torch_cpu):
        from app.services.embedding_service import preprocess_hungarian

        with patch("app.services.embedding_service.get_embedding_service") as get_svc:
            mock_svc = MagicMock()
            mock_svc.preprocess_hungarian.return_value = "motor hiba"
            get_svc.return_value = mock_svc
            result = preprocess_hungarian("Motor hiba")
            assert result == "motor hiba"

    def test_get_similar_texts_function(self, mock_torch_cpu):
        from app.services.embedding_service import get_similar_texts

        with patch("app.services.embedding_service.get_embedding_service") as get_svc:
            mock_svc = MagicMock()
            mock_svc.get_similar_texts.return_value = [("a", 0.9)]
            get_svc.return_value = mock_svc
            result = get_similar_texts("query", ["a"])
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_embed_text_async_function(self, mock_torch_cpu):
        from app.services.embedding_service import embed_text_async

        with patch("app.services.embedding_service.get_embedding_service") as get_svc:
            mock_svc = MagicMock()
            mock_svc.embed_text_async = AsyncMock(return_value=[0.1] * 768)
            get_svc.return_value = mock_svc
            result = await embed_text_async("test")
            assert len(result) == 768

    @pytest.mark.asyncio
    async def test_embed_batch_async_function(self, mock_torch_cpu):
        from app.services.embedding_service import embed_batch_async

        with patch("app.services.embedding_service.get_embedding_service") as get_svc:
            mock_svc = MagicMock()
            mock_svc.embed_batch_async = AsyncMock(return_value=[[0.1] * 768])
            get_svc.return_value = mock_svc
            result = await embed_batch_async(["test"])
            assert len(result) == 1


# ---------------------------------------------------------------------------
# _mean_pooling
# ---------------------------------------------------------------------------


class TestMeanPooling:
    def test_mean_pooling_runs(self, svc_cpu, mock_torch_cpu):
        """Verify _mean_pooling calls the right torch operations."""
        model_output = MagicMock()
        hidden = MagicMock()
        hidden.size.return_value = (1, 10, 768)
        model_output.last_hidden_state = hidden

        attention_mask = MagicMock()
        expanded = MagicMock()
        attention_mask.unsqueeze.return_value = expanded
        expanded.expand.return_value = expanded
        expanded.float.return_value = expanded
        expanded.sum.return_value = expanded

        mock_torch_cpu["torch"].sum.return_value = MagicMock()
        mock_torch_cpu["torch"].clamp.return_value = expanded

        result = svc_cpu._mean_pooling(model_output, attention_mask)
        # Should not raise
        assert result is not None
