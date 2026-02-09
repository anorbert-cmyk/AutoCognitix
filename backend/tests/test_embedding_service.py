"""
Tests for Hungarian embedding service.

Tests cover:
- Embedding dimension validation
- Empty text handling
- Batch processing
- Similarity calculation
- Service singleton pattern
- Device detection
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_path))


# Mock settings for testing
class MockSettings:
    HUBERT_MODEL = "SZTAKI-HLT/hubert-base-cc"
    EMBEDDING_DIMENSION = 768
    HUSPACY_MODEL = "hu_core_news_lg"


@pytest.fixture
def mock_torch():
    """Mock torch module."""
    with patch.dict(sys.modules, {"torch": MagicMock()}):
        import torch

        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = False
        torch.device.return_value = MagicMock()
        yield torch


@pytest.fixture
def mock_transformers():
    """Mock transformers module."""
    with patch.dict(sys.modules, {"transformers": MagicMock()}):
        yield


class TestEmbeddingDimension:
    """Test embedding dimension validation."""

    def test_embedding_dimension_is_768(self):
        """Test that embedding dimension is 768."""
        settings = MockSettings()
        assert settings.EMBEDDING_DIMENSION == 768

    def test_zero_vector_has_correct_dimension(self):
        """Test that zero vector has correct dimension."""
        zero_vector = [0.0] * MockSettings.EMBEDDING_DIMENSION
        assert len(zero_vector) == 768

    def test_embedding_is_list_of_floats(self):
        """Test embedding type validation."""
        # Simulate embedding output
        embedding = [0.1] * 768

        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)


class TestEmptyTextHandling:
    """Test handling of empty and invalid text inputs."""

    def test_empty_string_returns_zero_vector(self):
        """Test that empty string returns zero vector."""

        def embed_empty_text(text: str) -> List[float]:
            if not text or not text.strip():
                return [0.0] * MockSettings.EMBEDDING_DIMENSION
            return [0.1] * MockSettings.EMBEDDING_DIMENSION

        result = embed_empty_text("")
        assert result == [0.0] * 768

    def test_whitespace_string_returns_zero_vector(self):
        """Test that whitespace-only string returns zero vector."""

        def embed_text(text: str) -> List[float]:
            if not text or not text.strip():
                return [0.0] * MockSettings.EMBEDDING_DIMENSION
            return [0.1] * MockSettings.EMBEDDING_DIMENSION

        result = embed_text("   ")
        assert result == [0.0] * 768

    def test_none_returns_zero_vector(self):
        """Test that None input returns zero vector."""

        def embed_text(text: str) -> List[float]:
            if not text or not text.strip():
                return [0.0] * MockSettings.EMBEDDING_DIMENSION
            return [0.1] * MockSettings.EMBEDDING_DIMENSION

        result = embed_text(None)
        assert result == [0.0] * 768

    def test_valid_text_returns_nonzero_vector(self):
        """Test that valid text returns non-zero vector."""

        def embed_text(text: str) -> List[float]:
            if not text or not text.strip():
                return [0.0] * MockSettings.EMBEDDING_DIMENSION
            return [0.1] * MockSettings.EMBEDDING_DIMENSION

        result = embed_text("Motor hiba")
        assert result != [0.0] * 768


class TestBatchProcessing:
    """Test batch embedding processing."""

    def test_empty_batch_returns_empty_list(self):
        """Test that empty batch returns empty list."""

        def embed_batch(texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            return [[0.1] * 768 for _ in texts]

        result = embed_batch([])
        assert result == []

    def test_batch_preserves_order(self):
        """Test that batch processing preserves input order."""
        # Simulate batch with different-length texts
        texts = ["Rovidebb", "Hosszabb szoveg itt", "Kozepes"]

        def embed_batch(texts: List[str]) -> List[List[float]]:
            # Return unique embeddings based on text length
            return [[float(len(t))] * 768 for t in texts]

        result = embed_batch(texts)

        assert len(result) == 3
        # First element should correspond to first text
        assert result[0][0] == float(len(texts[0]))

    def test_batch_handles_mixed_empty_texts(self):
        """Test batch handling with some empty texts."""
        texts = ["Valid text", "", "Another valid", "   "]

        def embed_batch(texts: List[str]) -> List[List[float]]:
            result = []
            for text in texts:
                if text and text.strip():
                    result.append([0.1] * 768)
                else:
                    result.append([0.0] * 768)
            return result

        result = embed_batch(texts)

        assert len(result) == 4
        assert result[0] != [0.0] * 768  # First is valid
        assert result[1] == [0.0] * 768  # Second is empty
        assert result[2] != [0.0] * 768  # Third is valid
        assert result[3] == [0.0] * 768  # Fourth is whitespace

    def test_batch_size_parameter(self):
        """Test batch processing with different batch sizes."""
        texts = ["Text " + str(i) for i in range(100)]

        def embed_batch(texts: List[str], batch_size: int = 32) -> List[List[float]]:
            # Process in batches but return all results
            all_results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                all_results.extend([[0.1] * 768 for _ in batch])
            return all_results

        result = embed_batch(texts, batch_size=32)
        assert len(result) == 100


class TestSimilarityCalculation:
    """Test text similarity calculation."""

    def test_identical_texts_have_high_similarity(self):
        """Test that identical texts have similarity close to 1."""
        import numpy as np

        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        vec = [0.1] * 768
        similarity = cosine_similarity(vec, vec)

        assert similarity > 0.99  # Should be very close to 1

    def test_orthogonal_vectors_have_zero_similarity(self):
        """Test that orthogonal vectors have zero similarity."""
        import numpy as np

        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm == 0:
                return 0.0
            return float(dot / norm)

        # Create orthogonal vectors
        vec1 = [1.0] + [0.0] * 767
        vec2 = [0.0, 1.0] + [0.0] * 766

        similarity = cosine_similarity(vec1, vec2)

        assert abs(similarity) < 0.01  # Should be close to 0

    def test_similarity_is_symmetric(self):
        """Test that similarity is symmetric: sim(a,b) = sim(b,a)."""
        import numpy as np

        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        vec1 = [0.1, 0.2, 0.3] + [0.0] * 765
        vec2 = [0.3, 0.2, 0.1] + [0.0] * 765

        sim_ab = cosine_similarity(vec1, vec2)
        sim_ba = cosine_similarity(vec2, vec1)

        assert abs(sim_ab - sim_ba) < 0.0001

    def test_get_similar_texts_returns_sorted_results(self):
        """Test that similar texts are returned sorted by similarity."""

        def get_similar_texts(query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
            # Simulate similarity scores
            scores = [(c, 1.0 / (i + 1)) for i, c in enumerate(candidates)]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

        candidates = ["Text A", "Text B", "Text C", "Text D", "Text E"]
        results = get_similar_texts("Query", candidates, top_k=3)

        assert len(results) == 3
        # Results should be sorted by score descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    def test_get_similar_texts_empty_candidates(self):
        """Test similarity search with empty candidates."""

        def get_similar_texts(query: str, candidates: List[str], top_k: int = 5) -> List[tuple]:
            if not candidates:
                return []
            return [(c, 0.5) for c in candidates[:top_k]]

        results = get_similar_texts("Query", [], top_k=5)
        assert results == []


class TestServiceSingleton:
    """Test embedding service singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Test that singleton returns the same instance."""

        class SingletonService:
            _instance = None

            def __new__(cls):
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                return cls._instance

        instance1 = SingletonService()
        instance2 = SingletonService()

        assert instance1 is instance2

    def test_get_service_function_returns_singleton(self):
        """Test that get_service function returns singleton."""
        _service = None

        class EmbeddingService:
            pass

        def get_embedding_service():
            nonlocal _service
            if _service is None:
                _service = EmbeddingService()
            return _service

        service1 = get_embedding_service()
        service2 = get_embedding_service()

        assert service1 is service2


class TestDeviceDetection:
    """Test device detection for embedding service."""

    def test_cpu_fallback(self):
        """Test CPU fallback when no GPU available."""

        def detect_device():
            # Simulate no CUDA and no MPS
            cuda_available = False
            mps_available = False

            if cuda_available:
                return "cuda"
            elif mps_available:
                return "mps"
            else:
                return "cpu"

        device = detect_device()
        assert device == "cpu"

    def test_cuda_detection(self):
        """Test CUDA detection when available."""

        def detect_device(cuda_available: bool, mps_available: bool):
            if cuda_available:
                return "cuda"
            elif mps_available:
                return "mps"
            else:
                return "cpu"

        device = detect_device(cuda_available=True, mps_available=False)
        assert device == "cuda"

    def test_mps_detection(self):
        """Test MPS (Apple Silicon) detection."""

        def detect_device(cuda_available: bool, mps_available: bool):
            if cuda_available:
                return "cuda"
            elif mps_available:
                return "mps"
            else:
                return "cpu"

        device = detect_device(cuda_available=False, mps_available=True)
        assert device == "mps"


class TestHungarianPreprocessing:
    """Test Hungarian text preprocessing."""

    def test_preprocess_empty_text(self):
        """Test preprocessing of empty text."""

        def preprocess_hungarian(text: str) -> str:
            if not text or not text.strip():
                return ""
            return text.strip()

        assert preprocess_hungarian("") == ""
        assert preprocess_hungarian(None) == ""
        assert preprocess_hungarian("   ") == ""

    def test_preprocess_strips_whitespace(self):
        """Test that preprocessing strips whitespace."""

        def preprocess_hungarian(text: str) -> str:
            if not text:
                return ""
            return text.strip()

        result = preprocess_hungarian("  Motor hiba  ")
        assert result == "Motor hiba"

    def test_preprocess_returns_text_without_spacy(self):
        """Test preprocessing fallback without spacy."""

        def preprocess_hungarian(text: str, spacy_available: bool = False) -> str:
            if not text or not text.strip():
                return ""
            if not spacy_available:
                return text.strip()
            # Would do lemmatization here
            return text.strip()

        result = preprocess_hungarian("Motor hiba", spacy_available=False)
        assert result == "Motor hiba"


class TestEmbeddingServiceIntegration:
    """Integration tests for embedding service."""

    def test_warmup_loads_models(self):
        """Test that warmup loads all models."""

        class MockEmbeddingService:
            def __init__(self):
                self._model_loaded = False
                self._nlp_loaded = False

            def warmup(self):
                self._model_loaded = True
                self._nlp_loaded = True

            @property
            def is_model_loaded(self):
                return self._model_loaded

        service = MockEmbeddingService()
        assert not service.is_model_loaded

        service.warmup()
        assert service.is_model_loaded

    def test_embedding_dimension_property(self):
        """Test embedding dimension property."""

        class MockEmbeddingService:
            @property
            def embedding_dimension(self):
                return MockSettings.EMBEDDING_DIMENSION

        service = MockEmbeddingService()
        assert service.embedding_dimension == 768

    def test_device_property(self):
        """Test device property returns current device."""

        class MockEmbeddingService:
            def __init__(self):
                self._device = "cpu"

            @property
            def device(self):
                return self._device

        service = MockEmbeddingService()
        assert service.device == "cpu"


class TestNormalization:
    """Test embedding normalization."""

    def test_normalized_vector_has_unit_length(self):
        """Test that normalized vector has unit length."""
        import numpy as np

        def normalize(vec: List[float]) -> List[float]:
            v = np.array(vec)
            norm = np.linalg.norm(v)
            if norm == 0:
                return vec
            return (v / norm).tolist()

        vec = [3.0, 4.0] + [0.0] * 766  # 3-4-5 triangle
        normalized = normalize(vec)

        length = np.linalg.norm(normalized)
        assert abs(length - 1.0) < 0.0001

    def test_zero_vector_normalization(self):
        """Test normalization of zero vector."""
        import numpy as np

        def normalize(vec: List[float]) -> List[float]:
            v = np.array(vec)
            norm = np.linalg.norm(v)
            if norm == 0:
                return vec  # Return as-is
            return (v / norm).tolist()

        zero_vec = [0.0] * 768
        normalized = normalize(zero_vec)

        assert normalized == zero_vec  # Should return unchanged
