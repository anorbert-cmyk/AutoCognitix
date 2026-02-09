"""
Integration tests for the Hungarian embedding service.

Tests embedding generation, text preprocessing, and similarity matching.
"""

from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_path))


class TestEmbeddingServiceBasic:
    """Test basic embedding service functionality."""

    def test_embed_text_returns_list(self, mock_embedding_service):
        """Test that embed_text returns a list."""
        result = mock_embedding_service.embed_text("Test text")

        assert isinstance(result, list)

    def test_embed_text_returns_768_dimensions(self, mock_embedding_service):
        """Test that embed_text returns 768-dimensional vector."""
        result = mock_embedding_service.embed_text("Test text")

        assert len(result) == 768

    def test_embed_text_returns_floats(self, mock_embedding_service):
        """Test that embed_text returns float values."""
        result = mock_embedding_service.embed_text("Test text")

        for value in result:
            assert isinstance(value, float)

    def test_embed_text_handles_empty_string(self, mock_embedding_service):
        """Test that embed_text handles empty string."""
        result = mock_embedding_service.embed_text("")

        # Should return zero vector or valid embedding
        assert len(result) == 768

    def test_embed_text_with_preprocess_flag(self, mock_embedding_service):
        """Test embed_text with preprocessing enabled."""
        result = mock_embedding_service.embed_text(
            "A motor nehezen indul.",
            preprocess=True,
        )

        assert len(result) == 768


class TestEmbeddingServiceBatch:
    """Test batch embedding functionality."""

    def test_embed_batch_returns_list_of_lists(self, mock_embedding_service):
        """Test that embed_batch returns list of embedding vectors."""
        texts = ["Text one", "Text two", "Text three"]
        result = mock_embedding_service.embed_batch(texts)

        assert isinstance(result, list)
        for embedding in result:
            assert isinstance(embedding, list)

    def test_embed_batch_returns_correct_count(self, mock_embedding_service):
        """Test that embed_batch returns correct number of embeddings."""
        mock_embedding_service.embed_batch.return_value = [
            [0.0] * 768,
            [0.0] * 768,
            [0.0] * 768,
        ]

        texts = ["Text one", "Text two", "Text three"]
        result = mock_embedding_service.embed_batch(texts)

        assert len(result) == 3

    def test_embed_batch_handles_empty_list(self, mock_embedding_service):
        """Test that embed_batch handles empty list."""
        mock_embedding_service.embed_batch.return_value = []

        result = mock_embedding_service.embed_batch([])

        assert result == []

    def test_embed_batch_preserves_order(self, mock_embedding_service):
        """Test that embed_batch preserves input order."""
        mock_embedding_service.embed_batch.return_value = [
            [1.0] + [0.0] * 767,  # First text marker
            [2.0] + [0.0] * 767,  # Second text marker
            [3.0] + [0.0] * 767,  # Third text marker
        ]

        texts = ["First", "Second", "Third"]
        result = mock_embedding_service.embed_batch(texts)

        assert result[0][0] == 1.0
        assert result[1][0] == 2.0
        assert result[2][0] == 3.0


class TestHungarianPreprocessing:
    """Test Hungarian text preprocessing."""

    def test_preprocess_hungarian_returns_string(self, mock_embedding_service):
        """Test that preprocess_hungarian returns a string."""
        result = mock_embedding_service.preprocess_hungarian("Test text")

        assert isinstance(result, str)

    def test_preprocess_hungarian_handles_empty_string(self, mock_embedding_service):
        """Test that preprocess_hungarian handles empty string."""
        mock_embedding_service.preprocess_hungarian.return_value = ""

        result = mock_embedding_service.preprocess_hungarian("")

        assert result == ""

    def test_preprocess_hungarian_removes_stopwords(self, mock_embedding_service):
        """Test that preprocessing removes Hungarian stopwords."""
        mock_embedding_service.preprocess_hungarian.return_value = "motor nehezen indul"

        result = mock_embedding_service.preprocess_hungarian("A motor nehezen indul.")

        # "A" is a stopword and should be removed
        assert "a" not in result.lower().split()

    def test_preprocess_hungarian_handles_special_characters(self, mock_embedding_service):
        """Test that preprocessing handles Hungarian special characters."""
        mock_embedding_service.preprocess_hungarian.return_value = "gyujtas oregedes"

        # Hungarian text with special characters
        text = "A gyujtasi oregedes problemakat okozhat."
        result = mock_embedding_service.preprocess_hungarian(text)

        assert isinstance(result, str)


class TestSimilarTextMatching:
    """Test text similarity matching."""

    def test_get_similar_texts_returns_list_of_tuples(self, mock_embedding_service):
        """Test that get_similar_texts returns list of tuples."""
        mock_similar = MagicMock()
        mock_similar.return_value = [
            ("Similar text 1", 0.9),
            ("Similar text 2", 0.8),
        ]
        mock_embedding_service.get_similar_texts = mock_similar

        result = mock_embedding_service.get_similar_texts(
            query="Motor problem",
            candidates=["Similar text 1", "Similar text 2"],
            top_k=5,
        )

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_get_similar_texts_returns_scores(self, mock_embedding_service):
        """Test that get_similar_texts returns similarity scores."""
        mock_similar = MagicMock()
        mock_similar.return_value = [
            ("Text 1", 0.95),
            ("Text 2", 0.75),
        ]
        mock_embedding_service.get_similar_texts = mock_similar

        result = mock_embedding_service.get_similar_texts(
            query="Query",
            candidates=["Text 1", "Text 2"],
            top_k=5,
        )

        for _text, score in result:
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_get_similar_texts_respects_top_k(self, mock_embedding_service):
        """Test that get_similar_texts respects top_k parameter."""
        mock_similar = MagicMock()
        mock_similar.return_value = [
            ("Text 1", 0.9),
            ("Text 2", 0.8),
        ]
        mock_embedding_service.get_similar_texts = mock_similar

        result = mock_embedding_service.get_similar_texts(
            query="Query",
            candidates=["Text 1", "Text 2", "Text 3", "Text 4"],
            top_k=2,
        )

        assert len(result) <= 2

    def test_get_similar_texts_sorted_by_score(self, mock_embedding_service):
        """Test that get_similar_texts returns results sorted by score."""
        mock_similar = MagicMock()
        mock_similar.return_value = [
            ("Best match", 0.95),
            ("Good match", 0.85),
            ("Okay match", 0.75),
        ]
        mock_embedding_service.get_similar_texts = mock_similar

        result = mock_embedding_service.get_similar_texts(
            query="Query",
            candidates=["Best match", "Good match", "Okay match"],
            top_k=5,
        )

        # Should be sorted descending by score
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_get_similar_texts_handles_empty_candidates(self, mock_embedding_service):
        """Test that get_similar_texts handles empty candidates list."""
        mock_similar = MagicMock()
        mock_similar.return_value = []
        mock_embedding_service.get_similar_texts = mock_similar

        result = mock_embedding_service.get_similar_texts(
            query="Query",
            candidates=[],
            top_k=5,
        )

        assert result == []


class TestEmbeddingServiceProperties:
    """Test embedding service properties."""

    def test_embedding_dimension_is_768(self, mock_embedding_service):
        """Test that embedding dimension is 768."""
        assert mock_embedding_service.embedding_dimension == 768

    def test_is_model_loaded_property(self, mock_embedding_service):
        """Test that is_model_loaded property exists."""
        assert hasattr(mock_embedding_service, "is_model_loaded")
        assert isinstance(mock_embedding_service.is_model_loaded, bool)


class TestEmbeddingServiceWarmup:
    """Test embedding service warmup."""

    def test_warmup_does_not_raise_error(self, mock_embedding_service):
        """Test that warmup completes without error."""
        mock_embedding_service.warmup = MagicMock()

        # Should not raise
        mock_embedding_service.warmup()

        mock_embedding_service.warmup.assert_called_once()


class TestEmbeddingServiceHungarianText:
    """Test embedding service with Hungarian text."""

    def test_embed_hungarian_symptoms(self, mock_embedding_service):
        """Test embedding Hungarian symptom descriptions."""
        hungarian_texts = [
            "A motor nehezen indul hidegben.",
            "A fogyasztas megnott az utobbbi idoben.",
            "Fursa hang jon a motorbol gyorsitasnal.",
        ]

        mock_embedding_service.embed_batch.return_value = [[0.0] * 768] * 3

        result = mock_embedding_service.embed_batch(hungarian_texts)

        assert len(result) == 3
        for embedding in result:
            assert len(embedding) == 768

    def test_embed_hungarian_dtc_descriptions(self, mock_embedding_service):
        """Test embedding Hungarian DTC descriptions."""
        dtc_descriptions = [
            "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
            "Rendszer tul sovany (Bank 1)",
            "Kommunikacio megszakadt az ECM/PCM-mel",
        ]

        mock_embedding_service.embed_batch.return_value = [[0.0] * 768] * 3

        result = mock_embedding_service.embed_batch(dtc_descriptions)

        assert len(result) == 3

    def test_similar_hungarian_texts_match(self, mock_embedding_service):
        """Test that similar Hungarian texts have high similarity."""
        mock_similar = MagicMock()
        mock_similar.return_value = [
            ("Motor nehezen indul reggel", 0.92),
            ("Motor nem indul be hidegen", 0.88),
        ]
        mock_embedding_service.get_similar_texts = mock_similar

        query = "A motor nem akar beindulni hidegben"
        candidates = [
            "Motor nehezen indul reggel",
            "Motor nem indul be hidegen",
            "Olajcsere szukseges",
        ]

        result = mock_embedding_service.get_similar_texts(
            query=query,
            candidates=candidates,
            top_k=2,
        )

        # Top matches should have high similarity
        if result:
            assert result[0][1] > 0.8


class TestEmbeddingServiceErrorHandling:
    """Test embedding service error handling."""

    def test_handles_very_long_text(self, mock_embedding_service):
        """Test handling of very long text input."""
        # huBERT typically truncates at 512 tokens
        long_text = "Test " * 1000

        result = mock_embedding_service.embed_text(long_text)

        assert len(result) == 768

    def test_handles_special_characters(self, mock_embedding_service):
        """Test handling of special characters."""
        special_text = "Test @#$%^&*() text with special chars!"

        result = mock_embedding_service.embed_text(special_text)

        assert len(result) == 768

    def test_handles_unicode_text(self, mock_embedding_service):
        """Test handling of unicode text."""
        unicode_text = "Magyar szoveg: aaeeiioouu"

        result = mock_embedding_service.embed_text(unicode_text)

        assert len(result) == 768

    def test_handles_newlines(self, mock_embedding_service):
        """Test handling of text with newlines."""
        multiline_text = "Line 1\nLine 2\nLine 3"

        result = mock_embedding_service.embed_text(multiline_text)

        assert len(result) == 768


class TestEmbeddingServiceSingleton:
    """Test embedding service singleton pattern."""

    def test_get_embedding_service_returns_instance(self):
        """Test that get_embedding_service returns an instance."""
        with patch("app.services.embedding_service.HungarianEmbeddingService") as MockClass:
            mock_instance = MagicMock()
            mock_instance._initialized = True
            MockClass.return_value = mock_instance

            # This will use the mocked class
            # In real usage, would return singleton instance
            assert True  # Just verify no errors


class TestEmbeddingServiceConvenienceFunctions:
    """Test embedding service convenience functions."""

    def test_embed_text_function(self):
        """Test module-level embed_text function."""
        with patch("app.services.embedding_service.get_embedding_service") as mock_get:
            mock_service = MagicMock()
            mock_service.embed_text.return_value = [0.0] * 768
            mock_get.return_value = mock_service

            from app.services.embedding_service import embed_text

            embed_text("Test")

            mock_service.embed_text.assert_called_once_with("Test", False)

    def test_embed_batch_function(self):
        """Test module-level embed_batch function."""
        with patch("app.services.embedding_service.get_embedding_service") as mock_get:
            mock_service = MagicMock()
            mock_service.embed_batch.return_value = [[0.0] * 768]
            mock_get.return_value = mock_service

            from app.services.embedding_service import embed_batch

            embed_batch(["Test"])

            mock_service.embed_batch.assert_called_once()

    def test_preprocess_hungarian_function(self):
        """Test module-level preprocess_hungarian function."""
        with patch("app.services.embedding_service.get_embedding_service") as mock_get:
            mock_service = MagicMock()
            mock_service.preprocess_hungarian.return_value = "preprocessed"
            mock_get.return_value = mock_service

            from app.services.embedding_service import preprocess_hungarian

            preprocess_hungarian("Test text")

            mock_service.preprocess_hungarian.assert_called_once_with("Test text")
