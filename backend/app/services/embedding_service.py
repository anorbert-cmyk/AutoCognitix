"""
Hungarian NLP and Embedding Service.

This module provides Hungarian language text embedding using huBERT model
and text preprocessing using HuSpaCy.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import spacy
import torch
from transformers import AutoModel, AutoTokenizer

from app.core.config import settings

logger = logging.getLogger(__name__)


class HungarianEmbeddingService:
    """
    Service for generating Hungarian text embeddings using huBERT model.

    Features:
    - huBERT model (SZTAKI-HLT/hubert-base-cc) for embeddings
    - HuSpaCy integration for text preprocessing
    - Automatic GPU/CPU detection
    - Batch processing support
    - Cosine similarity based text matching
    """

    _instance: Optional["HungarianEmbeddingService"] = None

    def __new__(cls) -> "HungarianEmbeddingService":
        """Singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the embedding service with models."""
        if self._initialized:
            return

        self._initialized = True
        self._device = self._detect_device()
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None
        self._nlp: Optional[spacy.Language] = None

        logger.info(f"HungarianEmbeddingService initialized with device: {self._device}")

    def _detect_device(self) -> torch.device:
        """
        Automatically detect and return the best available device.

        Returns:
            torch.device: CUDA if available, MPS for Apple Silicon, otherwise CPU.
        """
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

    def _load_hubert_model(self) -> None:
        """Load huBERT model and tokenizer lazily."""
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading huBERT model: {settings.HUBERT_MODEL}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.HUBERT_MODEL)
            self._model = AutoModel.from_pretrained(settings.HUBERT_MODEL)
            self._model.to(self._device)
            self._model.eval()
            logger.info("huBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load huBERT model: {e}")
            raise RuntimeError(f"Could not load huBERT model: {e}") from e

    def _load_huspacy_model(self) -> None:
        """Load HuSpaCy model lazily."""
        if self._nlp is not None:
            return

        logger.info(f"Loading HuSpaCy model: {settings.HUSPACY_MODEL}")

        try:
            self._nlp = spacy.load(settings.HUSPACY_MODEL)
            logger.info("HuSpaCy model loaded successfully")
        except OSError:
            logger.warning(
                f"HuSpaCy model '{settings.HUSPACY_MODEL}' not found. "
                "Install it with: python -m spacy download hu_core_news_lg"
            )
            # Fallback to blank Hungarian model
            self._nlp = spacy.blank("hu")
            logger.info("Using blank Hungarian spaCy model as fallback")
        except Exception as e:
            logger.error(f"Failed to load HuSpaCy model: {e}")
            raise RuntimeError(f"Could not load HuSpaCy model: {e}") from e

    def _mean_pooling(
        self,
        model_output: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mean pooling to token embeddings.

        Args:
            model_output: Model output containing last_hidden_state.
            attention_mask: Attention mask for valid tokens.

        Returns:
            torch.Tensor: Mean pooled sentence embedding.
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output.last_hidden_state

        # Expand attention mask to match embedding dimension
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        # Sum embeddings and divide by the number of valid tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def preprocess_hungarian(self, text: str) -> str:
        """
        Preprocess Hungarian text using HuSpaCy.

        Applies lemmatization and removes stopwords and punctuation.

        Args:
            text: Input text to preprocess.

        Returns:
            str: Preprocessed text with lemmatized tokens.
        """
        self._load_huspacy_model()

        if not text or not text.strip():
            return ""

        doc = self._nlp(text)

        # Extract lemmas, excluding stopwords and punctuation
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

        return " ".join(tokens)

    def embed_text(self, text: str, preprocess: bool = False) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.
            preprocess: Whether to apply Hungarian preprocessing first.

        Returns:
            List[float]: 768-dimensional embedding vector.
        """
        self._load_hubert_model()

        if preprocess:
            text = self.preprocess_hungarian(text)

        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * settings.EMBEDDING_DIMENSION

        # Tokenize
        encoded = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move to device
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        # Generate embeddings
        with torch.no_grad():
            model_output = self._model(**encoded)

        # Apply mean pooling
        embedding = self._mean_pooling(model_output, encoded["attention_mask"])

        # Normalize embedding
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        # Convert to list
        return embedding.squeeze().cpu().tolist()

    def embed_batch(
        self,
        texts: List[str],
        preprocess: bool = False,
        batch_size: int = 32
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batch processing.

        Args:
            texts: List of input texts to embed.
            preprocess: Whether to apply Hungarian preprocessing first.
            batch_size: Number of texts to process in each batch.

        Returns:
            List[List[float]]: List of 768-dimensional embedding vectors.
        """
        self._load_hubert_model()

        if not texts:
            return []

        # Preprocess if requested
        if preprocess:
            texts = [self.preprocess_hungarian(text) for text in texts]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Handle empty texts in batch
            non_empty_indices = []
            non_empty_texts = []

            for idx, text in enumerate(batch_texts):
                if text and text.strip():
                    non_empty_indices.append(idx)
                    non_empty_texts.append(text)

            # Initialize batch embeddings with zeros
            batch_embeddings = [
                [0.0] * settings.EMBEDDING_DIMENSION
                for _ in range(len(batch_texts))
            ]

            if non_empty_texts:
                # Tokenize batch
                encoded = self._tokenizer(
                    non_empty_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # Move to device
                encoded = {k: v.to(self._device) for k, v in encoded.items()}

                # Generate embeddings
                with torch.no_grad():
                    model_output = self._model(**encoded)

                # Apply mean pooling
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # Convert to list and assign to correct positions
                embeddings_list = embeddings.cpu().tolist()
                for orig_idx, embedding in zip(non_empty_indices, embeddings_list):
                    batch_embeddings[orig_idx] = embedding

            all_embeddings.extend(batch_embeddings)

            logger.debug(f"Processed batch {i // batch_size + 1}, texts: {len(batch_texts)}")

        return all_embeddings

    def get_similar_texts(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        preprocess: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Find most similar texts to a query using cosine similarity.

        Args:
            query: Query text to compare against.
            candidates: List of candidate texts to search through.
            top_k: Number of top similar texts to return.
            preprocess: Whether to apply Hungarian preprocessing.

        Returns:
            List[Tuple[str, float]]: List of (text, similarity_score) tuples,
                sorted by similarity in descending order.
        """
        if not candidates:
            return []

        if top_k <= 0:
            top_k = len(candidates)

        # Generate query embedding
        query_embedding = self.embed_text(query, preprocess=preprocess)
        query_vec = np.array(query_embedding)

        # Generate candidate embeddings
        candidate_embeddings = self.embed_batch(candidates, preprocess=preprocess)

        # Calculate cosine similarities
        similarities = []
        for idx, candidate_embedding in enumerate(candidate_embeddings):
            candidate_vec = np.array(candidate_embedding)

            # Cosine similarity (vectors are already normalized)
            similarity = float(np.dot(query_vec, candidate_vec))
            similarities.append((candidates[idx], similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return similarities[:top_k]

    @property
    def device(self) -> torch.device:
        """Get the current device being used."""
        return self._device

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return settings.EMBEDDING_DIMENSION

    @property
    def is_model_loaded(self) -> bool:
        """Check if the huBERT model is loaded."""
        return self._model is not None and self._tokenizer is not None

    def warmup(self) -> None:
        """
        Warm up the service by loading all models.

        Call this at application startup to avoid cold start latency.
        """
        logger.info("Warming up HungarianEmbeddingService...")
        self._load_hubert_model()
        self._load_huspacy_model()

        # Run a test embedding to warm up GPU
        _ = self.embed_text("Teszt szoveg a bemelegiteshez.")
        logger.info("HungarianEmbeddingService warmup complete")


# Global service instance
_embedding_service: Optional[HungarianEmbeddingService] = None


def get_embedding_service() -> HungarianEmbeddingService:
    """
    Get the global embedding service instance.

    Returns:
        HungarianEmbeddingService: The singleton embedding service instance.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = HungarianEmbeddingService()
    return _embedding_service


# Convenience functions for direct usage
def embed_text(text: str, preprocess: bool = False) -> List[float]:
    """
    Generate embedding for a single text.

    Args:
        text: Input text to embed.
        preprocess: Whether to apply Hungarian preprocessing first.

    Returns:
        List[float]: 768-dimensional embedding vector.
    """
    return get_embedding_service().embed_text(text, preprocess)


def embed_batch(
    texts: List[str],
    preprocess: bool = False,
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts with batch processing.

    Args:
        texts: List of input texts to embed.
        preprocess: Whether to apply Hungarian preprocessing first.
        batch_size: Number of texts to process in each batch.

    Returns:
        List[List[float]]: List of 768-dimensional embedding vectors.
    """
    return get_embedding_service().embed_batch(texts, preprocess, batch_size)


def preprocess_hungarian(text: str) -> str:
    """
    Preprocess Hungarian text using HuSpaCy.

    Args:
        text: Input text to preprocess.

    Returns:
        str: Preprocessed text with lemmatized tokens.
    """
    return get_embedding_service().preprocess_hungarian(text)


def get_similar_texts(
    query: str,
    candidates: List[str],
    top_k: int = 5,
    preprocess: bool = False
) -> List[Tuple[str, float]]:
    """
    Find most similar texts to a query using cosine similarity.

    Args:
        query: Query text to compare against.
        candidates: List of candidate texts to search through.
        top_k: Number of top similar texts to return.
        preprocess: Whether to apply Hungarian preprocessing.

    Returns:
        List[Tuple[str, float]]: List of (text, similarity_score) tuples.
    """
    return get_embedding_service().get_similar_texts(query, candidates, top_k, preprocess)
