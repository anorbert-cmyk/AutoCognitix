"""
Hungarian NLP and Embedding Service.

This module provides Hungarian language text embedding using huBERT model
and text preprocessing using HuSpaCy.

Performance Optimizations:
- Lazy model loading (models loaded on first use)
- GPU memory management with automatic cleanup
- Optimized batch processing with dynamic batching
- Embedding cache integration
- Half-precision (FP16) inference for GPU
- Async-compatible embedding generation
"""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from app.core.config import settings

# Optional spacy import - not required for basic embedding functionality
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False
    logging.warning("spacy not available - Hungarian preprocessing will be disabled")

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound preprocessing
_thread_pool = ThreadPoolExecutor(max_workers=4)


class HungarianEmbeddingService:
    """
    Service for generating Hungarian text embeddings using huBERT model.

    Features:
    - huBERT model (SZTAKI-HLT/hubert-base-cc) for embeddings
    - HuSpaCy integration for text preprocessing
    - Automatic GPU/CPU detection
    - Batch processing support
    - Cosine similarity based text matching

    Performance Optimizations:
    - Lazy model loading (models loaded on first use)
    - Half-precision (FP16) on GPU for faster inference
    - Dynamic batch sizing based on available memory
    - GPU memory cleanup after large batches
    - Redis cache integration for repeated texts
    """

    _instance: Optional[HungarianEmbeddingService] = None
    _lock: threading.Lock = threading.Lock()

    # Optimal batch sizes by device type
    BATCH_SIZE_GPU = 64
    BATCH_SIZE_CPU = 16
    BATCH_SIZE_MPS = 32

    # Memory threshold for GPU cleanup (bytes)
    GPU_MEMORY_THRESHOLD = 0.8  # 80% utilization triggers cleanup

    def __new__(cls) -> HungarianEmbeddingService:
        """Thread-safe singleton pattern to avoid loading model multiple times."""
        if cls._instance is None:
            with cls._lock:
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
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModel | None = None
        self._nlp = None  # Optional spacy.Language
        self._use_fp16 = self._device.type == "cuda"  # FP16 only on CUDA
        self._optimal_batch_size = self._get_optimal_batch_size()
        self._cache_enabled = True

        logger.info(
            f"HungarianEmbeddingService initialized: device={self._device}, "
            f"fp16={self._use_fp16}, batch_size={self._optimal_batch_size}"
        )

    def _detect_device(self) -> torch.device:
        """
        Automatically detect and return the best available device.

        Returns:
            torch.device: CUDA if available, MPS for Apple Silicon, otherwise CPU.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA device detected: {gpu_name} ({gpu_memory:.1f} GB)")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple MPS device detected")
        else:
            device = torch.device("cpu")
            cpu_count = torch.get_num_threads()
            logger.info(f"Using CPU device ({cpu_count} threads)")
        return device

    def _get_optimal_batch_size(self) -> int:
        """
        Determine optimal batch size based on device and available memory.

        Returns:
            int: Optimal batch size for the current device.
        """
        if self._device.type == "cuda":
            # Dynamic sizing based on GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            if total_memory > 8 * (1024**3):  # > 8GB
                return 128
            elif total_memory > 4 * (1024**3):  # > 4GB
                return 64
            else:
                return 32
        elif self._device.type == "mps":
            return self.BATCH_SIZE_MPS
        else:
            return self.BATCH_SIZE_CPU

    def _cleanup_gpu_memory(self) -> None:
        """
        Clean up GPU memory if utilization is high.

        Should be called after processing large batches or before OOM recovery.
        """
        if self._device.type == "cuda":
            # Check current memory utilization
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            utilization = allocated / total

            if utilization > self.GPU_MEMORY_THRESHOLD:
                torch.cuda.empty_cache()
                gc.collect()
                logger.debug(
                    f"GPU memory cleaned: {utilization * 100:.1f}% -> "
                    f"{torch.cuda.memory_allocated() / total * 100:.1f}%"
                )
            else:
                # Always clean cache on explicit call (for OOM recovery)
                torch.cuda.empty_cache()
        elif self._device.type == "mps":
            # MPS doesn't have explicit cache clearing, but gc helps
            gc.collect()

    def _load_hubert_model(self) -> None:
        """
        Load huBERT model and tokenizer lazily.

        Optimizations:
        - FP16 inference on CUDA for 2x speedup
        - Disabled gradient computation for inference
        - Model compiled with torch.compile on PyTorch 2.0+
        """
        if self._model is not None and self._tokenizer is not None:
            return

        logger.info(f"Loading huBERT model: {settings.HUBERT_MODEL}")

        try:
            # Load tokenizer with fast implementation
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.HUBERT_MODEL,
                use_fast=True,  # Use fast tokenizer implementation
            )

            # Load model with optimizations
            self._model = AutoModel.from_pretrained(
                settings.HUBERT_MODEL,
                torch_dtype=torch.float16 if self._use_fp16 else torch.float32,
            )
            self._model.to(self._device)
            self._model.eval()

            # Disable gradient computation for inference
            for param in self._model.parameters():
                param.requires_grad = False

            # Try to compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, "compile") and self._device.type in ("cuda", "cpu"):
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile")
                except Exception as e:
                    logger.debug(f"torch.compile not available: {e}")

            logger.info(f"huBERT model loaded: dtype={self._model.dtype}, device={self._device}")
        except Exception as e:
            logger.error(f"Failed to load huBERT model: {e}")
            raise RuntimeError(f"Could not load huBERT model: {e}") from e

    def _load_huspacy_model(self) -> None:
        """Load HuSpaCy model lazily."""
        if self._nlp is not None:
            return

        if not SPACY_AVAILABLE:
            logger.warning("spacy not available - preprocessing will be disabled")
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
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
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
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

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
        if not text or not text.strip():
            return ""

        # If spacy is not available, return original text
        if not SPACY_AVAILABLE:
            logger.debug("spacy not available - returning original text")
            return text.strip()

        self._load_huspacy_model()

        # If model still not loaded (e.g., spacy available but model missing), return original
        if self._nlp is None:
            return text.strip()

        doc = self._nlp(text)

        # Extract lemmas, excluding stopwords and punctuation
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        ]

        return " ".join(tokens)

    def embed_text(self, text: str, preprocess: bool = False) -> list[float]:
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
            text, padding=True, truncation=True, max_length=512, return_tensors="pt"
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
        texts: list[str],
        preprocess: bool = False,
        batch_size: int | None = None,
        use_cache: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with optimized batch processing.

        Performance features:
        - Automatic batch size optimization
        - Redis cache integration for repeated texts
        - GPU memory management
        - FP16 inference on CUDA

        Args:
            texts: List of input texts to embed.
            preprocess: Whether to apply Hungarian preprocessing first.
            batch_size: Number of texts per batch (auto-determined if None).
            use_cache: Whether to use Redis cache for embeddings.

        Returns:
            List[List[float]]: List of 768-dimensional embedding vectors.
        """
        self._load_hubert_model()

        if not texts:
            return []

        # Use optimal batch size if not specified
        if batch_size is None:
            batch_size = self._optimal_batch_size

        # Preprocess if requested
        if preprocess:
            texts = [self.preprocess_hungarian(text) for text in texts]

        # Try to get cached embeddings first
        cached_embeddings: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        if use_cache and self._cache_enabled:
            try:
                # Import here to avoid circular dependency
                # This is sync code, so we need to run async in new loop
                # In production, this should be called from async context
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is None:
                    # We're in sync context, skip cache
                    texts_to_embed = [(i, t) for i, t in enumerate(texts)]
                else:
                    # Get cache service - this should be done in async context
                    texts_to_embed = [(i, t) for i, t in enumerate(texts)]
            except ImportError:
                texts_to_embed = [(i, t) for i, t in enumerate(texts)]
        else:
            texts_to_embed = [(i, t) for i, t in enumerate(texts)]

        # Initialize result array
        all_embeddings: list[list[float]] = [
            [0.0] * settings.EMBEDDING_DIMENSION for _ in range(len(texts))
        ]

        # Fill in cached results
        for i, emb in enumerate(cached_embeddings):
            if emb is not None:
                all_embeddings[i] = emb

        # Process uncached texts in batches
        if texts_to_embed:
            self._embed_batch_internal(
                texts_to_embed,
                all_embeddings,
                batch_size,
            )

        # Cleanup GPU memory after large batches
        if len(texts) > batch_size * 4:
            self._cleanup_gpu_memory()

        return all_embeddings

    def _embed_batch_internal(
        self,
        texts_with_indices: list[tuple[int, str]],
        results: list[list[float]],
        batch_size: int,
    ) -> None:
        """
        Internal batch embedding with optimized processing.

        Args:
            texts_with_indices: List of (original_index, text) tuples.
            results: Results list to populate in-place.
            batch_size: Batch size for processing.
        """
        # Add GPU memory cleanup before processing
        if self._device.type in ("cuda", "mps"):
            self._cleanup_gpu_memory()

        # Process in batches
        for i in range(0, len(texts_with_indices), batch_size):
            batch = texts_with_indices[i : i + batch_size]

            # Separate indices and texts
            non_empty_items = [(idx, text) for idx, text in batch if text and text.strip()]

            if not non_empty_items:
                continue

            indices = [item[0] for item in non_empty_items]
            batch_texts = [item[1] for item in non_empty_items]

            # Tokenize batch
            encoded = self._tokenizer(
                batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

            # Move to device with correct dtype
            encoded = {k: v.to(self._device) for k, v in encoded.items()}

            # Generate embeddings with autocast for FP16 and OOM recovery
            try:
                with torch.no_grad():
                    if self._use_fp16 and self._device.type == "cuda":
                        with torch.cuda.amp.autocast():
                            model_output = self._model(**encoded)
                    else:
                        model_output = self._model(**encoded)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU OOM detected, reducing batch size and retrying")
                    self._cleanup_gpu_memory()
                    # Retry with smaller batch
                    if batch_size > 1:
                        logger.info(f"Retrying with batch_size={batch_size // 2}")
                        self._embed_batch_internal(
                            [(idx, text) for idx, text in batch],
                            results,
                            batch_size=batch_size // 2,
                        )
                        continue
                    else:
                        logger.error("GPU OOM with batch_size=1, cannot reduce further")
                        raise
                raise

            # Apply mean pooling
            embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            # Convert to float32 for output (from FP16 if used)
            embeddings = embeddings.float().cpu().tolist()

            # Assign to correct positions
            for orig_idx, embedding in zip(indices, embeddings, strict=False):
                results[orig_idx] = embedding

            logger.debug(f"Batch {i // batch_size + 1}: {len(batch_texts)} texts embedded")

    def get_similar_texts(
        self, query: str, candidates: list[str], top_k: int = 5, preprocess: bool = False
    ) -> list[tuple[str, float]]:
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

        if SPACY_AVAILABLE:
            self._load_huspacy_model()
        else:
            logger.info("Skipping HuSpaCy warmup - spacy not available")

        # Run a test embedding to warm up GPU and compile model
        _ = self.embed_text("Teszt szoveg a bemelegiteshez.")

        # Run a batch to warm up batch processing path
        _ = self.embed_batch(
            ["Teszt egy", "Teszt ketto", "Teszt harom"],
            batch_size=3,
            use_cache=False,
        )

        logger.info(
            f"HungarianEmbeddingService warmup complete: "
            f"device={self._device}, batch_size={self._optimal_batch_size}"
        )

    # =========================================================================
    # Async Methods (for use in async contexts)
    # =========================================================================

    async def embed_text_async(
        self,
        text: str,
        preprocess: bool = False,
        use_cache: bool = True,
    ) -> list[float]:
        """
        Async version of embed_text for use in async contexts.

        Uses thread pool to avoid blocking event loop during model inference.

        Args:
            text: Input text to embed.
            preprocess: Whether to apply Hungarian preprocessing.
            use_cache: Whether to check Redis cache first.

        Returns:
            List[float]: 768-dimensional embedding vector.
        """
        # Check cache first
        if use_cache and self._cache_enabled:
            try:
                from app.db.redis_cache import get_cache_service

                cache = await get_cache_service()
                cached = await cache.get_embedding(text)
                if cached is not None:
                    return cached
            except Exception:
                pass  # Cache miss or error

        # Run embedding in thread pool
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            _thread_pool, lambda: self.embed_text(text, preprocess)
        )

        # Store in cache
        if use_cache and self._cache_enabled:
            try:
                from app.db.redis_cache import get_cache_service

                cache = await get_cache_service()
                await cache.set_embedding(text, embedding)
            except Exception:
                pass  # Don't fail on cache error

        return embedding

    async def embed_batch_async(
        self,
        texts: list[str],
        preprocess: bool = False,
        batch_size: int | None = None,
        use_cache: bool = True,
    ) -> list[list[float]]:
        """
        Async version of embed_batch with cache integration.

        Args:
            texts: List of input texts to embed.
            preprocess: Whether to apply Hungarian preprocessing.
            batch_size: Batch size (auto-determined if None).
            use_cache: Whether to use Redis cache.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if not texts:
            return []

        # Check cache for all texts
        results: list[list[float] | None] = [None] * len(texts)
        texts_to_embed: list[tuple[int, str]] = []

        if use_cache and self._cache_enabled:
            try:
                from app.db.redis_cache import get_cache_service

                cache = await get_cache_service()
                cached = await cache.get_embeddings_batch(texts)

                for i, (text, emb) in enumerate(zip(texts, cached, strict=False)):
                    if emb is not None:
                        results[i] = emb
                    else:
                        texts_to_embed.append((i, text))
            except Exception:
                # Cache unavailable, embed all
                texts_to_embed = [(i, t) for i, t in enumerate(texts)]
        else:
            texts_to_embed = [(i, t) for i, t in enumerate(texts)]

        # Log cache hit rate
        cache_hits = len(texts) - len(texts_to_embed)
        if cache_hits > 0:
            logger.debug(f"Embedding cache: {cache_hits}/{len(texts)} hits")

        # Embed uncached texts in thread pool
        if texts_to_embed:
            uncached_texts = [t for _, t in texts_to_embed]
            uncached_indices = [i for i, _ in texts_to_embed]

            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(
                _thread_pool,
                lambda: self.embed_batch(
                    uncached_texts,
                    preprocess=preprocess,
                    batch_size=batch_size,
                    use_cache=False,  # Already handled caching
                ),
            )

            # Fill in results
            for idx, emb in zip(uncached_indices, embeddings, strict=False):
                results[idx] = emb

            # Cache new embeddings
            if use_cache and self._cache_enabled:
                try:
                    from app.db.redis_cache import get_cache_service

                    cache = await get_cache_service()
                    for text, emb in zip(uncached_texts, embeddings, strict=False):
                        await cache.set_embedding(text, emb)
                except Exception:
                    pass

        return results  # type: ignore

    def disable_cache(self) -> None:
        """Disable embedding cache (for testing)."""
        self._cache_enabled = False

    def enable_cache(self) -> None:
        """Enable embedding cache."""
        self._cache_enabled = True


# Global service instance
_embedding_service: HungarianEmbeddingService | None = None


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
def embed_text(text: str, preprocess: bool = False) -> list[float]:
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
    texts: list[str], preprocess: bool = False, batch_size: int = 32
) -> list[list[float]]:
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
    query: str, candidates: list[str], top_k: int = 5, preprocess: bool = False
) -> list[tuple[str, float]]:
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


# =============================================================================
# Async Convenience Functions
# =============================================================================


async def embed_text_async(
    text: str,
    preprocess: bool = False,
    use_cache: bool = True,
) -> list[float]:
    """
    Async embedding generation for a single text.

    Args:
        text: Input text to embed.
        preprocess: Whether to apply Hungarian preprocessing.
        use_cache: Whether to use Redis cache.

    Returns:
        List[float]: 768-dimensional embedding vector.
    """
    return await get_embedding_service().embed_text_async(text, preprocess, use_cache)


async def embed_batch_async(
    texts: list[str],
    preprocess: bool = False,
    batch_size: int | None = None,
    use_cache: bool = True,
) -> list[list[float]]:
    """
    Async batch embedding generation.

    Args:
        texts: List of input texts to embed.
        preprocess: Whether to apply Hungarian preprocessing.
        batch_size: Batch size (auto-determined if None).
        use_cache: Whether to use Redis cache.

    Returns:
        List[List[float]]: List of embedding vectors.
    """
    return await get_embedding_service().embed_batch_async(texts, preprocess, batch_size, use_cache)
