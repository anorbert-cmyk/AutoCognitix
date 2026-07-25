"""
Hungarian NLP and Embedding Service.

This module provides Hungarian language text embedding using the huBERT model
and text preprocessing using HuSpaCy.

Inference backends
------------------
The transformer forward pass runs through one of two interchangeable backends;
everything around it (tokenization contract, mask-weighted mean pooling, L2
normalization, 768-dim output) is IDENTICAL in both, which is what keeps them in
the same embedding space as the ~54k vectors already indexed in Qdrant:

- ``onnx``  : ONNX Runtime fp32 on an exported huBERT graph, tokenized directly
  with ``tokenizers.BertWordPieceTokenizer``. This is the PRODUCTION backend.
  ``transformers`` MUST NOT be imported on this path - importing it drags torch
  back in (+367 MB RSS measured) and cancels the entire benefit of ONNX.
- ``torch`` : the original ``transformers`` + torch path. Used for local dev and
  by the offline indexing scripts, which are the anchor of the embedding space.

Measured parity in this project's pinned environment (torch 2.2.0+cpu,
transformers 4.37.2, onnxruntime 1.28.0): cosine >= 0.9999991, max abs element
delta 1.5e-07, identical token IDs on every probe text. No re-indexing is
required to switch backends. See docs/EMBEDDING_ARCHITECTURE_DECISION.md.

There is deliberately NO third "silently return a zero vector" backend. A zero
vector is type-correct, dimension-correct and mathematically meaningless: it
matches nothing in cosine search, so Hungarian semantic search failed silently
for months. A missing backend now raises ``EmbeddingUnavailableError``.

Performance Optimizations:
- Lazy model loading (models loaded on first use, NOT at import/boot)
- GPU memory management with automatic cleanup (torch backend)
- Optimized batch processing with dynamic batching
- Embedding cache integration (versioned cache keys)
- Half-precision (FP16) inference for GPU (torch backend)
- Async-compatible embedding generation
"""

import asyncio
import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.core.exceptions import EmbeddingUnavailableError

# Lazy imports for torch and transformers - NOT installed in the production
# image on purpose (the ONNX backend replaces them there).
try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment,misc]
    AutoTokenizer = None  # type: ignore[assignment,misc]
    TORCH_AVAILABLE = False
    logging.info(
        "torch/transformers not available - the ONNX Runtime backend will be used "
        "for embedding generation (expected in the production image)."
    )

# ONNX Runtime + the standalone Rust tokenizer: the production inference stack.
# Both are torch-free and transformers-free.
try:
    import onnxruntime as ort
    from tokenizers import BertWordPieceTokenizer

    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]
    BertWordPieceTokenizer = None  # type: ignore[assignment,misc]
    ONNX_RUNTIME_AVAILABLE = False

# Optional spacy import - not required for basic embedding functionality
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    SPACY_AVAILABLE = False
    logging.warning("spacy not available - Hungarian preprocessing will be disabled")

logger = logging.getLogger(__name__)

# Tokenizer contract - shared by BOTH backends and by the offline indexer.
# Changing this changes the embedding space and requires a full reindex.
MAX_SEQUENCE_LENGTH = 512

# Redis embedding cache-key version. Bump this whenever the produced vectors
# could change (backend swap, model/revision bump, pooling change) so pre-fix
# entries become unreachable at deploy time with ZERO manual steps - no
# `SCAN`+`DEL` of `embed:*` and no hour-long TTL window in which the fix looks
# broken. "v2" retires the poisoned zero vectors cached by the pre-ONNX build.
EMBEDDING_CACHE_VERSION = "v2"

# Thread pool for heavy CPU/GPU-bound model inference (HuBERT embeddings)
_thread_pool = ThreadPoolExecutor(max_workers=4)

# Separate small pool for lightweight NLP work (e.g. HuSpaCy preprocessing) so
# fast preprocessing calls never queue behind multi-second model inference.
_nlp_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="nlp")


def _mean_pool_l2_numpy(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mask-weighted mean pooling + L2 normalization, in numpy.

    This is a literal transcription of the torch reference path
    (:meth:`HungarianEmbeddingService._mean_pooling` followed by
    ``torch.nn.functional.normalize(..., p=2, dim=1)``) and it is the reason the
    ONNX backend lands in the SAME embedding space as the already-indexed
    vectors: the ONNX graph only replaces the transformer forward pass, pooling
    and normalization stay outside the model and stay identical.

    The ``clamp(min=1e-9)`` on the token counts and the ``ord=2, axis=1``
    normalization are not stylistic choices - they are the identity of the
    embedding space. Do not "simplify" them.

    Args:
        last_hidden: (batch, seq, dim) float32 last_hidden_state.
        attention_mask: (batch, seq) attention mask.

    Returns:
        np.ndarray: (batch, dim) float32 L2-normalized embeddings.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden.astype(np.float32) * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    pooled = summed / counts
    # torch F.normalize uses eps=1e-12 in the denominator clamp.
    norms = np.clip(np.linalg.norm(pooled, ord=2, axis=1, keepdims=True), 1e-12, None)
    return np.asarray(pooled / norms, dtype=np.float32)


class _OnnxEmbeddingBackend:
    """
    ONNX Runtime fp32 forward pass for huBERT - torch-free and transformers-free.

    IMPORTANT: nothing in this class may import or reference ``transformers`` or
    ``torch``. Importing ``transformers`` pulls torch into the process (+367 MB
    RSS measured) and makes the ONNX path WORSE than plain torch, defeating the
    whole architecture. ``tests/unit/test_embedding_backends.py`` enforces this.

    Tokenization uses ``BertWordPieceTokenizer(vocab.txt, lowercase=False)``.
    ``lowercase=False`` is mandatory (the model config says
    ``do_lower_case: false``); getting it wrong degrades embeddings silently.
    It was measured to produce token IDs identical to ``AutoTokenizer``.
    """

    name = "onnx"

    def __init__(self, model_path: str, vocab_path: str, threads: int) -> None:
        if not ONNX_RUNTIME_AVAILABLE:
            raise EmbeddingUnavailableError(
                "onnxruntime/tokenizers are not installed.",
                details={"backend": "onnx"},
            )

        session_options = ort.SessionOptions()
        # Explicit thread caps: ORT otherwise claims every core, and the two
        # gunicorn workers would oversubscribe the container.
        session_options.intra_op_num_threads = max(1, threads)
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.model_path = model_path
        self.vocab_path = vocab_path
        self._session = ort.InferenceSession(
            model_path, session_options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        self._tokenizer = BertWordPieceTokenizer(vocab_path, lowercase=False)
        self._tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        # Pad to the longest sequence in the batch == transformers' padding=True.
        self._tokenizer.enable_padding()

    def encode(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Tokenize a batch exactly like ``padding=True, truncation=True, max_length=512``."""
        encodings = self._tokenizer.encode_batch(texts)
        encoded = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        # Only feed inputs the exported graph actually declares.
        return {k: v for k, v in encoded.items() if k in self._input_names}

    def forward(self, encoded: Dict[str, np.ndarray]) -> np.ndarray:
        """Run the graph and return (batch, seq, dim) last_hidden_state."""
        outputs = self._session.run(["last_hidden_state"], encoded)
        return np.asarray(outputs[0])

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Tokenize -> forward -> shared pooling/normalization -> plain lists."""
        encoded = self.encode(texts)
        last_hidden = self.forward(encoded)
        pooled = _mean_pool_l2_numpy(last_hidden, encoded["attention_mask"])
        return [row.tolist() for row in pooled]


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

    _instance: Optional["HungarianEmbeddingService"] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False

    # Optimal batch sizes by device type
    BATCH_SIZE_GPU = 64
    BATCH_SIZE_CPU = 16
    BATCH_SIZE_MPS = 32

    # Memory threshold for GPU cleanup (bytes)
    GPU_MEMORY_THRESHOLD = 0.8  # 80% utilization triggers cleanup

    def __new__(cls) -> "HungarianEmbeddingService":
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
        self._tokenizer = None
        self._model = None
        self._nlp = None  # Optional spacy.Language
        self._cache_enabled = True
        # Guards lazy model loading so concurrent thread-pool workers never
        # observe a half-initialized model (assigned but not yet .to()/.eval()).
        self._model_load_lock = threading.Lock()
        self._onnx_backend: Optional[_OnnxEmbeddingBackend] = None

        # Decide WHICH backend to use now (cheap: an env read + two stat calls).
        # Actually LOADING it stays lazy - see _load_onnx_backend /
        # _load_hubert_model - so process boot never waits on a ~440 MB model
        # and the Railway healthcheck window is unaffected.
        self._backend_name = self._select_backend_name()

        if self._backend_name == "torch":
            self._device = self._detect_device()
            self._use_fp16 = self._device.type == "cuda"  # FP16 only on CUDA
            self._optimal_batch_size = self._get_optimal_batch_size()
            logger.info(
                f"HungarianEmbeddingService initialized: backend=torch, "
                f"device={self._device}, fp16={self._use_fp16}, "
                f"batch_size={self._optimal_batch_size}"
            )
        elif self._backend_name == "onnx":
            self._device = None
            self._use_fp16 = False
            self._optimal_batch_size = self.BATCH_SIZE_CPU
            logger.info(
                "HungarianEmbeddingService initialized: backend=onnx (ONNX Runtime fp32, "
                "no torch/transformers), model=%s, batch_size=%d",
                settings.HUBERT_ONNX_PATH,
                self._optimal_batch_size,
            )
        else:
            self._device = None
            self._use_fp16 = False
            self._optimal_batch_size = self.BATCH_SIZE_CPU
            logger.error(
                "HungarianEmbeddingService initialized with NO embedding backend "
                "(EMBEDDING_BACKEND=%s, onnxruntime=%s, onnx_model=%s, torch=%s). "
                "Semantic search is DISABLED and embed calls will raise "
                "EmbeddingUnavailableError - they will NOT return a zero vector.",
                settings.EMBEDDING_BACKEND,
                ONNX_RUNTIME_AVAILABLE,
                settings.HUBERT_ONNX_PATH,
                TORCH_AVAILABLE,
            )

    @staticmethod
    def _onnx_artifacts_present() -> bool:
        """True if onnxruntime/tokenizers are installed AND both artifacts exist."""
        if not ONNX_RUNTIME_AVAILABLE:
            return False
        return (
            Path(settings.HUBERT_ONNX_PATH).is_file() and Path(settings.HUBERT_VOCAB_PATH).is_file()
        )

    def _select_backend_name(self) -> Optional[str]:
        """
        Resolve the inference backend from ``settings.EMBEDDING_BACKEND``.

        Order for "auto": ONNX (production image) -> torch (dev/indexer) -> none.

        Returns:
            Optional[str]: "onnx", "torch", or None when nothing is available.
            None is NOT a fallback that produces vectors - it makes every embed
            call raise :class:`EmbeddingUnavailableError`.
        """
        mode = (settings.EMBEDDING_BACKEND or "auto").strip().lower()

        if mode == "disabled":
            return None
        if mode == "onnx":
            return "onnx" if self._onnx_artifacts_present() else None
        if mode == "torch":
            return "torch" if TORCH_AVAILABLE else None

        # "auto" (and any unrecognised value, which we treat as auto but flag)
        if mode != "auto":
            logger.warning("Unknown EMBEDDING_BACKEND=%r - falling back to 'auto' selection.", mode)
        if self._onnx_artifacts_present():
            return "onnx"
        if TORCH_AVAILABLE:
            return "torch"
        return None

    @property
    def backend_name(self) -> Optional[str]:
        """Active inference backend ("onnx", "torch") or None when unavailable."""
        return self._backend_name

    def _require_backend(self) -> str:
        """
        Return the active backend name or raise.

        Raises:
            EmbeddingUnavailableError: when no backend is available. This
                REPLACES the old ``return [0.0] * 768`` silent fallback.
        """
        if self._backend_name is None:
            raise EmbeddingUnavailableError(
                "No embedding backend available (EMBEDDING_BACKEND="
                f"{settings.EMBEDDING_BACKEND!r}, onnxruntime={ONNX_RUNTIME_AVAILABLE}, "
                f"torch={TORCH_AVAILABLE}). Semantic search is DISABLED - "
                "refusing to emit a zero vector.",
                details={
                    "embedding_backend_setting": settings.EMBEDDING_BACKEND,
                    "onnxruntime_available": ONNX_RUNTIME_AVAILABLE,
                    "torch_available": TORCH_AVAILABLE,
                    "onnx_model_path": settings.HUBERT_ONNX_PATH,
                },
            )
        return self._backend_name

    def _load_onnx_backend(self) -> _OnnxEmbeddingBackend:
        """
        Lazily build the ONNX Runtime session (double-checked locking).

        Cold session creation was measured at ~0.9 s; it happens on the first
        embed call, never at import or process boot.
        """
        backend = self._onnx_backend
        if backend is not None:
            return backend

        with self._model_load_lock:
            if self._onnx_backend is not None:
                return self._onnx_backend
            logger.info("Loading ONNX huBERT graph: %s", settings.HUBERT_ONNX_PATH)
            try:
                backend = _OnnxEmbeddingBackend(
                    model_path=settings.HUBERT_ONNX_PATH,
                    vocab_path=settings.HUBERT_VOCAB_PATH,
                    threads=settings.EMBEDDING_ORT_THREADS,
                )
            except EmbeddingUnavailableError:
                raise
            except Exception as e:
                logger.error(f"Failed to load ONNX huBERT graph: {e}")
                raise EmbeddingUnavailableError(
                    f"Could not load the ONNX huBERT graph: {e}",
                    details={"onnx_model_path": settings.HUBERT_ONNX_PATH},
                    original_error=e,
                ) from e
            # Publish only after full construction, like the torch path.
            self._onnx_backend = backend
            logger.info("ONNX huBERT graph loaded (fp32, CPUExecutionProvider)")
            return backend

    def _detect_device(self):
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
        # Fast path: model already fully published, no lock needed.
        if self._model is not None and self._tokenizer is not None:
            return

        # Double-checked locking: serialize concurrent loaders so only one
        # thread performs the load and others wait for it to finish.
        with self._model_load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            logger.info(f"Loading huBERT model: {settings.HUBERT_MODEL}")

            try:
                # Load tokenizer with fast implementation
                tokenizer = AutoTokenizer.from_pretrained(  # nosec B614 B615
                    settings.HUBERT_MODEL,
                    revision=settings.HUBERT_REVISION,
                    use_fast=True,  # Use fast tokenizer implementation
                )

                # Load model with optimizations
                model = AutoModel.from_pretrained(  # nosec B614 B615
                    settings.HUBERT_MODEL,
                    revision=settings.HUBERT_REVISION,
                    torch_dtype=torch.float16 if self._use_fp16 else torch.float32,
                )
                if model is None:
                    raise RuntimeError("Failed to load huBERT model")
                model.to(self._device)
                model.eval()

                # Disable gradient computation for inference
                for param in model.parameters():
                    param.requires_grad = False

                # Try to compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, "compile") and self._device.type in ("cuda", "cpu"):
                    try:
                        model = torch.compile(model, mode="reduce-overhead")
                        logger.info("Model compiled with torch.compile")
                    except Exception as e:
                        logger.debug(f"torch.compile not available: {e}")

                # Publish fully-initialized objects only after setup completes,
                # so fast-path readers never see a partially built model.
                self._tokenizer = tokenizer
                self._model = model

                logger.info(f"huBERT model loaded: dtype={model.dtype}, device={self._device}")
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

    def _mean_pooling(self, model_output, attention_mask):
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

    def embed_text(self, text: str, preprocess: bool = False) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed.
            preprocess: Whether to apply Hungarian preprocessing first.

        Returns:
            List[float]: 768-dimensional, L2-normalized embedding vector.
            The only zero vector this can return is for empty input (see below).

        Raises:
            EmbeddingUnavailableError: If no inference backend is available.
                Deliberately NOT a zero vector - see the module docstring.
        """
        backend = self._require_backend()

        if preprocess:
            text = self.preprocess_hungarian(text)

        if not text or not text.strip():
            # Empty input has no semantic content. This zero vector is defensible
            # on the INDEXING side (an empty record must not get a random point);
            # on the query side callers must not search with it, and
            # QdrantService.search() rejects it via the norm guard.
            return [0.0] * settings.EMBEDDING_DIMENSION

        if backend == "onnx":
            return self._load_onnx_backend().embed([text])[0]

        self._load_hubert_model()

        # Tokenize
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        if self._model is None:
            raise RuntimeError("Model not loaded")
        encoded = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
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
        batch_size: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[List[float]]:
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

        Raises:
            EmbeddingUnavailableError: If no inference backend is available.
                Deliberately NOT a list of zero vectors.
        """
        if not texts:
            return []

        self._require_backend()

        if self._backend_name == "torch":
            self._load_hubert_model()

        # Use optimal batch size if not specified
        if batch_size is None:
            batch_size = self._optimal_batch_size

        # Preprocess if requested
        if preprocess:
            texts = [self.preprocess_hungarian(text) for text in texts]

        # Try to get cached embeddings first
        cached_embeddings: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[Tuple[int, str]] = []

        if use_cache and self._cache_enabled:
            try:
                # asyncio already imported at module level
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
        all_embeddings: List[List[float]] = [
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

        # Cleanup GPU memory after large batches (torch backend only)
        if self._device is not None and len(texts) > batch_size * 4:
            self._cleanup_gpu_memory()

        return all_embeddings

    def _embed_batch_onnx(
        self,
        texts_with_indices: List[Tuple[int, str]],
        results: List[List[float]],
        batch_size: int,
    ) -> None:
        """
        ONNX Runtime batch path - mirrors the torch path's batching semantics.

        Empty/whitespace texts are skipped (they keep their zero-vector slot,
        exactly as in the torch path) so the tokenizer never sees them.
        """
        backend = self._load_onnx_backend()

        for i in range(0, len(texts_with_indices), batch_size):
            batch = texts_with_indices[i : i + batch_size]
            non_empty_items = [(idx, text) for idx, text in batch if text and text.strip()]
            if not non_empty_items:
                continue

            indices = [item[0] for item in non_empty_items]
            batch_texts = [item[1] for item in non_empty_items]

            for orig_idx, embedding in zip(indices, backend.embed(batch_texts)):
                results[orig_idx] = embedding

            logger.debug(f"ONNX batch {i // batch_size + 1}: {len(batch_texts)} texts embedded")

    def _embed_batch_internal(
        self,
        texts_with_indices: List[Tuple[int, str]],
        results: List[List[float]],
        batch_size: int,
    ) -> None:
        """
        Internal batch embedding with optimized processing.

        Dispatches to the active backend; both produce vectors in the same
        embedding space because pooling and L2 normalization are shared.

        Args:
            texts_with_indices: List of (original_index, text) tuples.
            results: Results list to populate in-place.
            batch_size: Batch size for processing.
        """
        if self._require_backend() == "onnx":
            self._embed_batch_onnx(texts_with_indices, results, batch_size)
            return

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
            if self._tokenizer is None:
                raise RuntimeError("Tokenizer not loaded")
            if self._model is None:
                raise RuntimeError("Model not loaded")
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt",
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
            for orig_idx, embedding in zip(indices, embeddings):
                results[orig_idx] = embedding

            logger.debug(f"Batch {i // batch_size + 1}: {len(batch_texts)} texts embedded")

    def get_similar_texts(
        self, query: str, candidates: List[str], top_k: int = 5, preprocess: bool = False
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
    def device(self):
        """Get the current device being used."""
        return self._device

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return settings.EMBEDDING_DIMENSION

    @property
    def is_model_loaded(self) -> bool:
        """Check if an inference backend is loaded and ready."""
        if self._backend_name == "onnx":
            return self._onnx_backend is not None
        return self._model is not None and self._tokenizer is not None

    def warmup(self) -> None:
        """
        Warm up the service by loading all models.

        Call this explicitly (e.g. from a background task) if you want to pay
        the model-load cost up front. It is deliberately NOT called from the
        FastAPI lifespan: the Railway healthcheck has a 100 s budget and the
        boot path must not wait on a ~440 MB graph.
        """
        if self._backend_name is None:
            logger.error(
                "Skipping warmup - no embedding backend available. "
                "Semantic search will raise EmbeddingUnavailableError."
            )
            return

        logger.info("Warming up HungarianEmbeddingService (backend=%s)...", self._backend_name)
        if self._backend_name == "onnx":
            self._load_onnx_backend()
        else:
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
            f"HungarianEmbeddingService warmup complete: backend={self._backend_name}, "
            f"device={self._device}, batch_size={self._optimal_batch_size}"
        )

    # Fixed Hungarian probe sentence for the health self-test. Kept constant so
    # the produced vector is comparable across deploys.
    SELF_TEST_TEXT = "A motor rangat gyorsitaskor."

    def self_test(self) -> Dict[str, Any]:
        """
        Prove the embedding path can actually produce a usable vector.

        This is the third layer of the "never a silent zero vector" guard: the
        health endpoint calls it, so an unavailable or degenerate embedding
        backend is visible from outside the process instead of only showing up
        as mysteriously empty semantic search results.

        Returns:
            Dict[str, Any]: ``status`` is one of:
                - "ok"          - backend produced a unit-norm 768-dim vector
                - "degraded"    - backend answered, but the vector is wrong
                                  (zero norm, wrong dimension, non-finite)
                - "unavailable" - no backend at all / it failed to load
        """
        result: Dict[str, Any] = {
            "backend": self._backend_name,
            "model": settings.HUBERT_MODEL,
            "revision": settings.HUBERT_REVISION,
            "configured_backend": settings.EMBEDDING_BACKEND,
            "dimension": settings.EMBEDDING_DIMENSION,
        }

        try:
            vector = self.embed_text(self.SELF_TEST_TEXT)
        except EmbeddingUnavailableError as e:
            result["status"] = "unavailable"
            result["error"] = e.message
            return result
        except Exception as e:  # pragma: no cover - defensive
            result["status"] = "unavailable"
            result["error"] = f"{type(e).__name__}: {e}"
            return result

        norm = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
        result["self_test_norm"] = round(norm, 6)
        result["self_test_dimension"] = len(vector)

        if (
            len(vector) != settings.EMBEDDING_DIMENSION
            or not np.isfinite(norm)
            or abs(norm - 1.0) > 1e-4
        ):
            result["status"] = "degraded"
            result["error"] = (
                f"Self-test vector is not unit-length 768-dim (dim={len(vector)}, norm={norm})."
            )
            return result

        result["status"] = "ok"
        return result

    # =========================================================================
    # Async Methods (for use in async contexts)
    # =========================================================================

    def _cache_namespace(self, text: str) -> str:
        """
        Namespace the Redis embedding cache key.

        ``redis_cache._embedding_cache_key`` already salts with
        ``HUBERT_MODEL@HUBERT_REVISION``; this adds the two things it cannot
        know about:

        - ``EMBEDDING_CACHE_VERSION`` - bumped whenever the produced vectors
          could change. This is what retires the poisoned ``[0.0] * 768``
          entries the pre-ONNX production build wrote: they simply become
          unreachable the moment the new image boots. No `SCAN`+`DEL` of
          ``embed:*``, no operator step, no hour-long window where the fix looks
          like it did not work.
        - the active backend name, so a torch-produced and an ONNX-produced
          vector can never be served for one another. (They were measured
          equivalent to cos >= 0.9999991, so this is belt-and-braces - but it is
          one string concat, and it makes a backend swap self-invalidating.)

        Args:
            text: The raw text the caller asked to embed.

        Returns:
            str: The namespaced key material handed to the cache service.
        """
        return f"{EMBEDDING_CACHE_VERSION}|{self._backend_name or 'none'}|{text}"

    async def embed_text_async(
        self,
        text: str,
        preprocess: bool = False,
        use_cache: bool = True,
    ) -> List[float]:
        """
        Async version of embed_text for use in async contexts.

        Uses thread pool to avoid blocking event loop during model inference.

        Args:
            text: Input text to embed.
            preprocess: Whether to apply Hungarian preprocessing.
            use_cache: Whether to check Redis cache first.

        Returns:
            List[float]: 768-dimensional embedding vector.

        Raises:
            EmbeddingUnavailableError: If no inference backend is available.
        """
        cache_key_text = self._cache_namespace(text)

        # Check cache first
        if use_cache and self._cache_enabled:
            try:
                from app.db.redis_cache import get_cache_service

                cache = await get_cache_service()
                cached = await cache.get_embedding(cache_key_text)
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
                await cache.set_embedding(cache_key_text, embedding)
            except Exception:
                pass  # Don't fail on cache error

        return embedding

    async def embed_batch_async(
        self,
        texts: List[str],
        preprocess: bool = False,
        batch_size: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[Optional[List[float]]]:
        """
        Async version of embed_batch with cache integration.

        Args:
            texts: List of input texts to embed.
            preprocess: Whether to apply Hungarian preprocessing.
            batch_size: Batch size (auto-determined if None).
            use_cache: Whether to use Redis cache.

        Returns:
            List[Optional[List[float]]]: List of embedding vectors (None for failed entries).
        """
        if not texts:
            return []

        # Check cache for all texts
        results: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[Tuple[int, str]] = []

        if use_cache and self._cache_enabled:
            try:
                from app.db.redis_cache import get_cache_service

                cache = await get_cache_service()
                cached = await cache.get_embeddings_batch([self._cache_namespace(t) for t in texts])

                for i, (text, emb) in enumerate(zip(texts, cached)):
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
            for idx, emb in zip(uncached_indices, embeddings):
                results[idx] = emb

            # Cache new embeddings
            if use_cache and self._cache_enabled:
                try:
                    from app.db.redis_cache import get_cache_service

                    cache = await get_cache_service()
                    # Fire the cache writes concurrently instead of one serial
                    # round-trip per text (the client exposes no batched setter).
                    await asyncio.gather(
                        *(
                            cache.set_embedding(self._cache_namespace(text), emb)
                            for text, emb in zip(uncached_texts, embeddings)
                        )
                    )
                except Exception:
                    pass

        return results

    def disable_cache(self) -> None:
        """Disable embedding cache (for testing)."""
        self._cache_enabled = False

    def enable_cache(self) -> None:
        """Enable embedding cache."""
        self._cache_enabled = True


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


def embedding_self_test() -> Dict[str, Any]:
    """
    Run the embedding backend self-test (used by ``/health/detailed``).

    Returns:
        Dict[str, Any]: See :meth:`HungarianEmbeddingService.self_test`.
    """
    return get_embedding_service().self_test()


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
    texts: List[str], preprocess: bool = False, batch_size: int = 32
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


def shutdown_thread_pools() -> None:
    """
    Shut down both module-level thread pools (inference + NLP).

    Call this at application shutdown. After shutdown, async wrappers that
    submit work to these pools will raise RuntimeError.
    """
    try:
        _thread_pool.shutdown(wait=True)
    finally:
        # The NLP pool must stop even if the inference pool shutdown raises.
        _nlp_pool.shutdown(wait=True)
    logger.info("Embedding service thread pools shut down (inference + nlp)")


def get_similar_texts(
    query: str, candidates: List[str], top_k: int = 5, preprocess: bool = False
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


# =============================================================================
# Async Convenience Functions
# =============================================================================


async def preprocess_hungarian_async(text: str) -> str:
    """
    Async Hungarian text preprocessing.

    Runs in the dedicated lightweight NLP pool so it never queues behind
    multi-second HuBERT inference in the main thread pool.

    Args:
        text: Input text to preprocess.

    Returns:
        str: Preprocessed text with lemmatized tokens.

    Raises:
        RuntimeError: If the NLP pool has already been shut down.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_nlp_pool, preprocess_hungarian, text)


async def embed_text_async(
    text: str,
    preprocess: bool = False,
    use_cache: bool = True,
) -> List[float]:
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
    texts: List[str],
    preprocess: bool = False,
    batch_size: Optional[int] = None,
    use_cache: bool = True,
) -> List[Optional[List[float]]]:
    """
    Async batch embedding generation.

    Args:
        texts: List of input texts to embed.
        preprocess: Whether to apply Hungarian preprocessing.
        batch_size: Batch size (auto-determined if None).
        use_cache: Whether to use Redis cache.

    Returns:
        List of embedding vectors (None for failed embeddings).
    """
    return await get_embedding_service().embed_batch_async(texts, preprocess, batch_size, use_cache)
