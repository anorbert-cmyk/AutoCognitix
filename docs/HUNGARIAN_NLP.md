# Hungarian NLP Documentation

## Overview

AutoCognitix provides native Hungarian language support for vehicle diagnostics through specialized NLP (Natural Language Processing) components. This document explains the Hungarian language processing pipeline and how to effectively use it.

---

## Architecture

```
Hungarian Text Input
        |
        v
+-------------------+
|    HuSpaCy        |  (Preprocessing: lemmatization, stopword removal)
|  hu_core_news_lg  |
+-------------------+
        |
        v
+-------------------+
|     huBERT        |  (Embedding: 768-dim vectors)
| SZTAKI-HLT model  |
+-------------------+
        |
        v
+-------------------+
|     Qdrant        |  (Vector search: semantic similarity)
+-------------------+
        |
        v
+-------------------+
|    LLM (RAG)      |  (Response generation)
+-------------------+
        |
        v
Hungarian Diagnosis Output
```

---

## Components

### 1. huBERT Embedding Model

**Model:** `SZTAKI-HLT/hubert-base-cc`

The huBERT (Hungarian BERT) model is a transformer-based language model trained specifically for Hungarian text. It generates 768-dimensional vector representations of text that capture semantic meaning.

**Key Features:**
- Native Hungarian language understanding
- 768-dimensional embeddings
- Support for automotive and technical terminology
- Handles Hungarian accented characters (a, e, i, o, o, u, u, o, u)

**Configuration:**
```python
# backend/app/core/config.py
HUBERT_MODEL = "SZTAKI-HLT/hubert-base-cc"
EMBEDDING_DIMENSION = 768
```

### 2. HuSpaCy Preprocessing

**Model:** `hu_core_news_lg`

HuSpaCy provides Hungarian-specific text preprocessing including:
- Lemmatization (reduces words to base forms)
- Stopword removal
- Part-of-speech tagging
- Named entity recognition

**Example:**
```python
# Input
"A motor nehezen indul hidegben es egyenetlenul jar alapjaraton."

# After preprocessing (lemmatization)
"motor nehezen indul hideg egyenetlenul jar alapjarat"
```

---

## Embedding Service

The `HungarianEmbeddingService` is a singleton service that provides text embedding functionality.

### Location

```
backend/app/services/embedding_service.py
```

### Key Methods

#### `embed_text(text, preprocess=False)`

Generate embedding for a single text.

```python
from app.services.embedding_service import embed_text

# Without preprocessing
embedding = embed_text("A motor nehezen indul")
# Returns: [0.0234, -0.1567, ...] (768 floats)

# With Hungarian preprocessing
embedding = embed_text("A motor nehezen indul", preprocess=True)
```

#### `embed_batch(texts, preprocess=False)`

Generate embeddings for multiple texts with batch processing.

```python
from app.services.embedding_service import embed_batch

texts = [
    "Motor teljesitmenyvesztese",
    "Egyenetlen alapjarat",
    "Magas uzemanyag-fogyasztas"
]
embeddings = embed_batch(texts, preprocess=True)
# Returns: List of 768-dim vectors
```

#### `preprocess_hungarian(text)`

Apply Hungarian NLP preprocessing.

```python
from app.services.embedding_service import preprocess_hungarian

processed = preprocess_hungarian("A jarmu motorja egyenetlenul jar.")
# Returns: "jarmu motor egyenetlenul jar"
```

#### `get_similar_texts(query, candidates, top_k=5)`

Find most similar texts using cosine similarity.

```python
from app.services.embedding_service import get_similar_texts

query = "motor nehezen indul"
candidates = [
    "Hideginditas problemak",
    "Indito motor hiba",
    "Fekhiba",
    "Motor berregese indulaskor"
]

results = get_similar_texts(query, candidates, top_k=3)
# Returns: [("Hideginditas problemak", 0.89), ("Indito motor hiba", 0.85), ...]
```

### Async Methods

For async contexts (FastAPI endpoints), use the async variants:

```python
from app.services.embedding_service import embed_text_async, embed_batch_async

# In async function
embedding = await embed_text_async("A motor nehezen indul", preprocess=True)
embeddings = await embed_batch_async(texts, preprocess=True)
```

---

## Performance Optimizations

The embedding service includes several performance optimizations:

### 1. Device Auto-Detection

Automatically uses the best available hardware:
- **CUDA GPU**: For NVIDIA GPUs (fastest)
- **MPS**: For Apple Silicon (M1/M2/M3)
- **CPU**: Fallback with multi-threading

### 2. Lazy Model Loading

Models are loaded on first use, not at startup:

```python
# Model loads only when first called
embedding = embed_text("test")
```

### 3. FP16 Inference

Half-precision (FP16) is used on CUDA GPUs for 2x speedup.

### 4. Dynamic Batch Sizing

Batch sizes are automatically optimized based on available memory:
- GPU > 8GB: batch_size = 128
- GPU > 4GB: batch_size = 64
- GPU < 4GB: batch_size = 32
- MPS: batch_size = 32
- CPU: batch_size = 16

### 5. Redis Caching

Embeddings are cached in Redis to avoid recomputation:
- Cache TTL: 24 hours
- Automatic cache invalidation

---

## API Usage

### Symptom Text Input

When using the `/api/v1/diagnosis/analyze` endpoint, provide symptoms in Hungarian:

```json
{
  "vehicle_make": "Volkswagen",
  "vehicle_model": "Golf",
  "vehicle_year": 2018,
  "dtc_codes": ["P0101"],
  "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, es a fogyasztas megnott. Az uzemanyag szag is erezheto."
}
```

### DTC Search in Hungarian

The `/api/v1/dtc/search` endpoint supports Hungarian queries:

```bash
# Search by Hungarian symptoms
curl "http://localhost:8000/api/v1/dtc/search?q=motor%20nehezen%20indul&use_semantic=true"

# Search by Hungarian description
curl "http://localhost:8000/api/v1/dtc/search?q=levegotomeg%20mero%20hiba"
```

---

## Supported Hungarian Automotive Terminology

The system is trained to understand Hungarian automotive terms:

### Engine/Motor Terms
| Hungarian | English |
|-----------|---------|
| motor | engine |
| hengersor | cylinder bank |
| fojtoszelep | throttle |
| befecskendezés | injection |
| turbofeltoeltes | turbocharging |
| egeskamra | combustion chamber |
| duzsniafojtó | injector |

### Symptom Terms
| Hungarian | English |
|-----------|---------|
| nehezen indul | hard start |
| egyenetlen alapjarat | rough idle |
| teljesitmenyvesztes | power loss |
| magas fogyasztas | high consumption |
| berreges | vibration |
| kopogás | knocking |
| fustoles | smoking |

### Component Terms
| Hungarian | English |
|-----------|---------|
| levegotomeg-mero | MAF sensor |
| oxigenszenzor | oxygen sensor |
| gyertya | spark plug |
| injektor | injector |
| szenzor | sensor |
| vezetékcs | wiring |
| csatlakozo | connector |

---

## Best Practices

### 1. Write Clear Symptoms

Provide detailed symptom descriptions in natural Hungarian:

**Good:**
```
"A motor hidegindításkor nehezen indul, kb. 3-4 próbálkozás kell.
Miután beindul, egyenetlenül jár alapjáraton, és néha le is áll."
```

**Avoid:**
```
"motor rossz"
```

### 2. Include Context

Add relevant context for better diagnosis:

```json
{
  "symptoms": "A motor nehezen indul hidegben, különösen -5 fok alatt.",
  "additional_context": "A problema tavaly december ota jelentkezik. Uj akkumulátor van benne."
}
```

### 3. Combine DTCs with Symptoms

Provide both DTC codes and symptoms when available:

```json
{
  "dtc_codes": ["P0101", "P0171"],
  "symptoms": "A motor teljesítménye csökkent, és a fogyasztás megnőtt."
}
```

### 4. Use Standard Hungarian Characters

The system handles Hungarian accented characters properly:

```
Correct: "A motor üzemanyag-fogyasztása megnőtt"
Also OK: "A motor uzemanyag-fogyasztasa megnott"
```

---

## Troubleshooting

### Model Loading Errors

If the huBERT model fails to load:

```bash
# Check if model is downloaded
python -c "from transformers import AutoModel; AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')"

# Clear cache and re-download
rm -rf ~/.cache/huggingface/transformers/*hubert*
```

### HuSpaCy Installation

If HuSpaCy preprocessing is disabled:

```bash
# Install spacy
pip install spacy>=3.4.0

# Download Hungarian model
python -m spacy download hu_core_news_lg
```

### CUDA Out of Memory

If you encounter CUDA memory errors:

```python
# Reduce batch size in config
BATCH_SIZE_GPU = 32  # Lower from 64

# Or use CPU fallback
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

### Slow Embedding Generation

For better performance:

1. Enable GPU if available
2. Use batch processing for multiple texts
3. Enable Redis caching
4. Use preprocessing only when needed

---

## Configuration Reference

### Environment Variables

```bash
# Model settings
HUBERT_MODEL=SZTAKI-HLT/hubert-base-cc
HUSPACY_MODEL=hu_core_news_lg
EMBEDDING_DIMENSION=768

# Performance tuning
EMBEDDING_BATCH_SIZE=64
EMBEDDING_CACHE_TTL=86400

# Redis cache
REDIS_URL=redis://localhost:6379/0
```

### Service Configuration

```python
# backend/app/services/embedding_service.py
class HungarianEmbeddingService:
    BATCH_SIZE_GPU = 64
    BATCH_SIZE_CPU = 16
    BATCH_SIZE_MPS = 32
    GPU_MEMORY_THRESHOLD = 0.8  # 80%
```

---

## Integration Examples

### Custom Embedding Application

```python
from app.services.embedding_service import get_embedding_service

# Get singleton service
service = get_embedding_service()

# Warmup (load models)
service.warmup()

# Generate embeddings
symptoms = [
    "Motor berreges indulaskor",
    "Egyenetlen alapjarat hidegben",
    "Fekpedal erzekeny"
]

embeddings = service.embed_batch(symptoms, preprocess=True)

# Use embeddings for similarity search, clustering, etc.
```

### Vector Database Integration

```python
from app.services.embedding_service import embed_text
from app.db.qdrant_client import qdrant_client

# Index new DTC description
description = "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba"
embedding = embed_text(description, preprocess=True)

await qdrant_client.upsert_dtc(
    code="P0101",
    embedding=embedding,
    metadata={"description_hu": description}
)

# Search by symptoms
query = "motor teljesitmeny csokkenes"
query_embedding = embed_text(query, preprocess=True)

results = await qdrant_client.search_dtc(
    query_vector=query_embedding,
    limit=10
)
```

---

## Further Resources

- [huBERT Model Card](https://huggingface.co/SZTAKI-HLT/hubert-base-cc)
- [HuSpaCy Documentation](https://huggingface.co/huspacy)
- [Qdrant Vector Database](https://qdrant.tech/documentation/)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
