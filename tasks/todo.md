# Comprehensive Error Handling (2026-02-05)

## Status: COMPLETED

### Summary
Implemented comprehensive error handling across AutoCognitix backend and frontend with Hungarian error messages, structured JSON logging, and proper retry logic.

### Backend Changes

#### 1. Custom Exception Classes (`backend/app/core/exceptions.py`)
- **ErrorCode Enum**: 50+ standardized error codes (ERR_1xxx to ERR_5xxx)
- **Hungarian Messages**: All error messages translated to Hungarian
- **Exception Hierarchy**:
  - `AutoCognitixException` - Base class
  - `ValidationException` / `DTCValidationException` / `VINValidationException`
  - `NotFoundException` / `VehicleNotFoundException`
  - `DatabaseException` / `PostgresException` / `Neo4jException` / `QdrantException` / `RedisException`
  - `ExternalAPIException` / `NHTSAException` / `LLMException`
  - `DiagnosisException` / `EmbeddingException` / `RAGException`
  - `AuthenticationException` / `InvalidCredentialsException` / `TokenExpiredException`
  - `RateLimitException`

#### 2. Global Exception Handler (`backend/app/core/error_handlers.py`)
- **RequestContextMiddleware**: Adds request ID to all requests
- **Structured Error Responses**: JSON format with code, message, message_hu, details, request_id
- **Exception Handlers**:
  - `autocognitix_exception_handler` - Custom exceptions
  - `validation_exception_handler` - Pydantic validation errors
  - `sqlalchemy_exception_handler` - Database errors
  - `neo4j_exception_handler` - Neo4j graph database errors
  - `qdrant_exception_handler` - Qdrant vector database errors
  - `httpx_exception_handler` - External HTTP call errors
  - `generic_exception_handler` - Fallback for unhandled errors

#### 3. Enhanced Logging (`backend/app/core/logging.py`)
- **StructuredJsonFormatter**: JSON logs with request correlation
- **Context Variables**: `request_id_var`, `user_id_var`, `correlation_id_var`
- **RequestLoggingMiddleware**: Logs request start/complete with timing
- **PerformanceLogger**: Context manager for operation timing
- **Helper Functions**: `log_database_operation()`, `log_external_api_call()`

#### 4. Retry Utilities (`backend/app/core/retry.py`)
- **RetryConfig**: Configurable retry behavior
- **Exponential Backoff**: With jitter for rate limit handling
- **Decorators**: `@retry_async()`, `@retry_sync()`
- **Context Manager**: `RetryContext` for manual control
- **Pre-configured**: `DEFAULT_CONFIG`, `NHTSA_CONFIG`, `LLM_CONFIG`

#### 5. Database Error Handling
- **PostgreSQL** (`backend/app/db/postgres/session.py`):
  - Connection error detection (OperationalError)
  - Integrity error handling (IntegrityError)
  - Timeout detection (TimeoutError)
  - `check_database_connection()`, `get_database_info()`

- **Qdrant** (`backend/app/db/qdrant_client.py`):
  - Connection error handling
  - Search error handling
  - Collection creation error handling

### Frontend Changes

#### 1. Error Boundary (`frontend/src/components/ErrorBoundary.tsx`)
- **React Error Boundary**: Catches rendering errors
- **Hungarian UI**: Error messages in Hungarian
- **Actions**: Retry, Go Home, Reload
- **HOC**: `withErrorBoundary()` for easy wrapping

#### 2. Error State Component (`frontend/src/components/ui/ErrorState.tsx`)
- **Error Type Detection**: network, server, not_found, unauthorized, forbidden, validation, timeout, rate_limit, generic
- **Customizable Display**: title, message, actions
- **Compact Mode**: For inline display
- **Helper Components**: `InlineError`, `FieldError`

#### 3. Enhanced Error Message (`frontend/src/components/ui/ErrorMessage.tsx`)
- **Auto Error Detection**: From ApiError status
- **Type-specific Icons**: WifiOff, ServerOff, Clock, AlertCircle
- **Compact Mode**: For inline display
- **Hungarian Messages**: Contextual hints

#### 4. Error Hooks (`frontend/src/hooks/useErrorHandler.ts`)
- **useErrorHandler**: Error state management
- **useApiMutation**: Simplified mutation with error handling
- **Utility Functions**: `isNetworkError()`, `isRecoverableError()`, `getErrorMessage()`, `getErrorCode()`

#### 5. Query Client Configuration (`frontend/src/main.tsx`)
- **Custom Retry Logic**: Based on error type
- **Exponential Backoff**: For retries
- **Global Error Handler**: For logging

### Files Created/Modified

| File | Status | Description |
|------|--------|-------------|
| `backend/app/core/exceptions.py` | Created | Custom exception hierarchy |
| `backend/app/core/error_handlers.py` | Created | Global exception handlers |
| `backend/app/core/retry.py` | Created | Retry utilities |
| `backend/app/core/__init__.py` | Modified | Export all core modules |
| `backend/app/core/logging.py` | Enhanced | Structured JSON logging |
| `backend/app/db/postgres/session.py` | Modified | Database error handling |
| `backend/app/db/qdrant_client.py` | Modified | Qdrant error handling |
| `backend/app/api/v1/endpoints/diagnosis.py` | Modified | Use custom exceptions |
| `backend/app/main.py` | Modified | Register error handlers |
| `frontend/src/components/ErrorBoundary.tsx` | Created | React error boundary |
| `frontend/src/components/ui/ErrorState.tsx` | Created | Error state display |
| `frontend/src/components/ui/ErrorMessage.tsx` | Enhanced | Better error display |
| `frontend/src/hooks/useErrorHandler.ts` | Created | Error handling hooks |
| `frontend/src/hooks/index.ts` | Created | Hook exports |
| `frontend/src/components/ui/index.ts` | Modified | Component exports |
| `frontend/src/App.tsx` | Modified | Wrap with ErrorBoundary |
| `frontend/src/main.tsx` | Modified | Query client error handling |

### Error Response Format

```json
{
  "error": {
    "code": "ERR_4001",
    "message": "Invalid DTC code format",
    "message_hu": "Ervenytelen DTC hibakod formatum.",
    "details": {
      "invalid_codes": ["X1234"]
    },
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

### Usage Examples

#### Backend
```python
from app.core.exceptions import DTCValidationException, DiagnosisException
from app.core.retry import retry_async, LLM_CONFIG

# Raise custom exception
raise DTCValidationException(
    message="Invalid DTC code format",
    invalid_codes=["X1234"],
)

# Use retry decorator
@retry_async(config=LLM_CONFIG)
async def call_llm(prompt: str) -> str:
    ...
```

#### Frontend
```tsx
import ErrorBoundary from './components/ErrorBoundary'
import { ErrorState } from './components/ui'
import { useErrorHandler } from './hooks'

// Wrap with error boundary
<ErrorBoundary>
  <App />
</ErrorBoundary>

// Display error state
<ErrorState error={error} onRetry={refetch} />

// Use error handler hook
const { error, handleError, retry } = useErrorHandler({ onRetry: refetch })
```

---

# Comprehensive Vehicle Database Schema (2026-02-05)

## Status: COMPLETED

## Summary
Created comprehensive vehicle database schema with 50+ makes, 60+ models, 50+ engines, 40+ platforms, and DTC frequency tracking.

## Deliverables

### 1. PostgreSQL Migration
- **File**: `/backend/alembic/versions/005_vehicle_comprehensive_schema.py`
- **Tables Created**:
  - `vehicle_engines` - Engine specifications with displacement, power, fuel type
  - `vehicle_platforms` - Shared platforms across makes (e.g., MQB, TNGA, CLAR)
  - `vehicle_model_engines` - Many-to-many model/engine relationships
  - `vehicle_dtc_frequency` - Common DTC codes per vehicle/engine
  - `vehicle_tsb` - Technical Service Bulletins
  - Added `platform_id` FK to `vehicle_models`

### 2. SQLAlchemy Models Updated
- **File**: `/backend/app/db/postgres/models.py`
- **New Models**: `VehicleEngine`, `VehiclePlatform`, `VehicleModelEngine`, `VehicleDTCFrequency`, `VehicleTSB`
- **Updated Models**: `VehicleMake`, `VehicleModel` with new relationships

### 3. Neo4j Models Updated
- **File**: `/backend/app/db/neo4j_models.py`
- **New Nodes**: `EngineNode`, `PlatformNode`
- **New Relationships**: `UsesEngineRel`, `RequiresRepairRel`, `SharesPlatformRel`
- **New Functions**: `get_vehicle_common_issues()`, `get_engine_common_issues()`, `find_similar_vehicles()`

### 4. Vehicle Data Seed Script
- **File**: `/scripts/seed_vehicles.py`
- **Data Included**:
  - **51 Makes**: 32 European, 14 Asian, 5 American
  - **42 Platforms**: MQB, MLB, CLAR, TNGA, E-GMP, etc.
  - **51 Engines**: EA888, B58, M264, EcoBoost, etc.
  - **63 Models**: Golf, 3 Series, C-Class, Corolla, etc.
  - **17 DTC Frequencies**: Common issues per vehicle/engine

## Vehicle Makes Coverage

| Region | Count | Makes |
|--------|-------|-------|
| European | 32 | VW, BMW, Mercedes, Audi, Porsche, Skoda, SEAT, Cupra, Renault, Peugeot, Citroen, Opel, Fiat, Alfa Romeo, Volvo, Jaguar, Land Rover, Mini, Dacia, etc. |
| Asian | 14 | Toyota, Lexus, Honda, Acura, Nissan, Mazda, Subaru, Mitsubishi, Hyundai, Kia, Genesis, BYD, etc. |
| American | 5 | Ford, Lincoln, Chevrolet, GMC, Dodge, Jeep, Ram, Tesla, etc. |

## Usage

```bash
# Seed all vehicle data
python scripts/seed_vehicles.py --all

# Seed specific data types
python scripts/seed_vehicles.py --makes
python scripts/seed_vehicles.py --engines
python scripts/seed_vehicles.py --platforms
python scripts/seed_vehicles.py --models
python scripts/seed_vehicles.py --dtc-freq
python scripts/seed_vehicles.py --neo4j
```

## Run Migration

```bash
cd backend
alembic upgrade head
```

---

# API Endpoint Testing (2026-02-05)

## Status: COMPLETED

## Fixes Applied
1. **Python 3.9 Compatibility** - Fixed `str | Any` type hints to `Union[str, Any]` in:
   - `/backend/app/core/security.py`
   - `/backend/app/db/postgres/repositories.py`

2. **NHTSA Complaints odinumber Type** - Fixed int to string conversion in:
   - `/backend/app/services/nhtsa_service.py` (line 607-608)

## API Endpoint Test Results

### Working Endpoints (All Tested on port 8001)

| Endpoint | Method | Status | Notes |
|----------|--------|--------|-------|
| `/health` | GET | OK | Returns health status |
| `/api/v1/dtc/search?q=P0` | GET | OK | Returns matching DTC codes |
| `/api/v1/dtc/{code}` | GET | OK | Returns 404 if not in DB |
| `/api/v1/vehicles/makes` | GET | OK | Returns 20 static makes |
| `/api/v1/vehicles/models/{make_id}` | GET | OK | Returns models for VW |
| `/api/v1/vehicles/years` | GET | OK | Returns years 1980-2027 |
| `/api/v1/vehicles/{make}/{model}/{year}/recalls` | GET | OK | Fetches from NHTSA |
| `/api/v1/vehicles/{make}/{model}/{year}/complaints` | GET | OK | Fixed odinumber type |
| `/api/v1/vehicles/decode-vin` | POST | OK | Validates VIN checksum |
| `/api/v1/diagnosis/analyze` | POST | OK | Full diagnosis with NHTSA data |
| `/api/v1/auth/login` | POST | OK | Returns JWT tokens |
| `/api/v1/metrics` | GET | OK | Prometheus metrics |
| `/api/v1/docs` | GET | OK | Swagger UI available |

### Sample Successful Responses

**Diagnosis Response** - Includes:
- DTC code analysis with Hungarian descriptions
- NHTSA recall data integration
- Repair recommendations
- Confidence scores

**Complaints Response** - Returns actual NHTSA complaint data

### Notes
- DTC search returns results from in-memory data (dtc_definitions.py)
- Database integration for DTC codes requires seed data import
- VIN decode validates checksum (9th position)

---

# Current Task: Railway Deployment

## Status: ✅ COMPLETED

## Completed
- [x] Create Railway project
- [x] Add PostgreSQL database
- [x] Add Redis database
- [x] Connect GitHub repo
- [x] Configure backend service
- [x] Fix all backend startup issues
- [x] Deploy backend successfully
- [x] Add frontend service
- [x] Fix TypeScript build errors (tsconfig.json, vite-env.d.ts)
- [x] Fix TailwindCSS build errors (shadcn/ui color classes)
- [x] Deploy frontend successfully
- [x] Verify frontend is accessible and working
- [x] Update backend CORS with frontend URL
- [x] Document deployment URLs

## Deployment URLs

| Service | URL | Status |
|---------|-----|--------|
| **Frontend** | https://remarkable-beauty-production-8000.up.railway.app | ✅ Online |
| **Backend API** | https://autocognitix-production.up.railway.app | ✅ Online |
| **API Docs** | https://autocognitix-production.up.railway.app/docs | ✅ Available |
| **Health Check** | https://autocognitix-production.up.railway.app/health | ✅ Available |

## Railway Project
- Project: virtuous-harmony
- Environment: production
- Region: europe-west4-drams3a

## Services
1. **AutoCognitix** (Backend) - Dockerfile build
2. **remarkable-beauty** (Frontend) - Nixpacks build
3. **PostgreSQL** - Railway managed
4. **Redis** - Railway managed

## Next Steps
- [ ] Configure Neo4j Aura connection
- [ ] Configure Qdrant Cloud connection
- [ ] Add Anthropic/OpenAI API key for AI features
- [ ] Test full diagnosis flow

---

# DTC Data Import Scripts

## Status: ✅ CREATED

## Available Scripts

| Script | Source | Expected Output |
|--------|--------|-----------------|
| `import_obd_codes.py` | mytrile/obd-trouble-codes | ~11,000+ DTC codes |
| `scrape_klavkarr.py` | klavkarr.com | ~11,000+ DTC codes |
| `import_obdb.py` | OBDb GitHub (742 repos) | 738+ vehicles, manufacturer-specific DTCs |
| `translate_to_hungarian.py` | DeepSeek API | Hungarian translations |
| `index_qdrant.py` | Local data | Vector embeddings |

## Script Usage

### 1. Import from mytrile (Recommended First)
```bash
# Download and import to all databases
python scripts/import_obd_codes.py --all

# Just download to cache
python scripts/import_obd_codes.py --download
```

### 2. Scrape Klavkarr (Supplementary)
```bash
# Scrape and import
python scripts/scrape_klavkarr.py --all

# Merge with existing codes
python scripts/scrape_klavkarr.py --scrape --merge
```

### 3. Import OBDb Vehicle Data
```bash
# List available repos
python scripts/import_obdb.py --list

# Download and import (limited to 200 repos)
python scripts/import_obdb.py --all --max-repos 200
```

### 4. Translate to Hungarian
```bash
# Set API key first
export DEEPSEEK_API_KEY="your-key-here"

# Translate pending codes
python scripts/translate_to_hungarian.py --translate --limit 100

# Apply translations to databases
python scripts/translate_to_hungarian.py --apply --update-db
```

### 5. Index to Qdrant (Basic)
```bash
# Index DTC codes and symptoms
python scripts/index_qdrant.py --all --recreate
```

### 6. Full Qdrant Reindexing (Comprehensive)
```bash
# Index ALL data into 4 collections with Hungarian huBERT embeddings
python scripts/index_qdrant_full.py --all --recreate --verify

# Individual collection indexing
python scripts/index_qdrant_full.py --dtc              # 3,579+ DTC codes
python scripts/index_qdrant_full.py --symptoms         # Symptoms from DTC + Neo4j
python scripts/index_qdrant_full.py --components       # Vehicle components
python scripts/index_qdrant_full.py --repairs          # Repair procedures

# Verify semantic search works
python scripts/index_qdrant_full.py --verify
```

#### Qdrant Collections Created
| Collection | Content | Dimension | Count |
|------------|---------|-----------|-------|
| `dtc_embeddings_hu` | DTC codes (Hungarian) | 768-dim | 3,579+ |
| `symptom_embeddings_hu` | Vehicle symptoms | 768-dim | 200+ |
| `component_embeddings_hu` | Vehicle components | 768-dim | 30+ |
| `repair_embeddings_hu` | Repair procedures | 768-dim | 24+ |

## Environment Variables Needed

```env
# For DeepSeek translation
DEEPSEEK_API_KEY=your-deepseek-api-key

# For GitHub API (optional, for higher rate limits)
GITHUB_TOKEN=your-github-token
```

## Execution Order

1. `import_obd_codes.py` - Get base 11,000+ codes
2. `scrape_klavkarr.py --merge` - Supplement with additional codes
3. `import_obdb.py` - Add vehicle-specific data
4. `translate_to_hungarian.py` - Translate to Hungarian
5. `index_qdrant.py` - Index for semantic search

---

# Hungarian DTC Translation Progress

## Status: IN PROGRESS (2026-02-05)

### Current Statistics
| Database | Total Codes | Translated | Percentage |
|----------|-------------|------------|------------|
| PostgreSQL | 3,579 | 923 | 25.8% |
| Neo4j | 3,579 | 923 | 25.8% |
| Qdrant | 923 | 923 | 100% (of translated) |
| Translation Cache | 923 | - | - |

### By Category
| Category | Translated | Total | Percentage |
|----------|------------|-------|------------|
| body | 862 | 1,149 | 75.0% |
| powertrain | 50 | 1,633 | 3.1% |
| network | 6 | 305 | 2.0% |
| chassis | 5 | 492 | 1.0% |

### Target: 80% (2,863 codes)
- Current: 923 translated
- Needed: 1,940 more codes

### Completed Tasks
- [x] Fixed 290 existing translations (removed "erdo" -> "test", "huto" -> "akkumulator")
- [x] Synced translation cache with PostgreSQL
- [x] Updated 923 Neo4j DTCCode nodes with Hungarian descriptions
- [x] Re-indexed 923 vectors to Qdrant (dtc_embeddings_hu collection)
- [x] Created comprehensive translation script (`scripts/continue_translations.py`)

### Pending Tasks
- [ ] Configure translation API key (one of: ANTHROPIC_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY)
- [ ] Run batch translation for remaining 2,656 codes
- [ ] Re-sync Neo4j after new translations
- [ ] Re-index Qdrant after new translations

### How to Continue Translations

1. **Set up an API key** (choose one):
   ```bash
   # Option 1: Groq (free, fast)
   export GROQ_API_KEY="your-key-here"

   # Option 2: DeepSeek (5M free tokens)
   export DEEPSEEK_API_KEY="your-key-here"

   # Option 3: OpenRouter (free tier)
   export OPENROUTER_API_KEY="your-key-here"

   # Option 4: Anthropic (uses existing config)
   export ANTHROPIC_API_KEY="your-key-here"
   ```

2. **Run translations**:
   ```bash
   # Translate all remaining codes
   python scripts/continue_translations.py --translate

   # Or limit to specific count
   python scripts/continue_translations.py --translate --limit 500

   # Then sync to all databases
   python scripts/continue_translations.py --sync-neo4j --reindex-qdrant
   ```

3. **Full workflow**:
   ```bash
   python scripts/continue_translations.py --all
   ```

---

# Database Index Optimization

## Status: COMPLETED (2026-02-05)

### PostgreSQL Indexes (localhost:5432)

#### Existing Indexes (Before)
- `ix_dtc_codes_code` - UNIQUE btree on code column

#### New Indexes Added
| Index Name | Type | Column(s) | Size | Purpose |
|------------|------|-----------|------|---------|
| `ix_dtc_codes_category` | btree | category | 72 kB | Filter by DTC category |
| `ix_dtc_codes_severity` | btree | severity | 80 kB | Filter by severity level |
| `ix_dtc_codes_system` | btree | system | 96 kB | Filter by vehicle system |
| `ix_dtc_codes_is_generic` | btree | is_generic | 80 kB | Filter generic vs manufacturer codes |
| `ix_dtc_codes_category_severity` | btree | (category, severity) | 88 kB | Composite filter |
| `ix_dtc_codes_symptoms` | GIN | symptoms[] | 152 kB | Array contains search |
| `ix_dtc_codes_possible_causes` | GIN | possible_causes[] | 152 kB | Array contains search |
| `ix_dtc_codes_applicable_makes` | GIN | applicable_makes[] | 144 kB | Array contains search |
| `ix_dtc_codes_description_fts` | GIN | description_tsv | 192 kB | Full-text search (EN+HU) |

#### Full-Text Search Setup
- Added `description_tsv` tsvector column
- Created trigger `trg_update_dtc_description_tsv` for auto-update on INSERT/UPDATE
- Supports both English ('english' config) and Hungarian ('simple' config) text

#### Query Performance (EXPLAIN ANALYZE)
| Query Type | Execution Time | Index Used |
|------------|----------------|------------|
| Code lookup (`WHERE code = 'P0171'`) | 0.056 ms | ix_dtc_codes_code (Index Scan) |
| Category filter | 0.031 ms | ix_dtc_codes_category (Index Scan) |
| Severity filter | 0.031 ms | ix_dtc_codes_severity (Index Scan) |
| Full-text search (`'oxygen & sensor'`) | 0.100 ms | ix_dtc_codes_description_fts (Bitmap Index Scan) |

### Neo4j Indexes (localhost:7474)

#### Constraints Added
| Constraint Name | Type | Label | Property |
|-----------------|------|-------|----------|
| `dtc_code_unique` | UNIQUENESS | DTCCode | code |
| `symptom_name_unique` | UNIQUENESS | Symptom | name |

#### Indexes Added
| Index Name | Type | Label | Property(s) |
|------------|------|-------|-------------|
| `dtc_category_idx` | RANGE | DTCCode | category |
| `dtc_severity_idx` | RANGE | DTCCode | severity |
| `dtc_system_idx` | RANGE | DTCCode | system |
| `dtc_is_generic_idx` | RANGE | DTCCode | is_generic |
| `dtc_description_fulltext` | FULLTEXT | DTCCode | description_en, description_hu |

#### Query Performance (PROFILE)
| Query Type | DbHits | Execution Time |
|------------|--------|----------------|
| Code lookup (`{code: 'P0171'}`) | 10 | 40 ms |
| Category count | 1634 | 40 ms |
| Fulltext search (`'oxygen sensor'`) | Uses fulltext index | Fast |

### Recommendations for Future
1. Monitor index usage with `pg_stat_user_indexes` and Neo4j index stats
2. Consider partial indexes if queries often filter on specific values
3. Run `VACUUM ANALYZE dtc_codes` periodically for optimal query planning
4. Consider composite indexes for common multi-column filters in Neo4j

---

# Performance Optimization (2026-02-05)

## Status: COMPLETED

### Summary of Optimizations

| Area | Optimization | Expected Improvement |
|------|--------------|---------------------|
| Database | Connection pooling (10 connections, 20 overflow) | 50-100ms faster connection acquisition |
| Database | Statement caching (100 prepared statements) | 20-30% faster repeated queries |
| Database | Performance indexes (15 new indexes) | 5-10x faster filtered queries |
| Caching | Redis caching for DTC lookups (1hr TTL) | 95%+ cache hit rate, <1ms response |
| Caching | Redis caching for search results (15min TTL) | Eliminates repeated DB queries |
| Caching | Embedding vector caching (1hr TTL) | Avoids expensive GPU inference |
| API | GZip compression (responses >1KB) | 60-80% bandwidth reduction |
| API | ORJSON response serialization | 2-5x faster JSON encoding |
| Embedding | FP16 inference on GPU | 2x faster, 50% memory reduction |
| Embedding | Dynamic batch sizing (32-128 based on GPU) | Optimal throughput |
| Embedding | Async embedding methods | Non-blocking operations |
| Frontend | Code splitting with lazy loading | 40-60% smaller initial bundle |
| Frontend | Optimized chunk splitting | Better caching of vendor libs |
| Frontend | Tree shaking + minification | 20-30% smaller production build |

### 1. Database Connection Pooling (`backend/app/db/postgres/session.py`)

```python
# Pool Configuration
POOL_SIZE = 10          # Base connections (5 in debug mode)
MAX_OVERFLOW = 20       # Extra connections during peak
POOL_RECYCLE = 1800     # Recycle every 30 minutes
POOL_TIMEOUT = 30       # 30 second connection timeout

# asyncpg Optimization
connect_args = {
    "prepared_statement_cache_size": 100,
    "command_timeout": 60,
    "server_settings": {
        "work_mem": "16MB",
        "max_parallel_workers_per_gather": "2",
        "jit": "on",
    },
}
```

### 2. Performance Indexes (`alembic/versions/004_add_performance_indexes.py`)

| Table | Index Type | Purpose |
|-------|------------|---------|
| `dtc_codes` | Composite (category, severity) | Common filter combination |
| `dtc_codes` | Partial (is_generic=true) | Generic code queries |
| `dtc_codes` | GIN (related_codes, symptoms, applicable_makes) | Array contains queries |
| `dtc_codes` | Full-text (description_en + description_hu) | Text search |
| `known_issues` | GIN (related_dtc_codes, applicable_makes) | Array lookups |
| `known_issues` | Partial (confidence >= 0.7) | High-confidence issues |
| `diagnosis_sessions` | Composite (user_id, created_at DESC) | User history queries |
| `diagnosis_sessions` | Composite (vehicle_make, model, year) | Vehicle filtering |
| `diagnosis_sessions` | Partial (last 30 days) | Recent sessions |

### 3. Redis Caching (`backend/app/db/redis_cache.py`)

```python
# Cache TTLs
DTC_CODE = 3600        # 1 hour - DTC codes change rarely
DTC_SEARCH = 900       # 15 minutes - search results
KNOWN_ISSUES = 1800    # 30 minutes
VEHICLE_DATA = 86400   # 24 hours - very static
API_RESPONSE = 300     # 5 minutes - dynamic data
NHTSA_DATA = 21600     # 6 hours - external API
EMBEDDINGS = 3600      # 1 hour

# Features
- Connection pooling (20 max connections)
- Circuit breaker pattern (5 failures -> 30s cooldown)
- Batch operations (mget/mset)
- Automatic JSON serialization
```

### 4. Embedding Optimization (`backend/app/services/embedding_service.py`)

```python
# Performance Features
- FP16 inference on CUDA (2x speedup, 50% memory)
- Dynamic batch sizing:
  - GPU (>8GB): 128
  - GPU (>4GB): 64
  - GPU (<4GB): 32
  - MPS (Apple): 32
  - CPU: 16
- Automatic GPU memory cleanup at 80% utilization
- torch.compile for PyTorch 2.0+
- Redis cache integration for repeated texts
- Async methods (embed_text_async, embed_batch_async)
```

### 5. API Response Optimization (`backend/app/main.py`)

```python
# GZip Compression
application.add_middleware(GZipMiddleware, minimum_size=1000)

# ORJSON (already in place)
default_response_class=ORJSONResponse
```

### 6. Frontend Optimization (`frontend/vite.config.ts`)

```typescript
// Code Splitting
manualChunks: {
  'vendor-react': ['react', 'react-dom'],
  'vendor-router': ['react-router-dom'],
  'vendor-query': ['@tanstack/react-query'],
  'vendor-forms': ['react-hook-form', 'zod'],
  'vendor-ui': ['clsx', 'tailwind-merge', 'lucide-react'],
  'vendor-http': ['axios'],
}

// Optimization
target: 'es2020'
minify: 'terser'
drop_console: true (production)
```

### 7. Frontend Lazy Loading (`frontend/src/App.tsx`)

```tsx
// Lazy loaded pages
const HomePage = lazy(() => import('./pages/HomePage'))
const DiagnosisPage = lazy(() => import('./pages/DiagnosisPage'))
const ResultPage = lazy(() => import('./pages/ResultPage'))
const DTCDetailPage = lazy(() => import('./pages/DTCDetailPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))
```

### Files Modified

| File | Changes |
|------|---------|
| `backend/app/db/postgres/session.py` | Enhanced connection pooling, pool statistics |
| `backend/app/db/redis_cache.py` | New file - comprehensive Redis caching service |
| `backend/app/services/embedding_service.py` | FP16, dynamic batching, async methods, caching |
| `backend/app/api/v1/endpoints/dtc_codes.py` | Redis caching for search and detail endpoints |
| `backend/app/main.py` | GZip middleware, Redis initialization |
| `backend/alembic/versions/004_add_performance_indexes.py` | New migration for indexes |
| `frontend/vite.config.ts` | Optimized build config, code splitting |
| `frontend/src/App.tsx` | Lazy loading for all pages |

### Running the Migration

```bash
# Apply database indexes
cd backend && alembic upgrade head

# Verify indexes
psql -d autocognitix -c "SELECT indexname FROM pg_indexes WHERE tablename = 'dtc_codes';"
```

### Testing Cache Performance

```python
# In Python
from app.db.redis_cache import get_cache_service

cache = await get_cache_service()
stats = await cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']}%")
```

### Monitoring Metrics

- Pool status: `GET /api/v1/health/db` (includes pool statistics)
- Cache stats: Check Redis with `redis-cli INFO`
- Response times: Check Prometheus metrics at `/metrics`

---

# Neo4j Graph Expansion (2026-02-05)

## Status: COMPLETED

## Summary
Expanded the Neo4j diagnostic graph with comprehensive Component and Repair nodes, creating a complete diagnostic knowledge graph.

## Statistics

### Before Expansion
| Entity | Count |
|--------|-------|
| DTC nodes | 3,579 |
| Symptom nodes | 117 |
| CAUSES relationships | 187 |
| Component nodes | ~30 |
| Repair nodes | ~24 |

### After Expansion (Target)
| Entity | Count |
|--------|-------|
| DTC nodes | 3,579 |
| Symptom nodes | 117 |
| Component nodes | 520+ |
| Repair nodes | 320+ |
| INDICATES_FAILURE_OF relationships | 2,000+ |
| REPAIRED_BY relationships | 650+ |

## Files Created

### Data Files (`data/graph_expansion/`)
| File | Description | Size |
|------|-------------|------|
| `components.json` | 520 vehicle component definitions | Engine, transmission, brakes, electrical, HVAC, etc. |
| `repairs.json` | 320 repair procedures | Hungarian/English names, costs, difficulty, tools, time |
| `dtc_component_mappings.json` | 850 DTC-to-component mappings | Pattern matching for P0xxx, Cxxxx, Bxxxx, Uxxxx |
| `component_repair_mappings.json` | 650 component-to-repair mappings | Primary and alternative repair options |

### Script
| File | Description |
|------|-------------|
| `scripts/expand_neo4j_graph.py` | Batch seeding script with parameterized queries |

## Component Categories (520 nodes)

| System | Count | Examples |
|--------|-------|----------|
| Engine | 120+ | MAF, MAP, TPS, VVT, timing components |
| Fuel | 30+ | Injectors, pumps, filters, sensors |
| Ignition | 20+ | Spark plugs, coils, distributors |
| Exhaust/Emissions | 40+ | O2 sensors, catalytic converter, EGR, DPF |
| EVAP | 10+ | Purge valve, vent valve, canister |
| Cooling | 25+ | Thermostat, water pump, radiator |
| Transmission | 35+ | Solenoids, torque converter, TCM |
| Drivetrain | 25+ | CV joints, wheel bearings, differentials |
| Brakes | 40+ | Pads, rotors, calipers, ABS sensors |
| Suspension | 30+ | Struts, springs, control arms |
| Steering | 15+ | Rack, power steering, angle sensor |
| Electrical | 50+ | ECU, alternator, battery, wiring |
| HVAC | 20+ | A/C compressor, blower motor, filters |
| Safety | 25+ | Airbags, seat belts, crash sensors |
| EV/Hybrid | 15+ | Battery pack, inverter, BMS |

## Repair Categories (320 nodes)

| Category | Count | Difficulty Range |
|----------|-------|------------------|
| Maintenance | 25+ | beginner |
| Sensor Replacement | 50+ | beginner-intermediate |
| Cleaning | 15+ | beginner-intermediate |
| Ignition | 15+ | beginner-intermediate |
| Fuel System | 20+ | intermediate-professional |
| Cooling | 15+ | beginner-professional |
| Exhaust/Emissions | 20+ | intermediate-professional |
| Brakes | 25+ | beginner-advanced |
| Suspension | 20+ | intermediate-professional |
| Steering | 15+ | intermediate-professional |
| Transmission | 25+ | intermediate-professional |
| Electrical | 30+ | beginner-professional |
| HVAC | 15+ | intermediate-professional |
| Engine Internal | 10+ | professional |
| Diesel | 15+ | intermediate-professional |
| Diagnostics | 10+ | beginner-intermediate |

## Repair Properties

Each repair node includes:
- `id` - Unique identifier
- `name` - English name
- `name_hu` - Hungarian name
- `difficulty` - beginner/intermediate/advanced/professional
- `time_minutes` - Estimated repair time
- `cost_min/cost_max` - Cost range in HUF
- `tools` - Required tools list
- `category` - Repair category

## Usage

```bash
# Full expansion (components, repairs, relationships)
python scripts/expand_neo4j_graph.py --all

# Individual operations
python scripts/expand_neo4j_graph.py --components
python scripts/expand_neo4j_graph.py --repairs
python scripts/expand_neo4j_graph.py --relationships

# Show statistics
python scripts/expand_neo4j_graph.py --stats

# Validate graph consistency
python scripts/expand_neo4j_graph.py --validate
```

## Security Features

- Parameterized Cypher queries (no injection vulnerabilities)
- String sanitization for all node/relationship properties
- Batch operations with transaction handling
- Proper error recovery and rollback

## Graph Query Examples

```cypher
-- Find repairs for a DTC code
MATCH (d:DTC {code: 'P0171'})-[:INDICATES_FAILURE_OF]->(c:Component)-[:REPAIRED_BY]->(r:Repair)
RETURN d.code, c.name_hu, r.name_hu, r.cost_min, r.cost_max, r.difficulty

-- Get all components in a system
MATCH (c:Component {system: 'engine'})
RETURN c.name, c.name_hu, c.subsystem, c.criticality
ORDER BY c.criticality DESC

-- Find cheapest repair for a component
MATCH (c:Component {id: 'maf_sensor'})-[:REPAIRED_BY]->(r:Repair)
RETURN r.name_hu, r.cost_min, r.difficulty
ORDER BY r.cost_min ASC
LIMIT 1
```

## Next Steps

- [ ] Run the expansion script against production Neo4j
- [ ] Add Part nodes with pricing
- [ ] Create vehicle-specific repair cost adjustments
- [ ] Add TSB (Technical Service Bulletin) relationships
