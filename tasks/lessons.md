# Lessons Learned - AutoCognitix

## Railway Deployment

### 2024-02-04 - Frontend Build Failures

**Problem:** Frontend build kept failing on Railway with various errors.

**Root Causes:**
1. `.gitignore` had `*.json` rule that excluded `tsconfig.json` and `package.json`
2. `serve` package was in devDependencies instead of dependencies
3. `vite-env.d.ts` was missing for `import.meta.env` types
4. TailwindCSS shadcn/ui custom classes (`border-border`, `bg-background`) not defined

**Solutions:**
- Add explicit exceptions in `.gitignore`: `!frontend/tsconfig.json`, `!frontend/package.json`
- Move `serve` to dependencies for production runtime
- Create `src/vite-env.d.ts` with Vite client types
- Add CSS variable-based colors to `tailwind.config.js`

**Prevention:**
- Always verify all config files are tracked in git before pushing
- Test `npm run build` locally before deployment
- Check for shadcn/ui specific TailwindCSS requirements

---

## Git Patterns

### Always check tracked files before deployment
```bash
git ls-files frontend/ | grep -E "\.(json|ts)$"
```

### Verify .gitignore exceptions work
```bash
git status --ignored
```

---

## 2024-02-05 - DTC Import Scripts Multi-Agent Review

### KRITIKUS HIBÁK (Javítva)

#### 1. Neo4j boolean string-ként tárolva
- **Fájlok:** `scrape_klavkarr.py`, `import_obdb.py`
- **Probléma:** `is_generic=str(...).lower()` → "true"/"false" string
- **Hatás:** Neo4j lekérdezések nem működnek boolean-ra
- **Megoldás:** Boolean típus megtartása
- **Szabály:** SOHA ne konvertálj boolean-t string-re adatbázisba

#### 2. Tranzakció kezelés hiánya
- **Probléma:** Nincs rollback hiba esetén
- **Hatás:** Részleges adatok a DB-ben
- **Megoldás:** try/except/rollback minden batch műveletnél

#### 3. Mutable dict módosítás (`import_obdb.py`)
- **Probléma:** `dtc["vehicle_id"] = vehicle_id` módosítja az eredeti dict-et
- **Megoldás:** `.copy()` használata
- **Szabály:** Új dict mutable objektumok módosításakor

#### 4. JSON injection kockázat
- **Probléma:** Web response-ok nincsenek szanitizálva
- **Megoldás:** Input validálás és szanitizálás

#### 5. ReDoS kockázat (`translate_to_hungarian.py`)
- **Probléma:** `(.+?)` regex DOTALL flag-gel
- **Megoldás:** Specifikusabb regex, méret limitálás

### FIGYELMEZTETÉSEK

- Race condition SELECT + INSERT → `ON CONFLICT` használata
- Unbounded retry → Total timeout hozzáadása
- Path traversal → Fájl útvonal validálás
- `datetime.utcnow()` deprecated → `datetime.now(timezone.utc)`

### KÓD DUPLIKÁCIÓ

- `get_sync_db_url()` 3 fájlban → Shared utils modul
- DeepSeek/Kimi translation 90% azonos → Absztrakt factory

---

## Szabályok a Jövőre

### Adatbázis Műveletek
```python
# HELYES - tranzakció kezeléssel
try:
    with Session(engine) as session:
        for item in batch:
            session.add(item)
        session.commit()
except Exception as e:
    session.rollback()
    raise
```

### Mutable Objektumok
```python
# HELYTELEN
for item in items:
    item["new_key"] = value  # Módosítja az eredetit!

# HELYES
for item in items:
    new_item = item.copy()
    new_item["new_key"] = value
```

### Boolean Típusok
```python
# HELYTELEN
is_generic=str(value).lower()  # "true"/"false" string

# HELYES
is_generic=bool(value)  # Igazi boolean
```

### Input Szanitizálás
```python
# Mindig validáld a külső inputot
def sanitize_description(text: str, max_length: int = 1000) -> str:
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)  # HTML tag-ek eltávolítása
    return text[:max_length]
```

---

## 2026-02-05 - RepairPal Scraping

### Cloudflare Protection Handling

**Problem:** RepairPal uses Cloudflare protection which blocks standard HTTP requests (httpx, requests).

**Symptoms:**
- `curl` returns HTML with "Just a moment..." challenge page
- JavaScript verification required
- 403 Forbidden responses for automated requests

**Solution:**
- Use Playwright instead of httpx for sites with Cloudflare
- Launch browser in headless mode with stealth settings
- Add init script to hide webdriver detection

```python
# Cloudflare bypass with Playwright
browser = await p.chromium.launch(
    headless=True,
    args=['--disable-blink-features=AutomationControlled']
)
await context.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
""")
```

**Prevention:**
- Always check robots.txt first (even if blocked, indicates protection)
- Respect rate limits (2+ seconds between requests)
- Use realistic User-Agent strings
- Add viewport and locale settings

### Scraping Best Practices

**Always Do:**
1. Check robots.txt before scraping
2. Implement rate limiting (respect crawl-delay)
3. Use try/except with retries for transient failures
4. Validate all scraped data before saving
5. Map external data to existing internal structures

**Never Do:**
1. Bypass authentication
2. Ignore rate limits
3. Store unvalidated external data
4. Scrape without checking legal terms

### Data Mapping Strategy

For repair cost data linked to DTC codes:
```python
# Create bidirectional mapping
DTC_REPAIR_MAPPING = {
    "P0130": ["oxygen-sensor-replacement"],
    "P0420": ["catalytic-converter-replacement"],
}

# Enrich DTC codes with repair estimates
def update_dtc_codes_with_repairs(repairs):
    for dtc_code, repair_info in repair_map.items():
        if dtc_code in dtc_codes:
            dtc_codes[dtc_code]["repair_estimates"] = repair_info
```

---

## 2026-02-05 - Hungarian DTC Translation

### LLM Translation Quality Issues

**Problem:** LLM translations produced incorrect Hungarian automotive terms:
- "ground" translated as "erdo" (forest) instead of "test" or "fold"
- "battery" translated as "huto" (cooler/refrigerator) instead of "akkumulator"
- "circuit" translated as "nyomkor" instead of "aramkor"

**Root Cause:** LLMs without automotive domain knowledge confuse homonyms:
- "ground" = ground/earth (fold/talaj) vs electrical ground (test)
- "battery" = battery/cooler similar context in some languages

**Solution:** Created `fix_translations.py` with 42 regex patterns to detect and fix:
1. Pattern-based replacement for known mistranslations
2. Quality validation before saving
3. Detailed prompt with explicit "NEVER translate as..." rules

**Prevention Rules:**
```python
# Always include explicit negative examples in translation prompts
TRANSLATION_RULES = """
FONTOS SZABALYOK:
1. A "ground" szo MINDIG "test" vagy "fold" legyen, SOHA NEM "erdo"
2. A "battery" szo MINDIG "akkumulator" legyen, SOHA NEM "huto"
3. Az "open circuit" MINDIG "aramkor szakadas"
4. A "short to ground" MINDIG "testre zaras"
"""
```

### Multi-Database Sync Strategy

**Problem:** Translations need to be synced across PostgreSQL, Neo4j, and Qdrant consistently.

**Solution:** Hierarchical sync approach:
1. PostgreSQL is source of truth for structured data
2. Translation cache file for checkpointing
3. Neo4j synced from PostgreSQL
4. Qdrant re-indexed from PostgreSQL

```python
# Sync order
1. API -> PostgreSQL (with validation)
2. PostgreSQL -> Translation Cache (backup)
3. PostgreSQL -> Neo4j (Cypher MATCH/SET)
4. PostgreSQL -> Qdrant (full re-index with embeddings)
```

### API Key Management for Translation

**Problem:** Multiple LLM providers available, need fallback strategy.

**Solution:** Provider priority with auto-detection:
```python
PRIORITY = ["anthropic", "groq", "deepseek", "openrouter", "mistral", "kimi"]

def get_available_provider():
    for provider in PRIORITY:
        api_key = os.environ.get(PROVIDERS[provider]["env_key"])
        if api_key and "your_" not in api_key.lower():
            return provider, api_key
    return None
```

**Key Insight:** Load `.env` file explicitly with `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")
```

---

## 2026-02-05 - Performance Optimization

### Database Connection Pooling

**Problem:** Default connection settings caused connection acquisition delays during peak load.

**Solution:** Optimized pool configuration:
```python
POOL_SIZE = 10          # More base connections
MAX_OVERFLOW = 20       # Allow burst capacity
POOL_RECYCLE = 1800     # Prevent stale connections
pool_pre_ping = True    # Detect dead connections
```

**Key Insight:** Statement caching in asyncpg provides significant speedup:
```python
connect_args = {"prepared_statement_cache_size": 100}
```

### Redis Caching Patterns

**Problem:** Repeated database queries for frequently accessed data (DTC codes).

**Solution:** Multi-tier caching strategy:
1. **DTC codes**: 1 hour TTL (rarely change)
2. **Search results**: 15 min TTL (query-dependent)
3. **Embeddings**: 1 hour TTL (expensive to compute)
4. **NHTSA data**: 6 hour TTL (external API)

**Circuit Breaker Pattern:**
```python
# Prevent cascading failures when Redis is down
if self._failure_count >= self._max_failures:
    self._circuit_open = True
    asyncio.create_task(self._reset_circuit())  # Reset after 30s
```

**Key Insight:** Always return sensible defaults when cache is unavailable - don't fail the request.

### Embedding Service Optimization

**Problem:** GPU memory exhaustion with large batches, slow cold start.

**Solutions:**
1. **FP16 inference on CUDA** - 2x speedup, 50% memory
2. **Dynamic batch sizing** - Based on available GPU memory
3. **Lazy loading** - Load model only on first use
4. **GPU memory cleanup** - At 80% utilization threshold

```python
# Dynamic batch sizing
if gpu_memory > 8GB: batch_size = 128
elif gpu_memory > 4GB: batch_size = 64
else: batch_size = 32
```

**Key Insight:** Use `torch.compile` on PyTorch 2.0+ for additional 10-20% speedup.

### Frontend Code Splitting

**Problem:** Large initial bundle size affecting Time to Interactive.

**Solution:** Lazy loading with Suspense:
```tsx
const DiagnosisPage = lazy(() => import('./pages/DiagnosisPage'))

<Suspense fallback={<LoadingSpinner />}>
  <Routes>...</Routes>
</Suspense>
```

**Vite chunk optimization:**
```typescript
manualChunks: {
  'vendor-react': ['react', 'react-dom'],  // Rarely changes
  'vendor-query': ['@tanstack/react-query'], // Changes sometimes
  'vendor-ui': ['lucide-react'],  // UI specific
}
```

**Key Insight:** Separate vendor chunks by change frequency for optimal caching.

### GZip Compression

**Best Practice:** Enable for responses > 1KB:
```python
application.add_middleware(GZipMiddleware, minimum_size=1000)
```

**Don't compress:**
- Already compressed content (images, videos)
- Very small responses (overhead not worth it)
- WebSocket connections

### Performance Monitoring Rules

1. **Always measure before optimizing** - Profile first
2. **Cache hit rate matters** - Target 90%+ for static data
3. **Database indexes need monitoring** - Check usage with `pg_stat_user_indexes`
4. **Connection pool metrics** - Monitor checked_out vs pool_size
5. **GPU memory tracking** - Cleanup before OOM, not after

---

## 2026-02-05 - Error Handling Architecture

### Exception Hierarchy Design

**Problem:** Scattered error handling with inconsistent responses and no Hungarian translations.

**Solution:** Centralized exception hierarchy with:
1. **Base exception class** with standardized attributes
2. **Category-based error codes** (ERR_1xxx to ERR_5xxx)
3. **Hungarian message mapping** for all error codes
4. **Structured JSON responses** with request ID tracing

```python
# Exception hierarchy pattern
class AutoCognitixException(Exception):
    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[Dict] = None,
        status_code: int = 500,
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        self.status_code = status_code

# Specialized exceptions inherit and set defaults
class DatabaseException(AutoCognitixException):
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(
            message=message,
            code=ErrorCode.DATABASE_ERROR,
            status_code=503,
        )
        self.original_error = original_error
```

**Key Insight:** Store the original exception for debugging while showing user-friendly messages.

### Global Exception Handler Pattern

**Problem:** FastAPI default error responses don't include request context or Hungarian translations.

**Solution:** Register handlers for all exception types:
```python
def setup_exception_handlers(app: FastAPI):
    app.add_exception_handler(AutoCognitixException, custom_handler)
    app.add_exception_handler(ValidationError, validation_handler)
    app.add_exception_handler(SQLAlchemyError, database_handler)
    app.add_exception_handler(Exception, generic_handler)  # Fallback last
```

**Key Insight:** Order matters - specific handlers before generic, custom before library exceptions.

### Request ID Tracing

**Problem:** Difficult to correlate logs with specific user requests.

**Solution:** Middleware that adds request ID to all requests:
```python
class RequestContextMiddleware:
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

**Key Insight:** Use context variables for async-safe request ID propagation to logging.

### Retry with Exponential Backoff

**Problem:** Transient failures (network, rate limits) cause immediate request failures.

**Solution:** Configurable retry decorator:
```python
@retry_async(config=RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True,
))
async def call_external_api():
    ...
```

**Key Patterns:**
1. **Jitter** - Random variation prevents thundering herd
2. **Rate limit detection** - Honor Retry-After headers
3. **Selective retry** - Only retry recoverable errors (5xx, network)
4. **Never retry 4xx** - Client errors (except 429) are permanent

### Frontend Error Boundary

**Problem:** Uncaught React rendering errors crash the entire application.

**Solution:** Error boundary with recovery options:
```tsx
class ErrorBoundary extends Component {
    componentDidCatch(error, errorInfo) {
        // Log error
        this.setState({ hasError: true, error })
    }

    render() {
        if (this.state.hasError) {
            return <ErrorFallback onRetry={this.handleRetry} />
        }
        return this.props.children
    }
}
```

**Key Insight:** Wrap at multiple levels - app root for catastrophic, page level for recoverable.

### API Error Type Detection

**Problem:** Frontend needs to show appropriate UI for different error types.

**Solution:** Detect error type from status code and error response:
```typescript
function detectErrorType(error: ApiError): ErrorType {
    if (error.isNetworkError) return 'network'
    switch (error.status) {
        case 401: return 'unauthorized'
        case 403: return 'forbidden'
        case 404: return 'not_found'
        case 429: return 'rate_limit'
        case 500: return 'server'
        default: return 'generic'
    }
}
```

**Key Insight:** Show contextual actions (login for 401, retry for 500/network).

### TanStack Query Error Handling

**Problem:** Default retry behavior retries all errors including permanent ones.

**Solution:** Custom retry function:
```typescript
function shouldRetry(failureCount: number, error: unknown): boolean {
    if (failureCount >= 3) return false

    if (error instanceof ApiError) {
        // Don't retry client errors except rate limit
        if (error.status >= 400 && error.status < 500 && error.status !== 429) {
            return false
        }
        return true
    }
    return true
}
```

**Key Insight:** Configure `retryDelay` with exponential backoff: `Math.min(1000 * 2 ** attemptIndex, 30000)`

### Error Handling Rules

1. **Always include original error** for debugging, but don't expose to users
2. **Translate error messages** at the boundary (API response), not in business logic
3. **Use error codes** for programmatic handling, messages for display
4. **Log context** (request_id, user_id, resource_id) with every error
5. **Different log levels**: WARNING for client errors (4xx), ERROR for server errors (5xx)
6. **Never swallow exceptions** silently - log or propagate
7. **Graceful degradation** - return partial results when possible
8. **Circuit breaker** for external services - prevent cascade failures

---

## 2026-02-05 - CI/CD Pipeline Lint Failures

### Ruff Linting Configuration

**Problem:** GitHub Actions CI pipeline failed with 100+ ruff lint errors after code generation.

**Root Causes:**
1. Code generated using older Python typing syntax (pre-3.9 style)
2. Missing ruff configuration for ignore rules
3. Test files need different rules than production code
4. Lazy imports (common in FastAPI) flagged as errors

**Common Error Categories:**

| Error Code | Description | Fix |
|------------|-------------|-----|
| UP035 | `typing.Dict/List` deprecated | Use built-in `dict/list` or ignore |
| UP006 | Use `dict` instead of `Dict` | Add to ignore list for legacy code |
| UP045 | Use `X \| None` instead of `Optional[X]` | Modern Python 3.10+ syntax |
| PLC0415 | Import not at top of file | Required for lazy/conditional imports |
| ERA001 | Commented out code | Sometimes needed for TODO/docs |
| I001 | Import block unsorted | Use ruff --fix or isort |

**Solution:** Comprehensive `ruff.toml` configuration:

```toml
# backend/ruff.toml
[lint]
ignore = [
    "E501",    # Line too long (handled by formatter)
    "B008",    # Function call in default argument (FastAPI Depends)
    "UP035",   # typing.Dict/List deprecated
    "UP006",   # Use dict/list instead of Dict/List
    "UP045",   # Use X | None instead of Optional
    "PLC0415", # Import not at top (lazy imports)
    "PLW0603", # Global statement (singletons)
    "ERA001",  # Commented out code
    "I001",    # Import block un-sorted
]

[lint.per-file-ignores]
# Tests have relaxed rules
"tests/**/*.py" = [
    "PLR2004",  # Magic value
    "S101",     # Assert
    "ARG001",   # Unused arg
    "E402",     # Import not at top
    "F401",     # Unused import
]
# Init files can have unused imports
"__init__.py" = ["F401"]
```

**Auto-Fix Command:**
```bash
python3 -m ruff check app tests --fix --unsafe-fixes
```

**Prevention Rules:**

1. **Before committing:**
   ```bash
   python3 -m ruff check app tests
   python3 -m ruff format app tests
   ```

2. **Configure CI to run with same config:**
   ```yaml
   - name: Run Ruff linter
     run: |
       cd backend
       ruff check app tests --output-format=github
   ```

3. **Add ruff to pre-commit hooks:**
   ```yaml
   # .pre-commit-config.yaml
   - repo: https://github.com/astral-sh/ruff-pre-commit
     rev: v0.2.1
     hooks:
       - id: ruff
         args: [--fix, --exit-non-zero-on-fix]
   ```

### Key Insights

1. **Older typing imports are fine** - `typing.Dict`, `Optional[X]` still work, just deprecated
2. **Lazy imports are a pattern** - FastAPI often needs them to avoid circular imports
3. **Test files need different rules** - Magic values, assertions, unused fixtures are normal
4. **ruff --fix saves time** - Auto-fixes 80% of issues safely

### GitHub Actions Security Workflow

**Problem:** `security.yml` workflow shows "workflow file issue" error.

**Common Causes:**
1. YAML syntax errors (indentation)
2. Invalid action versions
3. Missing permissions
4. Invalid paths in checkout

**Prevention:**
```bash
# Validate YAML locally
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/security.yml'))"

# Use GitHub CLI to check
gh workflow view security.yml
```

### Workflow Dependencies

**Pattern:** CD workflow depends on CI passing before deployment.

```yaml
jobs:
  deploy:
    needs: [ci-success]
    if: needs.ci-success.result == 'success'
```

**Key Insight:** Always gate deployments on all quality checks passing.

---

## 2026-02-05 - Docker Frontend Build Failures

### package-lock.json Sync Issues

**Problem:** `npm ci` fails with "Missing: package@version from lock file"

**Root Cause:** `package.json` and `package-lock.json` are out of sync. New packages added to `package.json` but `npm install` wasn't run to update the lock file.

**Solution:**
```bash
# Always run npm install after adding packages
cd frontend && npm install

# Verify lock file is committed
git add package-lock.json
```

**Prevention Rules:**
1. **Before committing package.json changes:** Always run `npm install`
2. **Before Docker builds:** Verify `npm ci` works locally first
3. **In CI:** Use `npm ci` (not `npm install`) for reproducible builds

### npm ci Invalid Flags

**Problem:** `npm ci --only=production=false` is deprecated in npm 10+

**Error Message:**
```
npm warn invalid config only="production=false" set in command line options
npm warn invalid config Must be one of: null, prod, production
```

**Solution:** Use simple `npm ci` (includes all deps) or use `--omit=dev` for production-only.

**Correct Dockerfile:**
```dockerfile
# For build (need devDependencies)
RUN npm ci

# For runtime-only (production deps only)
RUN npm ci --omit=dev
```

**Key Insight:** Always test Docker builds locally before pushing:
```bash
docker build -f frontend/Dockerfile.prod -t test frontend/
```

---

## 2026-02-05 - SQLAlchemy Reserved Words (CRITICAL)

### A `metadata` Oszlopnév Probléma

**Problem:** SQLAlchemy `metadata` oszlopnév használata hibát okoz, mert a `metadata` fenntartott szó a SQLAlchemy-ben.

**Root Cause:** A SQLAlchemy `DeclarativeBase` osztály a `metadata` attribútumot használja a tábla metaadatokhoz. Ha ugyanezt a nevet használjuk oszlopnévként, ütközés keletkezik.

**Hibás kód:**
```python
# HELYTELEN - metadata fenntartott szó!
class NHTSAVehicleSyncTracking(Base):
    __tablename__ = "nhtsa_vehicle_sync_tracking"

    metadata: Mapped[dict | None] = mapped_column(JSONB)  # HIBA!
```

**Hiba típusa:**
```
AttributeError: 'Column' object has no attribute 'tables'
# vagy
TypeError: 'MetaData' object is not callable
```

**Megoldás:**
```python
# HELYES - használj alternatív nevet
class NHTSAVehicleSyncTracking(Base):
    __tablename__ = "nhtsa_vehicle_sync_tracking"

    sync_metadata: Mapped[dict | None] = mapped_column(JSONB)  # OK!
```

**További SQLAlchemy Fenntartott Szavak:**

| Fenntartott szó | Alternatíva |
|-----------------|-------------|
| `metadata` | `sync_metadata`, `item_metadata`, `extra_data` |
| `registry` | `item_registry`, `data_registry` |
| `query` | `search_query`, `query_text` |
| `columns` | `column_list`, `field_columns` |
| `tables` | `table_list`, `related_tables` |

**Prevention Rules:**

1. **Migrációk írása előtt:**
   - Ellenőrizd a SQLAlchemy dokumentációt fenntartott szavakra
   - Használj prefixet: `sync_`, `item_`, `data_`, stb.

2. **Code review checklist:**
   - Új oszlopnév? Ellenőrizd: `hasattr(Base, column_name)`
   - Ha True, használj másik nevet

3. **Naming convention:**
   ```python
   # Projekt-szintű szabály: JSONB oszlopok prefixe
   sync_metadata: Mapped[dict | None] = mapped_column(JSONB)
   extra_data: Mapped[dict | None] = mapped_column(JSONB)
   raw_response: Mapped[dict | None] = mapped_column(JSONB)  # OK, specifikus
   ```

---

## 2026-02-05 - npm package-lock.json Sync Issues (CRITICAL)

### Lock File Szinkronizáció

**Problem:** Docker build vagy CI `npm ci` sikertelen: "Missing: package@version from lock file"

**Root Cause:** `package.json` és `package-lock.json` nincs szinkronban. Új package hozzáadva, de `npm install` nem futott.

**Hibaüzenet:**
```
npm ERR! Missing: @tanstack/react-query@^5.59.20 from lock file
npm ERR! Missing: lucide-react@^0.460.0 from lock file

npm ERR! `npm ci` can only install packages when your package.json and package-lock.json are in sync.
```

**Megoldás:**
```bash
# 1. Lokálisan szinkronizáld
cd frontend && npm install

# 2. Commit-old mindkét fájlt
git add package.json package-lock.json
git commit -m "chore: sync package-lock.json"
```

**npm ci vs npm install:**

| Parancs | Használat | Lock file |
|---------|-----------|-----------|
| `npm install` | Fejlesztés, új package hozzáadása | Frissíti |
| `npm ci` | CI/CD, Docker build | Nem frissíti, hibázik ha nem szinkron |

**Prevention Rules:**

1. **Package hozzáadása után MINDIG:**
   ```bash
   npm install <package>
   git add package.json package-lock.json
   ```

2. **Docker build előtt MINDIG:**
   ```bash
   npm ci  # Lokálisan tesztelés
   docker build -f Dockerfile.prod -t test .
   ```

3. **CI/CD workflow:**
   ```yaml
   - name: Install dependencies
     run: |
       cd frontend
       npm ci  # NEM npm install!
   ```

4. **Pre-commit hook (opcionális):**
   ```bash
   # .husky/pre-commit
   if git diff --cached --name-only | grep -q "package.json"; then
       if ! git diff --cached --name-only | grep -q "package-lock.json"; then
           echo "ERROR: package.json changed but package-lock.json wasn't updated"
           exit 1
       fi
   fi
   ```

### npm Deprecated Flags

**Problem:** `npm ci --only=production=false` deprecated npm 10+

**Hiba:**
```
npm warn invalid config only="production=false" set in command line options
npm warn invalid config Must be one of: null, prod, production
```

**Megoldás:**
```dockerfile
# Build fázis (kell devDependencies TypeScript-hez)
RUN npm ci

# Production runtime (csak dependencies)
RUN npm ci --omit=dev
```

---

## 2026-02-05 - GitHub Actions Conditional Execution

### Job Dependencies és Conditional Logic

**Problem:** CD workflow futott CI nélkül, sikertelen deployment.

**Root Cause:** Nem volt `needs` és `if` kondíció a deployment job-on.

**Megoldás - Job Dependencies:**
```yaml
jobs:
  lint:
    runs-on: ubuntu-latest
    steps: [...]

  test:
    runs-on: ubuntu-latest
    steps: [...]

  deploy:
    needs: [lint, test]  # Várd meg mindkettőt
    if: needs.lint.result == 'success' && needs.test.result == 'success'
    runs-on: ubuntu-latest
    steps: [...]
```

**Fontos `if` Kondíciók:**

| Kondíció | Jelentés |
|----------|----------|
| `if: always()` | Mindig fut, függetlenül a korábbi job-ok eredményétől |
| `if: success()` | Csak ha minden korábbi job sikeres (default) |
| `if: failure()` | Csak ha valamelyik korábbi job sikertelen |
| `if: cancelled()` | Csak ha a workflow cancelled |
| `if: needs.job-name.result == 'success'` | Specifikus job eredménye alapján |

**CI Success Gate Pattern:**
```yaml
# Központi "gate" job ami összefoglalja az összes ellenőrzést
ci-success:
  name: CI Success
  runs-on: ubuntu-latest
  needs:
    - backend-lint
    - backend-type-check
    - backend-test
    - frontend-lint
    - frontend-build
    - docker-build
  if: always()  # Mindig fusson, hogy jelenthessen
  steps:
    - name: Check all jobs passed
      run: |
        if [ "${{ needs.backend-lint.result }}" != "success" ] || \
           [ "${{ needs.backend-test.result }}" != "success" ] || \
           [ "${{ needs.frontend-build.result }}" != "success" ]; then
          echo "One or more required jobs failed"
          exit 1
        fi
        echo "All CI checks passed!"
```

**Branch Protection Integration:**
```yaml
# A ci-success job-ot add hozzá required status check-ként
# GitHub Settings → Branches → Branch protection rules
# Require status checks: "CI Success"
```

### Workflow Triggers

**Helyes trigger konfiguráció:**
```yaml
on:
  push:
    branches: [main, develop]
    paths:
      - 'backend/**'  # Csak ha backend változott
      - 'frontend/**'
  pull_request:
    branches: [main]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true  # Korábbi futó workflow-t állítsd le
```

### Environment és Secrets

**Environment-specifikus deployment:**
```yaml
deploy-railway:
  environment:
    name: ${{ needs.prepare.outputs.environment }}  # staging/production
    url: ${{ steps.deploy.outputs.url }}
  steps:
    - name: Deploy
      env:
        RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}  # Environment secret
```

---

## 2026-02-05 - Railway Deployment Patterns

### Railway.toml Konfiguráció

**Backend (backend/railway.toml):**
```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile.prod"

[deploy]
startCommand = "uvicorn app.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

**Frontend (frontend/railway.toml):**
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "npm start"
healthcheckPath = "/"
healthcheckTimeout = 300
```

### Environment Variables Pattern

**Railway Auto-Provided:**
```
DATABASE_URL=postgresql://...  # Railway PostgreSQL
REDIS_URL=redis://...           # Railway Redis
PORT=...                        # Railway port
```

**Manual Configuration Required:**
```
# External services
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_PASSWORD=...
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=...

# API keys
ANTHROPIC_API_KEY=...
JWT_SECRET_KEY=...

# App config
ENVIRONMENT=production
ALLOWED_ORIGINS=https://yourdomain.com
```

### Health Check Implementation

**Backend health endpoint:**
```python
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }
```

**CD Workflow Smoke Test:**
```yaml
- name: Health check - Backend
  run: |
    for i in {1..10}; do
      status=$(curl -s -o /dev/null -w "%{http_code}" "${{ vars.API_URL }}/health" || echo "000")
      if [ "$status" == "200" ]; then
        echo "Backend is healthy!"
        exit 0
      fi
      echo "Attempt $i: Backend returned $status, waiting..."
      sleep 10
    done
    echo "Backend health check failed after 10 attempts"
    exit 1
```

### Rollback Strategy

**Automatikus rollback issue creation:**
```yaml
rollback:
  needs: [prepare, smoke-tests]
  if: failure() && needs.smoke-tests.result == 'failure'
  steps:
    - name: Create rollback issue
      uses: actions/github-script@v7
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: `Deployment Failed - Rollback Required (${context.sha.substring(0, 8)})`,
            body: `## Deployment Failure\n\n- **Version:** ...\n- **Environment:** ...\n\nSmoke tests failed. Manual rollback may be required.`,
            labels: ['deployment', 'urgent', 'bug']
          })
```

### Alembic Migrations on Railway

**Fontos:** Migráció futtatása deployment után:
```yaml
run-migrations:
  needs: [deploy-railway]
  steps:
    - name: Run Alembic migrations
      run: |
        cd backend
        alembic upgrade head
      env:
        DATABASE_URL: ${{ secrets.DATABASE_URL }}
      continue-on-error: true  # Ne bukjon el a deployment ha a migráció már lefutott
```

### Railway Deployment Rules

1. **Mindig teszteld lokálisan Docker-rel:**
   ```bash
   docker build -f Dockerfile.prod -t test .
   docker run -p 8000:8000 test
   ```

2. **Health check legyen gyors (< 5 sec):**
   - Ne csatlakozz adatbázishoz a health check-ben
   - Csak az alkalmazás állapotát ellenőrizd

3. **Environment variables:**
   - Soha ne hardcode-olj secret-eket
   - Használj Railway Variables vagy GitHub Secrets-et

4. **Logging:**
   - Railway automatikusan gyűjti a stdout/stderr-t
   - Használj strukturált JSON logging-ot

5. **Cost management:**
   - Használj scale-to-zero staging-en
   - Figyelj a resource usage-re a dashboard-on

---

## CI/CD Checklist - Minden Commit Előtt

### Backend
```bash
# 1. Ruff linting
cd backend && python3 -m ruff check app tests --fix

# 2. Ruff formatting
cd backend && python3 -m ruff format app tests

# 3. Type check
cd backend && mypy app --ignore-missing-imports

# 4. Unit tests
cd backend && pytest tests -v --ignore=tests/integration --ignore=tests/e2e
```

### Frontend
```bash
# 1. Sync lock file
cd frontend && npm install

# 2. Lint
cd frontend && npm run lint

# 3. Type check
cd frontend && npx tsc --noEmit

# 4. Build
cd frontend && npm run build
```

### Docker
```bash
# Backend
docker build -f backend/Dockerfile.prod -t backend-test backend/

# Frontend
docker build -f frontend/Dockerfile.prod -t frontend-test frontend/
```

### Git
```bash
# Verify tracked files
git status

# Check for unintended changes
git diff

# Commit with clear message
git commit -m "feat/fix/chore: clear description"
```
