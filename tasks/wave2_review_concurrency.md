# Wave2 Push — Concurrency/Async Lead

## A) _thread_pool lifetime
- **severity:** LOW (managed but fragile)
- **finding:** `backend/app/services/embedding_service.py:57` — module-level `_thread_pool = ThreadPoolExecutor(max_workers=4)`
- **lifecycle audit:**
  - `backend/app/main.py:164-171` — lifespan shutdown calls `_thread_pool.shutdown(wait=True)` ✓
  - `backend/app/services/diagnosis_service.py:43` — direct import of the same module-level object
  - `backend/app/services/diagnosis_service.py:429` — uses it via `loop.run_in_executor(_thread_pool, preprocess_hungarian, symptoms)`
- **stale reference risk:** The import in `diagnosis_service.py:43` is `from app.services.embedding_service import _thread_pool` (name binding at import time). After lifespan shutdown the bound object is the *same* `ThreadPoolExecutor` instance — submitting to it post-shutdown raises `RuntimeError: cannot schedule new futures after shutdown`. Lifespan only fires once at process exit, so in normal request flow there is no race; but **any uvicorn `--reload` or hot-reload scenario will hold a stale executor reference on the *old* module while the new one runs**.
- **no singleton re-creation:** the executor is created at module top level (line 57), not lazily in `get_embedding_service()` — module re-import would create a *new* executor, but `diagnosis_service.py` has already cached the *old* reference.
- **recommendation:** Wrap in a `get_thread_pool()` accessor that recreates on `BrokenExecutor` / shutdown state; or move to a single `app.state.nlp_executor` owned by lifespan. Not blocking for production (no reload), but blocks dev workflow if anyone hits CTRL-C + restart mid-request.

## B) Sentry sync in async
- **severity:** LOW (not actually blocking)
- **finding:** `backend/app/core/error_handlers.py:331-345` — `with sentry_sdk.push_scope() as scope: ... sentry_sdk.capture_exception(exc)` inside async handler
- **transport reality check:** Sentry Python SDK's default `HttpTransport` runs a **`BackgroundWorker` daemon thread** with an in-memory queue (see `sentry_sdk/transport.py`). `capture_exception()` *enqueues* the event and returns immediately — it does NOT perform the HTTP POST inline.
- **`push_scope()` is sync but cheap:** it pushes a `Scope` object onto a thread-local Hub stack (no I/O). The `with` block is microseconds. Safe in async.
- **only blocking case:** if the user has explicitly swapped in a sync `HttpTransport` without the worker (rare custom config) or if the queue is full (`worker.submit()` would drop, not block, by default — `shutdown_timeout` only matters at process exit).
- **init verified:** `backend/app/core/logging.py:631` calls `sentry_sdk.init(...)` with no custom transport → default async-friendly worker is used.
- **no fix needed.** The `try/except` at `error_handlers.py:343-345` already guards against any propagation. `asyncio.to_thread` wrapping would be over-engineering.

## C) Axios timeout retry race
- **severity:** NONE (false alarm — config IS preserved)
- **finding:** `frontend/src/services/diagnosisService.ts:164-166` — `api.post('/diagnosis/analyze', request, { timeout: 120000 })`; `frontend/src/services/api.ts:152` extracts `originalRequest = error.config`; `api.ts:161` and `api.ts:183` retry via `api(originalRequest)`.
- **axios semantics:** `error.config` is the **merged** config that was used for the original request (axios merges defaults + instance + per-call into `config` before dispatch and attaches it to both the request and any resulting `AxiosError`). When you re-invoke `api(originalRequest)`, axios sees `config.timeout === 120000` already set, and the merge order (`defaults < instance < request`) keeps that explicit value — the global 30s default does not overwrite it.
- **verification path:** axios source `lib/core/Axios.js` → `mergeConfig(this.defaults, config)` where per-request `timeout` wins via the `defaultToConfig2` merge strategy.
- **secondary risk (out of scope):** `originalRequest._retry = true` is set on the same object reference, so a second 401 after refresh correctly falls through and throws (no infinite retry). The queue-promise path (`api.ts:157-163`) also passes `originalRequest` unchanged — timeout preserved.
- **no bug.** The commit message claim ("global 30s → 120s for /diagnosis/analyze") holds across the refresh-retry path.

## Olvasott fájlok
- `backend/app/services/embedding_service.py` (lines 1-913)
- `backend/app/services/diagnosis_service.py` (lines 1-120, 420-445)
- `backend/app/main.py` (lines 100-190)
- `backend/app/core/error_handlers.py` (lines 300-360)
- `backend/app/core/logging.py` (lines 620-650)
- `frontend/src/services/api.ts` (lines 1-200)
- `frontend/src/services/diagnosisService.ts` (lines 1-170)
- `git show HEAD --stat`, `git diff main...HEAD --stat`
