# Wave2 Refix — Concurrency/Async

HEAD: `9ad6470` (`fix: 5 lead audit kritikus + magas találatok javítva`)

## A) Sentry import cost — LOW (observation)

- `backend/app/core/error_handlers.py:312-343` — `_capture_to_sentry` runs `import sentry_sdk`
  inside `try` on every error invocation. Python caches modules in `sys.modules`, so the
  parse/exec cost is paid once; subsequent calls are a dict lookup (~µs).
- The `try/except ImportError` wrapper is essentially free — `ImportError` is raised only
  the first time when `sentry_sdk` truly isn't installed; after that the `import` returns
  the cached module without re-running the finder chain. So the per-error cost is
  dominated by `Hub.current.client` access + `push_scope()`, not the import.
- Real concern is **not** import cost but the **negative cache miss**: if `sentry_sdk` is
  uninstalled, Python still caches a `ModuleNotFoundError` so the `except ImportError`
  fires fast on every 5xx. Verified — `sys.modules` stores `None` for failed imports
  since 3.3. No optimization warranted.
- Optional micro-improvement (defer to maintainer judgement): hoist
  `try: import sentry_sdk except ImportError: sentry_sdk = None` to module top, then
  do `if sentry_sdk is None: return` inside helper. Removes one frame per error and
  makes the dependency visible to static analyzers. NOT blocking.
- Thread-safety: `sentry_sdk.Hub.current` is async-context aware (uses contextvars
  internally in 1.x+), so calling from multiple concurrent FastAPI handlers is safe.
  `push_scope()` is a context manager scoped to the current hub. OK.

## B) Migration 019 two-step transaction — HIGH

- `backend/alembic/env.py:57` runs every migration inside `context.begin_transaction()`
  (default; `transaction_per_migration` not set). PostgreSQL DDL is transactional, so
  **both** `op.execute(...NOT VALID)` (`019_fix_diagnosis_archive_indexes_and_fk.py:68-75`)
  and `op.execute(...VALIDATE CONSTRAINT)` (`:76-78`) run inside one tx.
- Behaviour analysis:
  - If `VALIDATE` fails (e.g. concurrent insert of an orphan between step 1's DELETE
    and the VALIDATE; or a row inserted via `COPY` bypassing the FK on a replica) → the
    whole transaction rolls back → the `NOT VALID` FK is also reverted. **No orphan
    constraint left behind.** This is the opposite of the original concern.
  - Good news: PostgreSQL atomicity protects us here. The migration is safe to retry.
- BUT: `create_index(..., if_not_exists=True)` on lines `:53-64` — `CREATE INDEX` (not
  CONCURRENTLY) inside a transaction is fine, also rolled back if VALIDATE fails. The
  inline doc comment on `:49-52` mentions CONCURRENTLY but the actual code uses plain
  `create_index` — comment is misleading, not a bug. Recommend tightening the docstring.
- Real risk: between the `DELETE` (`:41-47`) and `ADD CONSTRAINT NOT VALID` (`:68-75`)
  a concurrent INSERT can sneak in a new orphan row, then `VALIDATE` aborts the deploy.
  Mitigation: take `LOCK TABLE diagnosis_archive IN SHARE MODE` at the start of upgrade()
  — blocks INSERTs while migration runs, released on commit. Currently absent.
- Severity: **HIGH** — deploy can flap if traffic is live during migration. Add explicit
  `LOCK TABLE ... IN SHARE MODE` after the orphan DELETE, before the FK add. Or run the
  migration during a maintenance window.

## C) HUBERT_REVISION runtime reload — MEDIUM

- `backend/app/services/embedding_service.py:79-98` — `HungarianEmbeddingService` is a
  thread-safe singleton (`_instance` + `_lock`). `__init__` is gated by `_initialized`,
  so re-constructing returns the same instance with the same `_model`.
- `_load_hubert_model()` at `:196-244` short-circuits at `:205-206` if `self._model is
  not None` — so once loaded, the model is **pinned for process lifetime**.
- `settings.HUBERT_REVISION` is read at `:214` and `:221` only inside the first call
  to `_load_hubert_model()`. After that, mutating `settings.HUBERT_REVISION` (via
  SIGHUP reload, monkeypatch, or hypothetical `settings.refresh()`) has **no effect** —
  the loaded weights stay in memory.
- Edge case: `settings` is a Pydantic `BaseSettings` singleton via `@lru_cache` in
  `app.core.config` — it's not even reload-capable without `settings.cache_clear()`.
  So the only realistic reload path is process restart, which is the correct semantic
  anyway (model weights = ~500MB, hot-swap would double RAM transiently).
- Cache-key concern (cross-reference): Redis cache keys in `embedding_service` (and
  `redis_cache.py` per commit message) now include `HUBERT_REVISION`. If a deploy
  changes `HUBERT_REVISION` env, **new process** picks up new revision AND emits new
  cache keys → no stale-key collision. Correct.
- Severity: **MEDIUM** (documentation). Add a docstring note on `_load_hubert_model`:
  "Revision is captured at first load; changing `HUBERT_REVISION` requires process
  restart, not just settings reload." Operator may otherwise expect hot-reload to work.

## Olvasott fájlok

- `backend/app/core/error_handlers.py` (1-614)
- `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py` (1-91)
- `backend/alembic/env.py` (50-87)
- `backend/app/services/embedding_service.py` (75-244)
