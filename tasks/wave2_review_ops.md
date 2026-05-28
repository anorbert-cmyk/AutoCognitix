# Wave2 Push — Operational/Observability Lead

Audit branch: `claude/sprint13-bugfixes-wave2` vs `main`. Focus: Sentry coverage,
120s timeout UX, migration deploy strategy. Code review against the actual diff
+ touched files; cross-checked CD pipeline + frontend wizard flows.

## A) Sentry coverage gaps

- severity: **HIGH** (silent loss of 5xx telemetry in prod)
- finding: `backend/app/core/error_handlers.py:303` got an explicit
  `sentry_sdk.capture_exception` (good). The other three registered handlers do
  NOT, so any exception they catch is invisible to Sentry whenever the project
  customises logging (Sentry `LoggingIntegration` is **off** the moment a
  non-stdlib root logger is reconfigured — and AutoCognitix already wraps
  logging in `app.core.logging`). The gap matters most for 5xx classes.
- `backend/app/core/error_handlers.py:240` `sqlalchemy_exception_handler` —
  returns 500/503/504 for `OperationalError`, `DBAPIError`,
  `SQLAlchemyTimeoutError`. These are real prod incidents (pool exhaustion,
  Neo4j-style outages). MUST capture; currently only `logger.error` (line 285).
- `backend/app/core/error_handlers.py:139` `autocognitix_exception_handler` —
  re-raises whatever `exc.status_code` the custom exception declared. For
  `status_code >= 500` Sentry should fire. Today they only `logger.warning`
  (line 146) regardless of severity, so 5xx custom errors silently drop.
- `backend/app/core/error_handlers.py:411` (`neo4j_exception_handler`, defined
  inside `setup_neo4j_exception_handler`) — every branch returns 503 and is
  always a real outage signal, but only `logger.error` is wired. Add capture.
- 4xx handlers (`validation_exception_handler:165`,
  `pydantic_validation_exception_handler:203`) → correctly skipped, agree with
  the brief. HTTPX handler at `:518` should capture only on `ConnectError` /
  500-class upstream, not on `TimeoutException` (already noisy).
- fix shape: extract the try/import/push_scope block from
  `generic_exception_handler:328-345` into a `_capture_to_sentry(exc, request,
  request_id, level)` helper and call it from the four 5xx-emitting handlers.
  Saves ~60 LOC of duplication.
- bonus: `sentry_sdk.Hub.current` is **deprecated** in sentry-sdk 2.x — use
  `sentry_sdk.get_client()` (returns `NonRecordingClient` when unset, so the
  `is not None` guard becomes unnecessary). Today's check still works but will
  warn on the next SDK bump.

## B) 120s timeout UI feedback

- severity: **HIGH** (silent UX regression in the simple wizard path)
- two diagnosis entrypoints exist and behave very differently:
  - `frontend/src/pages/DiagnosisPage.tsx:197` mounts `<AnalysisProgress>`
    which uses **SSE streaming** with step updates from the backend. User sees
    "Vector keresés / RAG / LLM válasz" steps tick over in real time, and a
    matching 2-min abort timer at
    `frontend/src/components/features/diagnosis/AnalysisProgress.tsx:351-362`
    (matches the new 120s axios timeout — good).
  - `frontend/src/pages/NewDiagnosisPage.tsx:84-118` is the problem. It
    `setInterval(..., 1500)` over 4-5 mock steps → animation *completes in
    ~7-9 seconds*, then `currentAnalysisStepIndex` sits on the last step
    while `mutateAsync` is still pending for another 50-110 seconds. Visual
    state = "done", actual state = "waiting" → user thinks it froze and
    refreshes (losing the response). Worse than no progress bar.
- no elapsed-time hint anywhere: `grep -n "elapsed|másodperc|Eltelt"` in
  `AnalysisProgress.tsx` returns zero results. After ~30s the user has no
  signal that "this is still legitimately working."
- fixes (cheap → less cheap):
  1. `NewDiagnosisPage.tsx:93` — drive the steps from a real elapsed-time
     curve (e.g. step N at 0%, 15%, 35%, 60%, 85% of expected 60s) and
     **never auto-advance past the penultimate step until `isPending=false`**.
     Pin the last step to a perpetual shimmer while waiting.
  2. `AnalysisProgress.tsx` — add a `const elapsed = useElapsedSeconds(start)`
     hook and render "Még folyamatban... ({elapsed}s eltelt, általában 30-90s)"
     after 20s. Trivial change, huge perceived-perf win.
  3. Wire the `NewDiagnosisPage` flow through `<AnalysisProgress>` too —
     `DiagnosisPage` already proves it works. Removes the dual-flow drift.
- axios timeout itself (`diagnosisService.ts:162-165`): 120s is correct given
  the 2-min backend SSE timeout. Good alignment.

## C) Migration deploy & rollback

- severity: **CRITICAL** (data loss / stuck deploy on first prod run)
- `cd.yml:232-256` runs `alembic upgrade head` as a **separate job after
  `deploy-railway` succeeds**. Order is wrong: new backend code is already
  live before the schema it depends on exists. If 019 fails (see below),
  the prod backend is serving with the old schema and `continue-on-error:
  false` only fails the migration job — the deployment job is already green.
- `019_fix_diagnosis_archive_indexes_and_fk.py:36-46` `op.create_foreign_key`
  is **not idempotent**. If the deploy retries (Railway is famous for retry
  loops on partial failures), the second run dies with
  `psycopg2.errors.DuplicateObject: constraint "diagnosis_archive_user_id_fkey"
  already exists`. Same applies to `create_index` at lines 26-34 unless an
  earlier 013 retry left them partially created.
- **orphan-row blocker**: 013 created `user_id UUID` with no FK, so prod
  almost certainly has rows where `diagnosis_archive.user_id` no longer maps
  to any `users.id` (deleted users, soft-deleted accounts, GDPR purges). The
  `create_foreign_key` at `019:36-46` WILL fail with
  `ForeignKeyViolation: insert or update on table "diagnosis_archive" violates
  foreign key constraint`. No pre-flight cleanup, no `NOT VALID` guard.
- rollback: `cd.yml:359-385` `rollback` job only triggers on `build-backend`/
  `build-frontend` failures — explicitly excludes `run-migrations.result ==
  'failure'`. A broken migration leaves the app deployed against a
  half-migrated DB with **zero auto-rollback**. The `downgrade()` at
  `019:48-54` is correct but never invoked by CI.
- required fixes before merge:
  1. Pre-flight: `DELETE FROM diagnosis_archive WHERE user_id NOT IN (SELECT
     id FROM users);` (or move them to a quarantine table) **before** the FK
     create. Add as a separate `op.execute(...)` step inside `upgrade()`.
  2. Make the FK creation idempotent — wrap in
     `op.execute("ALTER TABLE diagnosis_archive ADD CONSTRAINT ... NOT VALID;
     ALTER TABLE ... VALIDATE CONSTRAINT ...")` or guard with
     `IF NOT EXISTS` via `op.execute`. Same for `create_index` →
     `CREATE INDEX IF NOT EXISTS`.
  3. Move `run-migrations` to a `needs: [build-backend, build-frontend]`
     dependency that runs **before** `deploy-railway`, and gate
     `deploy-railway` on `needs.run-migrations.result == 'success'`. Today's
     order ships code that may need a schema that fails to apply.
  4. Add `rollback` trigger on `needs.run-migrations.result == 'failure'`
     and call `alembic downgrade -1` automatically.

## Olvasott fájlok

- `/home/user/AutoCognitix/backend/app/core/error_handlers.py`
- `/home/user/AutoCognitix/backend/app/core/config.py`
- `/home/user/AutoCognitix/backend/app/db/redis_cache.py`
- `/home/user/AutoCognitix/backend/app/services/diagnosis_service.py`
- `/home/user/AutoCognitix/backend/app/services/embedding_service.py`
- `/home/user/AutoCognitix/backend/app/services/nhtsa_service.py`
- `/home/user/AutoCognitix/backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py`
- `/home/user/AutoCognitix/backend/tests/test_sprint_review_audit.py`
- `/home/user/AutoCognitix/frontend/src/services/diagnosisService.ts`
- `/home/user/AutoCognitix/frontend/src/pages/DiagnosisPage.tsx`
- `/home/user/AutoCognitix/frontend/src/pages/NewDiagnosisPage.tsx`
- `/home/user/AutoCognitix/frontend/src/components/features/diagnosis/AnalysisProgress.tsx`
- `/home/user/AutoCognitix/.github/workflows/cd.yml`
