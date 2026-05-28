# Wave2 Refix — Operational Lead

## A) Migration 019 deploy

- **severity: HIGH**
- **finding: 019_fix_diagnosis_archive_indexes_and_fk.py:41-47** — `DELETE FROM diagnosis_archive WHERE user_id NOT IN (SELECT id FROM users)` runs as a single statement on a potentially huge archive. PostgreSQL takes a **ROW EXCLUSIVE lock** on `diagnosis_archive` plus a **per-row lock** for each deleted row. With 100k+ orphan rows this:
  - blocks any concurrent INSERT/UPDATE on the same rows (writers wait)
  - bloats WAL; on Railway's small Postgres tier (single connection, small `maintenance_work_mem`) the txn can stall the whole migration job
  - the `NOT IN (SELECT id FROM users)` correlated form is **slow on large tables** (no anti-join optimization if `user_id` is nullable) — should be `LEFT JOIN … WHERE u.id IS NULL` or `WHERE NOT EXISTS (…)`. Currently no batching, no `LIMIT`, no progress logging.
- **finding: 019…py:53-64** — `op.create_index(..., if_not_exists=True)` runs **without** `postgresql_concurrently=True`. Comment on line 49-52 claims CONCURRENTLY "needs autocommit block" and decides to "use plain create_index here". On a multi-million-row archive this takes an **ACCESS EXCLUSIVE lock**, blocking ALL reads/writes for the index build duration. The comment's "archive table is small in practice" assumption is unverified for production.

- **severity: HIGH**
- **finding: cd.yml:232-256** — `run-migrations` job has `continue-on-error: false` (good) but **no rollback path**. If 019 fails mid-way (e.g. VALIDATE CONSTRAINT discovers a race-condition orphan inserted between DELETE and VALIDATE), the job exits non-zero — and the deployment is left in a **partially-migrated state**: backend already deployed via `deploy-railway` (line 193, runs BEFORE migrations on line 235) pointing to a schema that lacks the FK or has half the indexes.
- **finding: cd.yml:359-385** — The `rollback` job only triggers when `build-backend.result == 'failure' || build-frontend.result == 'failure'` (line 364). Migration failures are NOT in the condition → **no automated rollback when migration fails**. Manual `alembic downgrade -1` is theoretically possible (downgrade() at 019…py:81-90 is well-formed: drops FK first, then indexes), but no runbook exists.
- **finding: cd.yml ordering bug** — `deploy-railway` runs at line 193 before `run-migrations` at line 232. The new backend (which expects FK + indexes from 019) is **live and serving traffic** while migration is still pending. Diagnosis writes during this window may violate new FK if user_id orphan-purge hasn't happened yet.

## B) Sentry quota

- **severity: MEDIUM**
- **finding: error_handlers.py:312-343** — `_capture_to_sentry` has **no rate limiting, no sampling, no dedup**. Sentry default `traces_sample_rate` doesn't apply to `capture_exception` (only to performance). Three handlers call it on every 5xx (line 158, 294, 374). NHTSA outage scenario: `httpx.TimeoutException` → 504 → captured at line 374 via generic handler. NHTSA outage of 1 hour at 100 req/min = **6000 events**, well above Sentry Free tier (5k/month) — burns the whole monthly quota in one outage.
- **finding: error_handlers.py:542-593** — `setup_httpx_exception_handler` returns 502/504 but **does not call `_capture_to_sentry`**. So NHTSA timeouts go through the httpx handler (no Sentry) BUT if httpx error escapes (e.g. wrapped in custom exception), it bubbles to `generic_exception_handler:374` which DOES capture. Inconsistency: some external API failures captured, some not — operator can't reliably alert on "NHTSA degraded".
- **No `before_send` hook visible** to dedup duplicate exceptions; identical stack traces from a hot loop will each consume a quota event.

## C) HUBERT docs operator gap

- **severity: MEDIUM**
- **finding: docs/HUNGARIAN_NLP.md:1-60** — Architecture section explains the embedding pipeline but **never mentions `HUBERT_REVISION`**. `grep` of full file: `HUBERT_REVISION` not present anywhere in docs (only `HUBERT_MODEL` at line 59 and 387). Yet `.env.example:131-136` and `.env.railway.example:82-85` both define `HUBERT_REVISION=main` (with `main` as the default tag, which is mutable on HuggingFace).
- **Silent embedding mismatch risk**: operator bumps `HUBERT_REVISION` to a new commit hash → 768-dim vectors shift in vector space → existing 35,000+ Qdrant vectors are now in the **old embedding space**, but new queries embed in the **new space** → cosine similarity meaningless → silently degraded RAG results, no error raised.
- **Missing runbook**: no section like "If you bump HUBERT_REVISION you MUST: (1) re-embed all Qdrant points with the new revision, (2) atomically swap the collection alias, (3) verify dim=768 unchanged". Default `main` is mutable → next HuggingFace upstream push silently changes embeddings on next container restart.

## Olvasott fájlok
- /home/user/AutoCognitix/.github/workflows/cd.yml
- /home/user/AutoCognitix/backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py
- /home/user/AutoCognitix/backend/app/core/error_handlers.py
- /home/user/AutoCognitix/docs/HUNGARIAN_NLP.md (lines 1-60)
- /home/user/AutoCognitix/.env.example, /home/user/AutoCognitix/.env.railway.example
