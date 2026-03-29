## SSE Streaming Logic Audit
**Auditor:** SSE-Logic Specialist
**Date:** 2026-03-29
**Files Reviewed:**
- `backend/app/services/streaming_service.py`
- `backend/app/api/v1/endpoints/diagnosis.py` (streaming route)
- `frontend/src/hooks/useStreamingDiagnosis.ts`
- `frontend/src/services/diagnosisService.ts`

---

### Summary

7 issues found: 2 HIGH, 4 MEDIUM, 1 LOW.

---

### Issue #1 — HIGH: Timeout check only between events, not mid-pipeline

**File:** `backend/app/api/v1/endpoints/diagnosis.py`, lines 1127–1152

**Description:**
The `_timeout_wrapper()` enforces the 300-second deadline by checking `loop.time() > deadline` **between yielded events**. However, the RAG pipeline step (`_run_rag_pipeline`) is a single `await` call with no internal yields. If the LLM takes longer than `STREAM_TIMEOUT_SECONDS`, the timeout check never fires because the pipeline never yields an event during that await. The deadline is only evaluated after `yield event`, meaning a slow LLM call can block indefinitely regardless of the configured timeout.

**Reproducer:** LLM takes > 300 seconds to respond → generator is stuck inside `_run_rag_pipeline`, `_timeout_wrapper` cannot interrupt it since no event is yielded.

**Checklist item:** #5 (Timeout)

---

### Issue #2 — HIGH: `_stream_semaphore` is a module-level global, not event-loop-aware

**File:** `backend/app/api/v1/endpoints/diagnosis.py`, line 61

```python
_stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
```

**Description:**
`asyncio.Semaphore` is created at **module import time**, before the event loop exists (or on the wrong event loop in test environments). Under ASGI servers that create new event loops (e.g. Hypercorn, uvicorn with `--workers > 1` + `--loop`), the semaphore is bound to a different loop and raises `RuntimeError: Task attached to different loop`. In uvicorn single-worker mode this works, but it is fragile. The semaphore should be created lazily (e.g. via a dependency or `startup` event).

**Checklist item:** #6 (Concurrent streams)

---

### Issue #3 — MEDIUM: `analysis` event sends `stage`/`message` fields, hook reads `text` field

**File (backend):** `backend/app/api/v1/endpoints/diagnosis.py`, lines 954–999
**File (frontend):** `frontend/src/hooks/useStreamingDiagnosis.ts`, line 73

**Description:**
The backend emits `analysis` events with `data: {"stage": "rag_start", "message": "..."}` and `data: {"stage": "rag_complete", "message": "..."}`. The frontend `onAnalysis` handler reads `eventData.text`:

```typescript
const chunk = typeof eventData.text === 'string' ? eventData.text : ''
```

Since `text` is never present in any `analysis` event payload, `chunk` is always `''`. The conditional `if (chunk)` then prevents any state update. The result: `fullText` is **never populated** from the analysis events, and `chunks` remains empty throughout streaming. Any UI that renders `fullText` to show LLM output will display nothing.

This is also confirmed by `streaming_service.py` (`stream_result_as_chunks`) which uses `chunk` as the field name, not `text` — though that service is not used in the main streaming route.

**Checklist item:** #9 (fullText accumulation)

---

### Issue #4 — MEDIUM: No reconnection logic; frontend has no automatic retry

**File:** `frontend/src/services/diagnosisService.ts`, `frontend/src/hooks/useStreamingDiagnosis.ts`

**Description:**
If the SSE connection drops mid-stream (network flap, server restart, proxy timeout), the frontend `fetch`-based client **silently stops**. The `reader.read()` loop resolves with `done: true` on connection close — no error is raised and no `onError` callback fires. The hook ends in state `{ isStreaming: false, isDone: false, error: null }` — a "stuck" state with no feedback to the user and no reconnect attempt. Standard `EventSource` would automatically reconnect using `Last-Event-ID`; this custom fetch-based implementation has no equivalent mechanism.

**Note:** There is no retry loop risk since there is no reconnection at all, but the user experience consequence is an invisible failure.

**Checklist item:** #4 (Reconnection)

---

### Issue #5 — MEDIUM: `streamDiagnosisGenerator` has a memory/cleanup issue when caller breaks early

**File:** `frontend/src/hooks/useStreamingDiagnosis.ts`, lines 154–223

**Description:**
The `streamDiagnosisGenerator` uses a closure-based queue and a hanging `Promise<void>` to bridge callbacks into an async generator. If the caller breaks out of `for await` early (e.g. component unmounts), the `finally` block calls `controller.abort()` — correct. However, the `streamDiagnosis` callbacks (`onAnalysis`, `onComplete`, `onError`) still hold references to `queue`, `resolve`, and `finished` via closure. After abort, the fetch may still be in-flight for a brief window. When `onAnalysis` fires on the aborted-but-not-yet-cancelled network response, it pushes to `queue` and calls `notify()`, which resolves the orphaned `Promise<void>` — but since the generator has already returned, nobody consumes the queue. The closure objects remain in memory until the next GC cycle. This is not a true leak (GC will collect), but in high-frequency usage the orphaned callbacks accumulate between abort and actual network cancellation.

**Checklist item:** #2 (Memory leak)

---

### Issue #6 — MEDIUM: `complete` event payload is a summary dict, not the full diagnosis result

**File (backend):** `backend/app/api/v1/endpoints/diagnosis.py`, lines 1021–1037
**File (frontend):** `frontend/src/hooks/useStreamingDiagnosis.ts`, lines 83–90

**Description:**
The `onComplete` callback receives `eventData` (the `data` field of the `complete` SSE event) and stores it as `fullResult`:

```typescript
onComplete: (eventData) => {
  setState((prev) => ({ ...prev, fullResult: eventData }))
}
```

The backend `complete` event only sends a summary:
```python
data={
    "diagnosis_id": str(diagnosis_id),
    "confidence_score": ...,
    "urgency_level": ...,
    "probable_causes_count": ...,
    "repairs_count": ...,
    ...
}
```

Any consumer of `fullResult` expecting the full `DiagnosisResponse` structure (with `probable_causes`, `recommended_repairs`, arrays etc.) will find it absent. The structured result is assembled and saved to DB (`_save_diagnosis_session`) but **never sent over the SSE stream**. To get the full result, the consumer must make a separate `GET /diagnosis/{id}` call. This is undocumented and mismatched with the `StreamChunk.full_result` interface exported from the hook.

**Checklist item:** #9 (fullText accumulation / complete event contract)

---

### Issue #7 — LOW: `event:` line in SSE format is ignored by `parseSSEEvents`

**File:** `frontend/src/services/diagnosisService.ts`, lines 279–309

**Description:**
The backend formats events with a named event line:
```
event: analysis
data: {"event_type": "analysis", ...}

```

The `parseSSEEvents` function only extracts `data:` lines and discards the `event:` line entirely. Routing is done via `event.event_type` inside the JSON body — which works correctly in practice. However, if a future event omits `event_type` from the JSON body (relying on the SSE `event:` field), it would be silently dropped. This is a robustness concern rather than a current bug, but it creates inconsistency: the `event:` field in the SSE envelope is written but never read.

**Checklist item:** #7 (SSE format)

---

### Checklist Results

| # | Check | Result | Issue |
|---|-------|--------|-------|
| 1 | Connection cleanup / `CancelledError` | PASS | `asyncio.CancelledError` caught in `generate_events` (line 1101), semaphore released in `finally` |
| 2 | Memory leak on stream abort | MEDIUM | Issue #5 — orphaned callbacks between abort and network cancel |
| 3 | Error propagation backend→SSE→frontend | PASS | Error events formatted and dispatched; `onError` callback triggered |
| 4 | Reconnection logic | MEDIUM | Issue #4 — no reconnect; silent failure on connection drop |
| 5 | Timeout | HIGH | Issue #1 — timeout only fires between events; LLM await not interruptible |
| 6 | Concurrent streams | HIGH | Issue #2 — module-level `asyncio.Semaphore` created before event loop |
| 7 | SSE format | LOW | Issue #7 — `event:` line written but not parsed |
| 8 | AbortController | PASS | `controller.abort()` called in `stopStreaming()` and generator `finally` block |
| 9 | fullText accumulation | MEDIUM | Issue #3 — `analysis` event uses `stage`/`message`; hook reads `text` → always empty |
| 10 | Type safety | MEDIUM | Issue #6 — `complete` event sends summary only; `fullResult` contract broken vs `StreamChunk` |

## Database Audit
**Auditor:** Database Specialist
**Date:** 2026-03-29
**Files Reviewed:**
- `backend/app/db/postgres/models.py`
- `backend/alembic/versions/001_initial_schema.py` through `017_add_password_reset_tokens.py` (all 18 migrations)
- `backend/app/db/postgres/repositories.py`
- `backend/app/db/postgres/session.py`

---

### Summary

10 issues found: 1 CRITICAL, 4 HIGH, 4 MEDIUM, 1 LOW.

---

### Issue DB-1 — CRITICAL: `DiagnosisSession.user_id` has no `ondelete` cascade — orphan risk

**File:** `backend/app/db/postgres/models.py`, line 236
**Migration:** `001_initial_schema.py`, line 109

**Description:**
`DiagnosisSession.user_id` is a nullable FK to `users.id` with **no `ondelete` rule**:

```python
user_id: Mapped[Optional[str]] = mapped_column(Uuid, ForeignKey("users.id"), index=True)
```

Migration 001 also omits `ondelete`:
```python
sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True)
```

When a user is deleted, PostgreSQL defaults to `RESTRICT` (raises an error) or `NO ACTION` (deferred RESTRICT). Since `user_id` is nullable, the intent appears to be `SET NULL` — anonymous sessions after user deletion. As written:
- If the application calls `db.delete(user)` directly, the constraint fires and raises `IntegrityError`.
- If using `CASCADE` from `UserRepository.deactivate()` (soft-only, never deletes), the DELETE path is currently skipped — but any future hard-delete will fail unexpectedly.
- `DiagnosisArchive.user_id` (line 277) has `ondelete="CASCADE"`, which is inconsistent: archiving works cleanly, but the live `diagnosis_sessions` table does not.

**Contrast:** Every other FK to `users.id` uses explicit `ondelete="CASCADE"` (lines 80, 277, 762, 801, 831).

**Risk:** `IntegrityError` on user deletion; silent orphan records if DELETE is bypassed.

**Recommendation:** Add `ondelete="SET NULL"` to `DiagnosisSession.user_id` FK and a corresponding migration `ALTER TABLE diagnosis_sessions ALTER COLUMN user_id DROP NOT NULL` (already nullable) + `ADD CONSTRAINT ... ON DELETE SET NULL`.

---

### Issue DB-2 — HIGH: `PasswordResetToken.expires_at` has no index — TTL lookups do full table scan

**File:** `backend/app/db/postgres/models.py`, line 83
**Migration:** `017_add_password_reset_tokens.py`

**Description:**
`expires_at` is `NOT NULL` but has no index:

```python
expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

Migration 017 creates only `ix_password_reset_tokens_user_id`. Any query of the form:
```sql
SELECT * FROM password_reset_tokens WHERE expires_at < NOW() AND used = false
```
(e.g., scheduled cleanup of expired tokens or validity check on presentation) requires a full table scan. As the table grows, this degrades linearly.

Additionally, **there is no automatic cleanup mechanism** for expired tokens. Rows where `expires_at < NOW()` accumulate indefinitely; the table will grow unbounded unless a cron/background task prunes them.

**Recommendation:**
1. Add `index=True` to `expires_at` in the model and a corresponding migration `CREATE INDEX`.
2. Add a periodic background task (or PostgreSQL `pg_cron`) to `DELETE FROM password_reset_tokens WHERE expires_at < NOW()`.

---

### Issue DB-3 — HIGH: Dual password-reset storage — `users` columns vs `password_reset_tokens` table are redundant and inconsistent

**File:** `backend/app/db/postgres/models.py`, lines 52–53 (`User` model) vs lines 73–89 (`PasswordResetToken` model)
**Migration:** `011_add_user_security_columns.py` (adds columns to `users`) and `017_add_password_reset_tokens.py` (creates dedicated table)

**Description:**
The schema maintains **two separate password-reset mechanisms** simultaneously:

**Mechanism A — columns on `users`:**
```python
password_reset_token: Mapped[Optional[str]] = mapped_column(String(255))
password_reset_expires: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
```
`UserRepository.set_password_reset_token()` (line 191) writes to these columns. `UserRepository.update_password()` (line 178) clears them.

**Mechanism B — dedicated `password_reset_tokens` table (Sprint 10):**
A proper hashed, single-use token table with `token_hash`, `expires_at`, `used` boolean, and FK to `users.id`.

Both mechanisms exist in parallel. Migration 017 created the table but there is no migration or code change removing the deprecated `users.password_reset_token` / `users.password_reset_expires` columns or redirecting `UserRepository` to use the new table. The old plaintext-token columns remain in production schema and in the ORM model, creating:
- Security risk: the old columns store the token **unhashed** (`String(255)` holding a raw JWT), while the new table stores `token_hash`.
- Confusion: callers may use either path; no single source of truth.
- Wasted storage: nullable columns on every user row.

**Recommendation:** Deprecate and drop `users.password_reset_token` + `users.password_reset_expires` in a follow-up migration. Redirect `UserRepository` methods to use `PasswordResetToken` exclusively.

---

### Issue DB-4 — HIGH: `get_dtc_frequency()` raw SQL always uses `user_id` filter even when `user_id is None`

**File:** `backend/app/db/postgres/repositories.py`, lines 688–714

**Description:**
The method `get_dtc_frequency()` builds two separate raw SQL strings:

```python
query = text("""
    SELECT code, COUNT(*) as count
    FROM diagnosis_sessions, unnest(dtc_codes) as code
    WHERE user_id = :user_id AND is_deleted = false
    ...
""")

if user_id:
    result = await self.db.execute(query, {"user_id": str(user_id), "limit": limit})
else:
    query = text("""
        SELECT code, COUNT(*) as count
        FROM diagnosis_sessions, unnest(dtc_codes) as code
        WHERE is_deleted = false
        ...
    """)
    result = await self.db.execute(query, {"limit": limit})
```

The first `query` (with `user_id = :user_id`) is constructed **unconditionally** but only executed if `user_id` is truthy. This is not a runtime bug but is fragile: if the branching logic changes, the parameterized `user_id` query could execute with `user_id = None`, binding `NULL` to the parameter, causing the `WHERE user_id = NULL` condition to match zero rows (NULL ≠ NULL in SQL) — silent wrong result.

More critically, the conditions list `conditions` (lines 687–689) is built but **never used** — it's dead code:

```python
conditions: List[ColumnElement[bool]] = [DiagnosisSession.is_deleted.is_(False)]
if user_id:
    conditions.append(DiagnosisSession.user_id == user_id)
```

These ORM conditions are assembled and then abandoned; the method switches to raw `text()` immediately after. This dead code misleads future maintainers into thinking the ORM query path is operative.

**Recommendation:** Remove the dead `conditions` list. Use a single parameterized query with conditional `user_id` binding, or consolidate into one ORM query.

---

### Issue DB-5 — HIGH: `MaintenanceCost` has no `updated_at` column — audit trail incomplete

**File:** `backend/app/db/postgres/models.py`, lines 821–845
**Migration:** `016_add_garage_tables.py`, lines 84–119

**Description:**
`MaintenanceCost` has `created_at` but no `updated_at`:

```python
created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
# No updated_at
```

Migration 016 mirrors this omission. Unlike `UserVehicle` and `MaintenanceReminder` (both of which have `updated_at`), cost records have no mutation timestamp. If a cost record is corrected (wrong amount, wrong date), there is no way to tell when the correction was made. For financial/audit data, this is a notable gap.

**Recommendation:** Add `updated_at` column to `maintenance_costs` with `server_default=func.now(), onupdate=func.now()` and a corresponding `ALTER TABLE` migration.

---

### Issue DB-6 — MEDIUM: `015_merge_heads` `down_revision` tuple retains `branch_labels = None` — diverges from lgtm-suppress convention

**File:** `backend/alembic/versions/015_merge_heads.py`, lines 14–16

**Description:**
Post-Sprint 9 convention (documented in CLAUDE.md) is to suppress CodeQL `py/unused-global-variable` warnings with `# lgtm[py/unused-global-variable]` on revision identifiers. Migration 015 still uses the old style without suppression comments:

```python
revision: str = '015_merge_heads'
down_revision: Union[str, None] = ('013_newsletter_subscribers', '014_add_diagnosis_dedup_index')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
```

Migrations 016 and 017 correctly omit `branch_labels`/`depends_on` and use `# lgtm` suppressions. Migration 015 also keeps `branch_labels` and `depends_on = None` which CLAUDE.md says are "safely removable". Minor CI/CodeQL noise risk.

---

### Issue DB-7 — MEDIUM: `diagnosis_archive` downgrade drops table without dropping `ix_diagnosis_archive_archived_at` index first

**File:** `backend/alembic/versions/013_add_diagnosis_archive_table.py`, lines 43–46

**Description:**
The `upgrade()` creates both the table and an explicit index:
```python
op.create_index("ix_diagnosis_archive_archived_at", "diagnosis_archive", ["archived_at"])
```

The `downgrade()` only calls:
```python
op.drop_table("diagnosis_archive")
```

While PostgreSQL implicitly drops indexes when a table is dropped, Alembic best practice (as demonstrated by migrations 003, 005, 016) is to drop indexes explicitly before the table in `downgrade()`. The asymmetry between `upgrade` and `downgrade` is inconsistent with the rest of the codebase and can cause issues if `op.drop_table` is ever replaced with a conditional/idempotent drop.

---

### Issue DB-8 — MEDIUM: `newsletter_subscribers.status` and `source` columns are NOT NULL in migration but nullable in model

**File:** `backend/app/db/postgres/models.py`, lines 99–104
**Migration:** `013_newsletter_subscribers.py`, lines 27–29

**Description:**
The migration creates `status` and `source` with explicit `nullable=False`:
```python
sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
```
But in `013_newsletter_subscribers.py`, `source` has no explicit `nullable=False`:
```python
sa.Column("source", sa.String(50), server_default="landing_page"),
```
The ORM model also defines `source` without `nullable=False`:
```python
source: Mapped[str] = mapped_column(String(50), default="landing_page")
```
In SQLAlchemy `Mapped[str]` (non-Optional) implies NOT NULL at the ORM level but the migration does not enforce it at the DB level. If direct SQL inserts or external tools write NULL to `source`, the ORM mapping will silently return empty strings. Minor schema drift.

---

### Issue DB-9 — MEDIUM: `user_vehicles` and `maintenance_reminders` lack composite uniqueness constraint

**File:** `backend/app/db/postgres/models.py`, lines 755–789, 791–819
**Migration:** `016_add_garage_tables.py`

**Description:**
A user can register the same VIN multiple times in `user_vehicles` — there is no `UniqueConstraint("user_id", "vin")`. For users who accidentally submit the same vehicle twice, duplicate entries accumulate silently. The `vin` index alone is not unique.

Similarly, `maintenance_reminders` has no uniqueness constraint preventing duplicate reminders of the same type for the same vehicle (e.g., two identical oil-change reminders for the same car). While this may be intentional for flexibility, the absence of any uniqueness guard means accidental duplicates go undetected.

**Recommendation:** Consider `UniqueConstraint("user_id", "vin")` on `user_vehicles` (where `vin IS NOT NULL`) as a partial unique index, to prevent duplicate VIN registrations per user.

---

### Issue DB-10 — LOW: `VehicleRecall` and `VehicleComplaint` lack composite index on `(make, model, model_year)` in the ORM model definition

**File:** `backend/app/db/postgres/models.py`, lines 518–548, 556–596

**Description:**
The composite indexes `ix_vehicle_recalls_make_model_year` and `ix_vehicle_complaints_make_model_year` are created in migration 003 via raw `op.execute()`. These indexes are invisible to SQLAlchemy's metadata introspection tools (`alembic check`, `--autogenerate`) because they are not reflected in `__table_args__`. This is a documentation/maintainability concern — a developer running `alembic revision --autogenerate` in a clean environment would see these indexes as "missing" and might drop them.

**Recommendation:** Add `Index("ix_vehicle_recalls_make_model_year", "make", "model", "model_year")` to `VehicleRecall.__table_args__` and the equivalent for `VehicleComplaint`.

---

### Migration Chain Verification

Full chain verified — no orphan migrations:

```
001 → 002 → 003 → 004 → 005 → 006 → 007 → 008 → 009 → 010 → 011 → 012
                                                                      ↓
015_merge_heads ← 013_newsletter_subscribers ← 012
015_merge_heads ← 014_add_diagnosis_dedup_index ← 013_add_diagnosis_archive_table ← 012
     ↓
    016 → 017
```

The merge-head pattern in 015 is correct. `down_revision` is a tuple pointing to both `013_newsletter_subscribers` and `014_add_diagnosis_dedup_index` (which itself chains from `013_add_diagnosis_archive_table`). Both `013_*` files independently set `down_revision = "012_epa_vehicles"` — the branch split and merge are valid.

---

### Checklist Results

| # | Check | Result | Issue |
|---|-------|--------|-------|
| 1 | Migration chain integrity | PASS | No orphan migrations; branch/merge correct |
| 2 | Missing FK indexes | PASS | All FKs indexed (model and migration) |
| 3 | Cascade rules / orphan records | CRITICAL | DB-1: `DiagnosisSession.user_id` has no `ondelete` |
| 4 | `PasswordResetToken.expires_at` indexed | HIGH | DB-2: No index; no cleanup mechanism |
| 5 | Nullable columns correctness | MEDIUM | DB-8: `newsletter_subscribers.source` nullable drift |
| 6 | Transaction boundaries | PASS | Repository uses `flush()` consistently; `get_db()` commits on success, rollbacks on all exception types |
| 7 | Connection pool / session leak | PASS | `get_db()` uses `try/finally` with `session.close()`; `check_database_connection()` uses `async with` (auto-close) |
| 8 | UUID vs Integer PK consistency | PASS | All user-facing entities use UUID; internal/reference tables use Integer — consistent within each domain |
| 9 | Timezone-aware timestamps | PASS | All `DateTime` columns use `timezone=True`; no naive datetimes in model definitions |
| 10 | N+1 queries / lazy loading | PASS | All ORM queries use explicit `select()`; no relationship attribute access in async context without explicit load |
| 11 | Dual password-reset mechanism | HIGH | DB-3: Plaintext token on `users` + hashed token table coexist |
| 12 | Dead code in repository | HIGH | DB-4: `conditions` list in `get_dtc_frequency()` assembled but never used |
| 13 | `MaintenanceCost.updated_at` | HIGH | DB-5: Missing audit timestamp on financial records |
| 14 | Downgrade symmetry | MEDIUM | DB-7: `diagnosis_archive` downgrade drops table without explicit index drop |
| 15 | Composite uniqueness constraints | MEDIUM | DB-9: No unique VIN-per-user constraint in `user_vehicles` |
| 16 | ORM metadata vs raw index sync | LOW | DB-10: Composite indexes on recall/complaint not reflected in `__table_args__` |
