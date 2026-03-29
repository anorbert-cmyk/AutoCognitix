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

---

## Frontend Security Audit
**Auditor:** Frontend-Security Specialist
**Date:** 2026-03-29
**Branch:** `claude/ralph-loop-global-memory-lFEhR`
**Files Reviewed:**
- `frontend/src/services/api.ts`
- `frontend/src/services/authService.ts`
- `frontend/src/contexts/AuthContext.tsx`
- `frontend/src/services/diagnosisService.ts`
- `frontend/src/hooks/useStreamingDiagnosis.ts`
- `frontend/index.html`
- `frontend/nginx.conf`
- `frontend/nginx.prod.conf`
- `frontend/vite.config.ts`
- `frontend/src/config/sentry.ts`
- `frontend/src/pages/LoginPage.tsx`
- `frontend/src/pages/ResetPasswordPage.tsx`
- `frontend/src/components/ui/ErrorState.tsx`
- `frontend/src/components/ErrorBoundary.tsx`

---

### Summary

8 issues found: 0 CRITICAL, 3 HIGH, 4 MEDIUM, 1 LOW.

---

### Issue #8 — HIGH: Open Redirect — login `from` not validated

**File:** `frontend/src/pages/LoginPage.tsx`, line 21 + line 41

**Description:**
After successful login the app redirects to the path stored in `location.state.from`:

```typescript
const from = (location.state as { from?: string })?.from || '/'
navigate(from, { replace: true })
```

`location.state` is React Router internal state and cannot carry an arbitrary external URL via a normal link — however, it **can** be set programmatically by any other code in the app (e.g. a compromised dependency, injected script, or a future bad PR). No validation is performed to ensure `from` is a relative path rather than an absolute URL such as `https://evil.example.com`. If `from` contains an external URL, `react-router-dom`'s `navigate()` will attempt to navigate there (depending on version behaviour). The safe fix is to strip any protocol/host and only allow paths that start with `/`.

**Severity:** HIGH — enables phishing via post-login redirect if an attacker can set the location state.

---

### Issue #9 — HIGH: Unvalidated `error.detail` rendered as React child — potential XSS via server responses

**Files:**
- `frontend/src/contexts/AuthContext.tsx`, lines 97, 116, 144, 156
- `frontend/src/pages/DiagnosisPage.tsx`, line 166 (`toast.error(err.detail)`)
- `frontend/src/pages/ForgotPasswordPage.tsx`, line 33
- `frontend/src/pages/ResetPasswordPage.tsx`, line 129
- `frontend/src/pages/LoginPage.tsx`, line 77 (`{displayError}`)

**Description:**
The `ApiError.detail` field is populated directly from `data.detail` in the backend HTTP response body (`api.ts` line 65: `const detail = data?.detail || error.message`). This raw server-provided string is then:
1. Stored in React state as `error` in `AuthContext`
2. Rendered directly into JSX as a React text child (e.g. `{displayError}` in LoginPage, `{displayMessage}` in ErrorState)
3. Passed to `toast.error(err.detail)` in DiagnosisPage

React's default JSX rendering **does** HTML-escape string children, so a direct string-to-JSX path is not an XSS vector in the typical case. **However**, the `toast.error()` implementation in the custom ToastProvider must be verified — if it uses `innerHTML` or `dangerouslySetInnerHTML` internally, it would be a direct XSS vector. Additionally, the pattern of rendering server-controlled strings without sanitisation is fragile: if any future developer wraps `displayError` in `dangerouslySetInnerHTML` for formatting purposes, the XSS surface immediately opens up. A `sanitizeUserContent()` wrapper around all server-sourced strings before render is recommended.

**Severity:** HIGH — currently protected by React's auto-escaping, but the architecture relies on that assumption silently across multiple render sites. Any change to rendering code could introduce XSS.

---

### Issue #10 — HIGH: `LoginResponse` interface includes `refresh_token` as plain string — API contract leakage risk

**File:** `frontend/src/services/api.ts`, lines 439–444

**Description:**
The exported `LoginResponse` interface declares `refresh_token: string`. The same type exists in `authService.ts` as `AuthTokens.refresh_token`. According to the security design (comments in `authService.ts` lines 65–66 and `getRefreshToken()` line 77: "Refresh token is in httpOnly cookie, not accessible to JS"), tokens are intentionally **not** stored in JavaScript. However, `login()` in `authService.ts` returns the full `AuthTokens` object (line 133: `return response.data`), which the backend populates with `access_token` and `refresh_token` fields in the JSON body (line 132: `setTokens(response.data)`).

If the backend is currently sending tokens in the **response body** AND as httpOnly cookies, the tokens are redundantly exposed in JavaScript-accessible memory. The `AuthTokens` interface and `LoginResponse` should either not include these fields (if the backend sends only cookies and a CSRF token), or they must be explicitly scrubbed from the object after `setTokens()` is called. Currently there is no scrubbing: `authService.login()` returns the token-bearing object to its callers, who could inadvertently log or store it.

Additionally, test code in `diagnosisService.test.ts` (lines 77, 531) still references `localStorage.setItem('access_token', ...)` — suggesting the old localStorage-based token storage pattern was not fully cleaned up from tests, creating a misleading precedent.

**Severity:** HIGH — tokens in JS-accessible memory are accessible to any XSS that executes in the same session (read from response objects before GC); test code preserving the old insecure pattern risks regression.

---

### Issue #11 — MEDIUM: CSP `unsafe-inline` + `unsafe-eval` in production nginx — drastically weakens XSS protection

**File:** `frontend/nginx.prod.conf`, line 117

```nginx
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; ..."
```

**Description:**
Both `'unsafe-inline'` and `'unsafe-eval'` are present in `script-src`. These directives completely disable the XSS-blocking capability of CSP for scripts:
- `'unsafe-inline'` allows injected `<script>` blocks and `javascript:` handlers.
- `'unsafe-eval'` allows `eval()`, `Function()`, `setTimeout(string)` — common XSS payload mechanisms.

In practice this renders the CSP header ineffective as a defence-in-depth layer. The development nginx config (`nginx.conf`) has no CSP header at all. The correct approach for Vite/React production builds is to use hash-based or nonce-based CSP, since Vite generates hashed filenames but no inline scripts.

Additionally, `font-src 'self' data:` allows data-URI fonts, which can be abused in some browser exploit chains. The `connect-src 'self' https:` allows connections to any HTTPS host, which is overly permissive (should be locked to the specific backend domain via `VITE_API_URL`).

**Severity:** MEDIUM — CSP exists but provides no practical XSS protection due to `unsafe-inline`/`unsafe-eval`.

---

### Issue #12 — MEDIUM: CSRF token absent on SSE streaming POST requests when `csrfToken` is null at startup

**Files:**
- `frontend/src/services/diagnosisService.ts`, lines 369–372
- `frontend/src/services/chatService.ts`, lines 144–148

**Description:**
Both `streamDiagnosis()` and `streamChatMessage()` attach the CSRF token to fetch headers only if `getCsrfToken()` returns a non-null value:

```typescript
const csrfToken = getCsrfToken()
if (csrfToken) {
  headers['X-CSRF-Token'] = csrfToken
}
```

The in-memory `csrfToken` variable is set on login and cleared on logout. However, on a **page reload**, the user may still have valid httpOnly session cookies but `csrfToken` is `null` (it was not persisted anywhere). In this scenario:
1. The axios interceptor for regular requests also sends no CSRF token (same conditional on line 127 of `api.ts`).
2. SSE streaming POSTs proceed without any CSRF protection.
3. If the backend enforces CSRF validation on the `/auth/refresh` endpoint, the silent token refresh on 401 also fails.

The app appears to handle page-reload re-auth via `initAuth()` in `AuthContext` which calls `getCurrentUser()` — if that succeeds it sets `authenticated = true` but does **not** obtain a fresh CSRF token (no `setTokens()` call). The CSRF token is only populated after explicit login or token refresh. This creates a window where authenticated requests are sent without CSRF tokens.

**Severity:** MEDIUM — authenticated state restored after reload but CSRF protection is absent until next 401→refresh cycle.

---

### Issue #13 — MEDIUM: `error.detail` from backend may expose internal stack traces or sensitive field names in production

**Files:**
- `frontend/src/services/api.ts`, line 65: `const detail = data?.detail || error.message`
- `frontend/src/components/ui/ErrorState.tsx`, lines 232–291

**Description:**
For HTTP 400 responses, `ApiError.fromAxiosError()` passes through the raw `detail` string from the backend response body without any filtering. FastAPI's default unhandled validation errors return a `detail` array with field names, types, and input values:

```json
{"detail": [{"loc": ["body", "vehicle_year"], "msg": "value is not a valid integer", "type": "type_error.integer", "input": "...user_input..."}]}
```

If the backend passes internal error messages in `detail` (database errors, model paths, etc.), these are rendered directly in the UI. The `ErrorState` component also has a `showDetails` prop defaulting to `import.meta.env.DEV`, but several call sites pass raw `apiError.detail` as the `message` prop (bypassing `showDetails`), so it appears in production regardless.

**Severity:** MEDIUM — information disclosure; exact exposure depends on backend error handling hygiene, which is a cross-layer risk.

---

### Issue #14 — MEDIUM: `isAuthenticated()` in `authService.ts` uses `authenticated` flag OR `getCsrfToken()` — dual-state inconsistency

**File:** `frontend/src/services/authService.ts`, lines 97–99

```typescript
export function isAuthenticated(): boolean {
  return authenticated || !!getCsrfToken()
}
```

**Description:**
`isAuthenticated()` returns `true` if either `authenticated === true` OR a CSRF token is present in memory. The `authenticated` flag is set to `true` in `setTokens()` and `false` in `clearTokens()`. However, `AuthContext.isAuthenticated` is computed as `!!user` (line 179), not from `isAuthenticated()`. The two auth signals are independent:
- A user could have `authenticated=false` (e.g. after a failed refresh that called `clearTokens()`) but still have a stale CSRF token in memory, causing `isAuthenticated()` to return `true`.
- The `refreshUser()` function in `AuthContext` calls `checkAuth()` (which is `isAuthenticated()`) to guard its execution, so a stale CSRF token could allow an unauthenticated user to trigger a `getCurrentUser()` call.

This inconsistency does not directly cause a security breach but weakens the integrity of the auth state machine.

**Severity:** MEDIUM — auth state inconsistency; potential for authenticated API calls after intended logout in edge cases.

---

### Issue #15 — LOW: `console.warn('Server logout failed...')` in `authService.ts` — minor information disclosure in production

**File:** `frontend/src/services/authService.ts`, line 146

```typescript
console.warn('Server logout failed, proceeding with local logout')
```

**Description:**
The `drop_console` terser option is configured for production builds (`vite.config.ts` line 52: `drop_console: process.env.NODE_ENV === 'production'`). However, `drop_console` only drops `console.log` by default — `console.warn` and `console.error` are typically not dropped unless `pure_funcs` is configured explicitly. Depending on the terser version, this warning may survive into the production bundle and be visible in browser DevTools. While this specific message is low-sensitivity, the pattern of leaving `console.warn/error` in production may cause other more sensitive messages to leak (e.g., `useGarage.ts` uses `console.error('Jármű törlés sikertelen:', error.message)` which could expose error message details).

**Severity:** LOW — minor information leakage risk; individual messages are low-sensitivity but the pattern is a code hygiene issue.

---

### Checklist Results

| # | Check | Result | Severity | Issue |
|---|-------|--------|----------|-------|
| 1 | Token storage (JWT localStorage) | PASS | — | Tokens in httpOnly cookies; CSRF token in memory only; no localStorage usage in production code |
| 2 | XSS — `dangerouslySetInnerHTML` | PASS | — | No `dangerouslySetInnerHTML` usage found anywhere in `frontend/src` |
| 3 | XSS — server error strings rendered | PARTIAL | HIGH | Issue #9 — `error.detail` rendered without sanitisation; currently safe via React escaping but fragile |
| 4 | CSRF token on state-changing requests | PARTIAL | MEDIUM | Issue #12 — CSRF absent after page reload until next explicit login/refresh |
| 5 | Sensitive data in URL params | PASS | — | No tokens in URL params; reset password token is in `?token=` query param (acceptable for email link flows) |
| 6 | Open redirect | FAIL | HIGH | Issue #8 — `from` in login redirect unvalidated |
| 7 | CORS / API base URL | PASS | — | `VITE_API_URL` env var used; falls back to `localhost:8000`; no hardcoded production URL |
| 8 | Error messages exposed to user | PARTIAL | MEDIUM | Issue #13 — raw backend `detail` strings rendered in production |
| 9 | Dependency injection (mock vs real) | PASS | — | No mock/stub service injection leaking into production builds |
| 10 | SSE security — token in URL | PASS | — | SSE uses `fetch()` POST with `credentials: 'include'`; CSRF token in header not URL |
| 11 | CSP headers | FAIL | MEDIUM | Issue #11 — `unsafe-inline` + `unsafe-eval` in production nginx; dev nginx has no CSP |
| 12 | `access_token`/`refresh_token` in JS | PARTIAL | HIGH | Issue #10 — tokens in response body JSON accessible in memory; test code uses localStorage pattern |
| 13 | Auth state consistency | PARTIAL | MEDIUM | Issue #14 — dual `isAuthenticated` signals can diverge |
| 14 | `console.*` leakage | PARTIAL | LOW | Issue #15 — `console.warn`/`console.error` not dropped by terser `drop_console` |

---

## API Logic Audit
**Auditor:** API-Logic Specialist
**Branch:** `claude/ralph-loop-global-memory-lFEhR`
**Date:** 2026-03-29
**Files audited:**
- `backend/app/api/v1/endpoints/diagnosis.py`
- `backend/app/api/v1/endpoints/garage.py`
- `backend/app/api/v1/endpoints/vehicles.py`
- `backend/app/api/v1/endpoints/dtc_codes.py`
- `backend/app/api/v1/schemas/diagnosis.py`
- `backend/app/api/v1/schemas/garage.py`

---

### CRITICAL

#### [CRITICAL-1] DTC Create/Bulk Import — No Role Check (Missing Authorization)
**File:** `backend/app/api/v1/endpoints/dtc_codes.py` — lines 796–956
**Description:** `POST /api/v1/dtc/` (create_dtc_code) and `POST /api/v1/dtc/bulk` (bulk_import_dtc_codes) only require any authenticated user (`get_current_user_from_token`) — there is no `require_role("admin")` dependency. Any registered user can create or overwrite DTC codes in the database, including bulk-importing thousands of records.
**Impact:** Unprivileged users can poison the DTC database (integrity risk), trigger large DB writes (DoS), or overwrite production diagnostic knowledge.
**Recommendation:** Add `require_role("admin")` dependency to both endpoints.

---

#### [CRITICAL-2] SSE Streaming Endpoint — No Rate Limiting Dependency Injected
**File:** `backend/app/api/v1/endpoints/diagnosis.py` — line 834
**Description:** The docstring on `analyze_vehicle_stream` claims "Rate limiting via middleware (5 req/min)" but no rate-limit dependency (`check_diagnosis_rate_limit` or similar) is injected into the endpoint signature. The `_stream_semaphore` limits concurrent streams globally (10 max) but does not limit per-user or per-IP request frequency. The non-streaming `/analyze` endpoint (line 203) also lacks an explicit rate-limit dependency injection.
**Impact:** A single unauthenticated client can submit unlimited diagnosis requests, exhausting LLM API quota and triggering costly AI calls without throttle.
**Recommendation:** Inject `check_diagnosis_rate_limit` as a Depends on both `/analyze` and `/analyze/stream`.

---

### HIGH

#### [HIGH-1] `POST /garage/vehicles` — No Per-User Vehicle Count Limit
**File:** `backend/app/api/v1/endpoints/garage.py` — line 156; `backend/app/services/vehicle_garage_service.py` — line 85
**Description:** `POST /garage/vehicles` has no check on how many vehicles the user already owns. The service `get_vehicles` supports a default `limit=50` but no business-logic cap is enforced before creating a new vehicle.
**Impact:** DB table bloat, potential abuse of health-score/recall computation resources, no fair-use control.
**Recommendation:** Before `service.create_vehicle(...)`, call `get_vehicles` and reject with 409/422 if `total >= MAX_VEHICLES_PER_USER` (suggested: 50).

---

#### [HIGH-2] `GET /garage/vehicles` — Pagination Parameters Not Exposed; Silent Truncation
**File:** `backend/app/api/v1/endpoints/garage.py` — line 116; service line 85
**Description:** The service `get_vehicles()` accepts `skip`/`limit` (default limit=50) and applies them to the DB query, but the endpoint never exposes these as Query parameters. The response returns `total` but the caller can never page through results if a user has more than 50 vehicles — they are silently dropped.
**Impact:** Silent data truncation; `total` may exceed the number of items returned with no way for the client to request the remainder.
**Recommendation:** Expose `skip: int = Query(0, ge=0)` and `limit: int = Query(50, ge=1, le=100)` on the endpoint and pass them to the service.

---

#### [HIGH-3] `GET /garage/costs` — Pagination Not Exposed; `total_cost_huf` Incorrect on Truncation
**File:** `backend/app/api/v1/endpoints/garage.py` — line 603; service line 392
**Description:** Same pattern as HIGH-2. The service `get_costs()` defaults to `limit=50`, but the endpoint does not expose `skip`/`limit` query parameters. Additionally, `total_cost_huf` in the response is computed only over the fetched (limited) rows, not all rows in the DB.
**Impact:** Users with extensive maintenance history get silently truncated results; `total_cost_huf` is incorrect (understated) when there are more than 50 records.
**Recommendation:** Expose pagination params; compute `total_cost_huf` with a separate SUM query over all matching rows, independent of the page limit.

---

#### [HIGH-4] `GET /garage/reminders` — DB Total Count Discarded, Recomputed from In-Memory Page
**File:** `backend/app/api/v1/endpoints/garage.py` — lines 423–447
**Description:** `list_reminders` calls `reminders, _ = await service.get_reminders(...)` — the DB-provided `total` is explicitly discarded (`_`). The response then sets `total=len(enriched)`, which is the count of the in-memory (possibly truncated) page, not the true DB count.
**Impact:** Reported `total` is incorrect if the service applies an internal limit; the client cannot know whether pagination is needed.
**Recommendation:** Use the DB-provided total: `reminders, total = await service.get_reminders(...)` and pass `total` to the response.

---

#### [HIGH-5] `POST /diagnosis/quick-analyze` — No Authentication, No Rate Limiting
**File:** `backend/app/api/v1/endpoints/diagnosis.py` — line 635
**Description:** `quick_analyze` requires no authentication and no rate-limit dependency. Any anonymous client can submit up to 10 DTC codes per request and query the database at unlimited frequency.
**Impact:** Information scraping of the full DTC database and DB read amplification without any throttle.
**Recommendation:** Add a `check_search_rate_limit` or anonymous-tier rate-limit dependency.

---

#### [HIGH-6] `GET /vehicles/{make}/{model}/{year}/recalls` and `/complaints` — Unvalidated Path Parameters Forwarded to External URL
**File:** `backend/app/api/v1/endpoints/vehicles.py` — lines 655, 717
**Description:** `make` and `model` are `str` Path parameters with no `max_length` constraint or character whitelist. They are passed directly to `nhtsa_service.get_recalls(make, model, year)` and `get_complaints(make, model, year)`, which constructs outbound HTTP URLs.
**Impact:** Potential SSRF or header injection via crafted `make`/`model` values if the NHTSA client does not properly URL-encode parameters (e.g. `model = "x%0d%0aInjected-Header: value"`).
**Recommendation:** Add `max_length=100` and alphanumeric/hyphen/space validation to `make` and `model` Path parameters; verify that the NHTSA HTTP client URL-encodes all parameters.

---

### MEDIUM

#### [MEDIUM-1] `POST /diagnosis/analyze` — Unauthenticated Use Creates Orphaned Sessions
**File:** `backend/app/api/v1/endpoints/diagnosis.py` — line 203
**Description:** `analyze_vehicle` uses `get_optional_current_user` — unauthenticated requests are accepted. When `user_id=None` the service may save a diagnosis record with no owning user, and there is no IP-level rate limit for unauthenticated callers.
**Impact:** DB accumulation of anonymous diagnosis sessions, AI API abuse without account-level tracking.
**Recommendation:** Either require authentication for persistence, or apply per-IP rate limiting and TTL-based cleanup for anonymous sessions.

---

#### [MEDIUM-2] `DELETE /garage/reminders/{reminder_id}` — No Explicit 404 on Not-Found
**File:** `backend/app/api/v1/endpoints/garage.py` — line 560
**Description:** `delete_reminder` catches `VehicleGarageServiceError` and returns 400. If the reminder does not exist or belongs to another user, there is no explicit `HTTP_404_NOT_FOUND` returned. Compare with `complete_reminder` (line 521) which checks `if not reminder: raise 404`. The pattern is inconsistent.
**Impact:** Clients receive 400 Bad Request instead of semantically correct 404 for missing/foreign reminders.
**Recommendation:** Check existence/ownership before calling `delete_reminder`, returning 404 if not found (mirror the `complete_reminder` pattern).

---

#### [MEDIUM-3] `GET /dtc/search` — No Rate-Limit Dependency; `make` Has No Max Length
**File:** `backend/app/api/v1/endpoints/dtc_codes.py` — line 396
**Description:** DTC search is fully public (no auth). With `use_semantic=True` (default), every request triggers HuBERT embedding + Qdrant vector search. No rate-limit dependency is injected. The `make` Query parameter has no `max_length`.
**Impact:** Unauthenticated callers can trigger expensive ML inference (CPU/GPU) per request with no throttle.
**Recommendation:** Inject `check_search_rate_limit` as a Depends; add `max_length=100` to the `make` Query parameter.

---

#### [MEDIUM-4] `GET /dtc/{code}/related` — No Auth/Rate Limit; Sync neomodel Call Blocks Event Loop
**File:** `backend/app/api/v1/endpoints/dtc_codes.py` — line 687
**Description:** The endpoint is fully public with no rate limiting. Additionally, the neomodel call `DTCNode.nodes.get_or_none(code=code)` and subsequent `.related_to.all()` (lines 737–741) are **synchronous** neomodel ORM calls inside an async handler, blocking the event loop for the duration of the Neo4j query.
**Impact:** Unauthenticated graph enumeration; event loop blocking under concurrent load.
**Recommendation:** Add rate limiting; replace synchronous neomodel calls with the async graph helper (`_get_neo4j_relationships`) already used elsewhere.

---

#### [MEDIUM-5] `GET /vehicles/makes` and `/models` — `search` Parameter Has No Max Length
**File:** `backend/app/api/v1/endpoints/vehicles.py` — lines 304, 382
**Description:** The `search` query parameter has `min_length=1` but no `max_length`. It is passed to `vehicle_service.get_all_makes(search=...)` and `get_models_for_make(search=...)` for Neo4j Cypher queries. Oversized or specially crafted strings reach the graph layer.
**Impact:** Unnecessary graph traversal; potential Cypher injection if parameterization is not used in the service layer.
**Recommendation:** Add `max_length=100` to the `search` Query parameter; verify the Neo4j service uses parameterized Cypher (`$param`) and not string interpolation.

---

#### [MEDIUM-6] DTC Detail Cache Key Collision — `include_graph` Flag Not Propagated to Redis
**File:** `backend/app/api/v1/endpoints/dtc_codes.py` — lines 641–666
**Description:** The local cache key is built as `f"{code}:{include_graph}"` (e.g. `P0101:True`), but `_cache_dtc_detail(cache_key, result.model_dump())` passes this to `cache.set_dtc_code(code, ...)`, which likely uses only the bare `code` as the Redis key, ignoring the `include_graph` suffix. A response cached with `include_graph=True` (full Neo4j enrichment) will be returned for a subsequent request with `include_graph=False`, and vice versa.
**Impact:** Incorrect (over- or under-populated) data served from cache depending on which request was cached first.
**Recommendation:** Ensure the Redis key used in `set_dtc_code` / `get_dtc_code` includes the full compound key string; or pass `cache_key` directly as the Redis key.

---

#### [MEDIUM-7] `POST /garage/costs` and `POST /garage/reminders` — No Ownership Check on `vehicle_id`
**File:** `backend/app/api/v1/endpoints/garage.py` — lines 471, 652; schema lines 132, 183
**Description:** Both `create_reminder` and `create_cost` pass a user-supplied `data.vehicle_id` string to the service without first verifying that the vehicle belongs to `current_user`. An authenticated user can record costs or reminders against another user's `vehicle_id`.
**Impact:** Cross-user data pollution — a malicious authenticated user can append records to vehicles they do not own.
**Recommendation:** Call `_get_vehicle_or_404(data.vehicle_id, str(current_user.id), db)` before creating the cost/reminder to enforce vehicle ownership.

---

### LOW

#### [LOW-1] `DiagnosisRequest.dtc_codes` — Individual DTC String Length Not Validated in Schema
**File:** `backend/app/api/v1/schemas/diagnosis.py` — line 24
**Description:** `dtc_codes: List[str]` validates the list length (1–20) but not the length or format of each individual DTC string. A request with `dtc_codes=["AAAAAAAAAAAAAAAAAAAAAAAA"]` reaches the service layer.
**Recommendation:** Add a `@field_validator("dtc_codes")` enforcing `max_length=10` per code and matching the pattern `^[PBCU]\d{4}$`.

---

#### [LOW-2] `GET /garage/vehicles/{vehicle_id}/recalls` — NHTSA Errors Silently Swallowed
**File:** `backend/app/api/v1/endpoints/garage.py` — lines 722–726
**Description:** A bare `except Exception` returns `[]` for all failures, including NHTSA service errors. Other NHTSA endpoints (`vehicles.py`) correctly raise HTTP 502. Clients cannot distinguish "no recalls found" from "recall lookup failed".
**Impact:** Silent failures; clients may incorrectly infer a clean recall history.
**Recommendation:** Re-raise `NHTSAError` as 502 (matching the pattern in `vehicles.py`); swallow only non-critical timeouts.

---

#### [LOW-3] `GET /diagnosis/history/list` — `date_from > date_to` Not Validated
**File:** `backend/app/api/v1/endpoints/diagnosis.py` — line 380
**Description:** `date_from` and `date_to` are accepted independently. Passing `date_from` later than `date_to` silently returns zero results rather than a 422 validation error.
**Recommendation:** Add a check (inline or in `DiagnosisHistoryFilter`) enforcing `date_from <= date_to` when both are provided.

---

#### [LOW-4] `POST /diagnosis/analyze` — Returns HTTP 201 for Detected Duplicate Submissions
**File:** `backend/app/api/v1/endpoints/diagnosis.py` — lines 237–243
**Description:** When a duplicate submission is detected, the endpoint returns `HTTP 201 Created` with an `X-Duplicate-Of` header. Semantically, a resource was not created; 201 is misleading.
**Recommendation:** Return `HTTP_200_OK` (or `HTTP_303_SEE_OTHER` + `Location: /api/v1/diagnosis/{duplicate_id}`) for detected duplicates.

---

#### [LOW-5] `GET /vehicles/{make}/{model}/common-issues` — No Result Limit; Unbounded Neo4j Response
**File:** `backend/app/api/v1/endpoints/vehicles.py` — line 779
**Description:** Fully public endpoint with no `limit` query parameter. If the Neo4j service returns a large result set, the entire list is serialized without bound.
**Recommendation:** Add a `limit: int = Query(20, ge=1, le=100)` parameter and pass it to the service layer.

---

### Summary Table

| ID | Severity | Endpoint | Issue |
|----|----------|----------|-------|
| CRITICAL-1 | CRITICAL | `POST /dtc/` + `POST /dtc/bulk` | No admin role check — any user can write/overwrite DTC DB |
| CRITICAL-2 | CRITICAL | `POST /diagnosis/analyze/stream` | Rate limiting claimed in docstring but not injected as dependency |
| HIGH-1 | HIGH | `POST /garage/vehicles` | No per-user vehicle count limit |
| HIGH-2 | HIGH | `GET /garage/vehicles` | Pagination params not exposed; silent 50-vehicle truncation |
| HIGH-3 | HIGH | `GET /garage/costs` | Pagination not exposed; `total_cost_huf` computed on truncated page only |
| HIGH-4 | HIGH | `GET /garage/reminders` | DB total count discarded; recomputed from in-memory paginated list |
| HIGH-5 | HIGH | `POST /diagnosis/quick-analyze` | No auth, no rate limit |
| HIGH-6 | HIGH | `GET /vehicles/{make}/{model}/{year}/recalls+complaints` | Unvalidated path params forwarded to external HTTP URL |
| MEDIUM-1 | MEDIUM | `POST /diagnosis/analyze` | Unauthenticated use, no IP rate limit, orphaned sessions |
| MEDIUM-2 | MEDIUM | `DELETE /garage/reminders/{id}` | Missing explicit 404; inconsistent with complete_reminder pattern |
| MEDIUM-3 | MEDIUM | `GET /dtc/search` | No rate-limit dependency injected; `make` param has no max_length |
| MEDIUM-4 | MEDIUM | `GET /dtc/{code}/related` | No auth/rate limit; sync neomodel calls block event loop |
| MEDIUM-5 | MEDIUM | `GET /vehicles/makes` + `/models` | `search` param has no max_length; Cypher injection risk |
| MEDIUM-6 | MEDIUM | `GET /dtc/{code}` | Cache key `include_graph` flag not propagated to Redis store |
| MEDIUM-7 | MEDIUM | `POST /garage/costs` + `POST /garage/reminders` | No ownership check on user-supplied `vehicle_id` |
| LOW-1 | LOW | `POST /diagnosis/analyze` | Individual DTC string length/format not validated in schema |
| LOW-2 | LOW | `GET /garage/vehicles/{id}/recalls` | NHTSA errors silently swallowed, returns empty list |
| LOW-3 | LOW | `GET /diagnosis/history/list` | date_from > date_to not validated |
| LOW-4 | LOW | `POST /diagnosis/analyze` | HTTP 201 returned for duplicate submissions (semantic error) |
| LOW-5 | LOW | `GET /vehicles/{make}/{model}/common-issues` | No result limit; unbounded Neo4j traversal |
