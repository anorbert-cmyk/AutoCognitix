# Sprint 12 SSE Cross-Review Notes

## SSE-Backend → SSE-Frontend Review

**Reviewer:** SSE-Backend Lead
**Date:** 2026-03-29
**Files reviewed:**
- `frontend/src/hooks/useStreamingDiagnosis.ts`
- `frontend/src/services/diagnosisService.ts`
- `frontend/src/types/streaming.ts`

---

### Endpoint URL — OK

`diagnosisService.ts:347`
```ts
const url = `${apiBaseUrl}/api/v1/diagnosis/analyze/stream`
```
Matches the backend route `POST /api/v1/diagnosis/analyze/stream`. Correct.

---

### SSE Parsing — OK

`parseSSEEvents()` in `diagnosisService.ts` splits on `\r?\n\r?\n` and extracts
`data:` lines. The backend `_format_sse_event()` emits:
```
event: {type}\ndata: {json}\n\n
```
The parser correctly ignores the `event:` line and only reads `data:` — this matches
the backend format. No issue.

---

### `StreamingEvent` interface — MINOR ISSUE

**File:** `frontend/src/types/streaming.ts:26`

```ts
export interface StreamingEvent {
  event_type: StreamingEventType;
  data: Record<string, unknown>;
  diagnosis_id: string;
  timestamp: string;
  progress: number;
}
```

The `progress` field is typed as `number` (non-nullable). The backend schema declares
`progress: Optional[float]` — it can be `None` / `null` for some events. Accessing
`event.progress` without a null-guard could surface `null` in UI.

**Impact:** Low — the hook updates state with `onProgress` only in
`useStreamingDiagnosis.ts:100`, and the `progress` field would just be `null` (rendered
as `0` by default in most progress bars). Not a breaking bug, but the type should be
`progress: number | null` to be accurate.

**Recommended fix (frontend lead):** Change `progress: number` → `progress: number | null`
in `streaming.ts` and guard reads as `event.progress ?? 0`.

---

### `onAnalysis` chunk field — POTENTIAL MISMATCH

**File:** `frontend/src/hooks/useStreamingDiagnosis.ts:73`

```ts
const chunk = typeof eventData.text === 'string' ? eventData.text : ''
```

The hook reads `eventData.text` to accumulate incremental LLM text. However, the
backend `analysis` events emitted during the streaming pipeline carry:
```json
{"stage": "rag_start", "message": "AI elemzes inditasa..."}
{"stage": "rag_complete", "message": "AI elemzes kesz", "model_used": "..."}
```
Neither contains a `text` field. The hook will always get `chunk = ''` from
`analysis` events, meaning `fullText` will remain empty throughout the stream.

The final structured result is delivered in the `complete` event's `data` dict
(not `full_result` as in the `stream_result_as_chunks()` fallback helper in
`streaming_service.py`). The `complete` event carries:
```json
{
  "diagnosis_id": "...",
  "confidence_score": 0.82,
  "urgency_level": "high",
  "probable_causes_count": 3,
  "repairs_count": 2,
  "recalls_count": 0,
  "complaints_count": 0,
  "message": "Diagnosztika befejezve"
}
```
This is stored in `fullResult` correctly via `onComplete`.

**Recommendation (frontend lead):**
- If a live text stream is desired, the backend `analysis` event would need a `text`
  field added (e.g. from LLM streaming). Currently the backend runs the full RAG
  pipeline and only streams structured events — not raw LLM tokens.
- Alternatively, accumulate display text from the structured `cause` and `repair`
  events instead of `analysis`.
- This is a design decision; the current behaviour (empty `fullText`, structured result
  in `fullResult`) still works end-to-end.

---

### `full_result` field — DESIGN NOTE (not a bug)

`streaming_service.py` (newly added file) wraps a completed result dict with:
```json
{"chunk": "", "done": true, "full_result": {...}}
```
The hook's `StreamChunk` interface (`useStreamingDiagnosis.ts:131`) matches this shape.
However, the main SSE endpoint uses the structured `complete` event format, not the
chunk format. The `stream_result_as_chunks()` helper is a reusable utility for simpler
consumers — it is not currently wired into the main endpoint. No conflict.

---

### TypeScript Types — OK

`DiagnosisFormData` imported from `diagnosisService` and passed to `streamDiagnosis`
is consistent. `StreamingCallbacks` interface is correctly implemented by the hook.

---

### Summary

| Check | Status | Notes |
|-------|--------|-------|
| Endpoint URL `/analyze/stream` | PASS | Exact match |
| SSE data parsing | PASS | Correct `data:` line extraction |
| `chunk` / `done` / `full_result` fields | PASS | Consistent with streaming_service.py |
| `StreamingEvent.progress` nullable | MINOR | Should be `number \| null` |
| `onAnalysis` text accumulation | NOTE | No `text` field in current backend events; `fullText` stays empty |
| TypeScript types overall | PASS | No type errors expected |

No blocking issues. Two items recommended for the frontend lead to consider.

---

## PasswordReset-DB → PasswordStrength Review

**Reviewer:** PasswordReset-DB Lead
**Date:** 2026-03-29
**Files reviewed:**
- `backend/app/core/security.py`
- `backend/app/api/v1/schemas/auth.py`

---

### Password hashing — PASS

`get_password_hash()` uses `CryptContext(schemes=["bcrypt"], deprecated="auto")`.
bcrypt is the correct choice. No issues.

---

### Password strength validation — PASS

`validate_password_strength()` in `security.py` enforces:
- Min 8 characters (`PASSWORD_MIN_LENGTH = 8`)
- Max 100 characters
- At least one lowercase letter `[a-z]`
- At least one uppercase letter `[A-Z]`
- At least one digit `\d`
- At least one special character `[!@#$%^&*()\\\_+\-=\[\]{}|;:,.<>?]`

This is comprehensive and exceeds the minimum brief (min 8 char, upper/lowercase, digit).

---

### `ResetPasswordRequest` schema — PASS

`auth.py` line 132–142:
```python
class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        return validate_password_strength(v)
```
The `ResetPasswordRequest` schema correctly:
- Defines `new_password` with min/max length constraints at the Field level
- Applies `validate_password_strength()` via `@field_validator` (uppercase + lowercase + digit + special)
- Imports `validate_password_strength` from `app.core.security`

No gaps found.

---

### `ForgotPasswordRequest` — PASS

Only requires `email: EmailStr`. No password field needed — correct by design.

---

### `UserCreate` and `UserPasswordUpdate` — PASS

Both also use `validate_password_strength` via `@field_validator`. Consistent across all password-bearing schemas.

---

### `check_password_strength()` vs `validate_password_strength()` — NOTE (not a bug)

There are two password checking utilities:
- `check_password_strength()` — returns a detailed dict with score (0–5), requirements, and Hungarian feedback. Used for UI feedback; does NOT raise.
- `validate_password_strength()` — raises `ValueError` on failure; used in Pydantic validators.

The `is_strong` threshold in `check_password_strength()` is `score >= 3` (3 out of 5 requirements met). However, `validate_password_strength()` requires ALL 5 requirements. There is a potential inconsistency: a password with `is_strong=True` from `check_password_strength()` (score=3) may still fail `validate_password_strength()`. This is a design discrepancy worth documenting but is not a security issue — the stricter validator always wins at the API boundary.

**Recommendation:** Document that `check_password_strength()` is for UI hints only, while `validate_password_strength()` is the authoritative gate.

---

### Summary

| Check | Status | Notes |
|-------|--------|-------|
| `get_password_hash()` uses bcrypt | PASS | CryptContext with bcrypt scheme |
| `validate_password_strength()` min 8 char | PASS | `PASSWORD_MIN_LENGTH = 8` |
| uppercase requirement | PASS | `re.search(r"[A-Z]", ...)` |
| lowercase requirement | PASS | `re.search(r"[a-z]", ...)` |
| digit requirement | PASS | `re.search(r"\d", ...)` |
| `ResetPasswordRequest` uses strength validator | PASS | `@field_validator("new_password")` |
| `check_password_strength` score vs validator threshold | NOTE | Score≥3 vs all-5 required — UI vs API discrepancy, not a security bug |

**No blocking issues.** The PasswordStrength implementation is solid and correctly integrated into `ResetPasswordRequest`.

---

## PasswordStrength → PasswordReset-DB Review

**Reviewer:** PasswordStrength Lead
**Date:** 2026-03-29
**Files reviewed:**
- `backend/app/db/postgres/models.py` (PasswordResetToken model)
- `backend/alembic/versions/017_add_password_reset_tokens.py` (migration)

---

### PasswordResetToken model — PASS

`PasswordResetToken` model exists in `models.py` with `__tablename__ = "password_reset_tokens"`.

| Requirement | Status | Detail |
|-------------|--------|--------|
| Model exists | PASS | `class PasswordResetToken(Base)` defined |
| `token_hash` unique constraint | PASS | `mapped_column(String(64), nullable=False, unique=True)` |
| `expires_at` DateTime field | PASS | `mapped_column(DateTime(timezone=True), nullable=False)` — timezone-aware |
| `used` boolean field | PASS | `mapped_column(Boolean, default=False, nullable=False)` |
| `user_id` FK with index | PASS | `ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True` |
| Relationship to User | PASS | `user: Mapped["User"]` backref + `User.reset_tokens` cascade delete-orphan |

---

### Migration 017 — PASS

| Requirement | Status | Detail |
|-------------|--------|--------|
| `down_revision` correct (not None) | PASS | `down_revision = "016_add_garage_tables"` — explicit, correct chain |
| `token_hash` UniqueConstraint | PASS | `sa.UniqueConstraint("token_hash")` in `op.create_table` |
| `expires_at` column | PASS | `sa.DateTime(timezone=True), nullable=False` |
| `used` boolean | PASS | `sa.Boolean(), nullable=False, server_default="false"` |
| Index on `user_id` | PASS | `op.create_index("ix_password_reset_tokens_user_id", ...)` |
| `downgrade()` drops index before table | PASS | Correct teardown order |
| Alembic unused-global suppression | PASS | `# lgtm[py/unused-global-variable]` on docstring and `revision`/`down_revision` lines |

---

### Summary

All checklist items pass. The PasswordReset-DB pair's implementation is complete and correct:
- Model and migration are consistent with each other
- Security properties (hashed token, expiry, single-use flag) are properly enforced at DB level
- `token_hash` uniqueness prevents token reuse/collision attacks
- CASCADE delete on `user_id` ensures no orphaned tokens on user deletion
- No blocking issues found.

---

## PasswordReset-API → Email-Templates Review

**Reviewer:** PasswordReset-API Lead
**Date:** 2026-03-29
**Files reviewed:**
- `backend/app/services/email_service.py`

---

### `send_password_reset_email()` exists — OK

The module-level convenience function is present at line 492:

```python
async def send_password_reset_email(to_email: str, name: str, reset_link: str) -> bool:
```

Delegates to `EmailService.send_password_reset()`, which formats the HTML/text template
and calls `_send_email()`. The function is async and importable — compatible with the
endpoint.

**Note on signature:** The task spec assumed
`send_password_reset_email(to_email, reset_token, username)`, but the actual signature is
`(to_email, name, reset_link)`. The endpoint implementation uses the actual signature:
constructs `reset_link` as `{FRONTEND_URL}/reset-password?token={plain_token}` and passes
`user.full_name or user.email` as `name`. This is correct.

---

### Reset URL path — OK

The endpoint builds:
```python
reset_link = f"{frontend_url}/reset-password?token={plain_token}"
```

The email template embeds `reset_link` as the button href and plain text link.
The expected frontend route `/reset-password?token=...` matches exactly.

---

### XSS Safety in HTML template — MEDIUM ISSUE

**File:** `email_service.py` lines 74–111 (`PASSWORD_RESET_TEMPLATE_HTML`)

```python
html_content = PASSWORD_RESET_TEMPLATE_HTML.format(
    name=name,
    reset_link=reset_link,
)
```

Neither `name` (user's `full_name`) nor `reset_link` is passed through `html.escape()`
before being interpolated into the HTML template. A malicious `full_name` containing
`<script>` tags or HTML entities would be injected verbatim into the email body.

`reset_link` is constructed internally from `settings.FRONTEND_URL` + a
`secrets.token_urlsafe(32)` value — low risk. But `name` comes directly from the
database `full_name` field, which is user-supplied at registration time.

**Recommended fix (Email-Templates agent):**
```python
import html as html_lib

html_content = PASSWORD_RESET_TEMPLATE_HTML.format(
    name=html_lib.escape(name),
    reset_link=reset_link,  # URL, not rendered as innerHTML — acceptable
)
```

The same fix is needed in `WELCOME_TEMPLATE_HTML.format(name=name, ...)`.

---

### Log fallback — OK

Three-tier fallback is correctly implemented:
1. n8n webhook (if `N8N_WEBHOOK_URL` is configured)
2. Resend API (if `RESEND_API_KEY` is configured)
3. Demo mode: `logger.info("[DEMO] Email küldése: ...")` — always available as last resort

`_sanitize_log()` is applied to all user-supplied values in log calls. No log injection risk.

---

### Találatok (Summary)

| Check | Status | Notes |
|-------|--------|-------|
| `send_password_reset_email()` exists | PASS | Correct async function, importable |
| Signature compatible with endpoint | PASS | Adjusted to actual `(to_email, name, reset_link)` |
| Reset URL path `/reset-password?token=...` | PASS | Matches endpoint construction |
| HTML XSS safety (`html.escape`) | MEDIUM | `name` not escaped before HTML interpolation |
| Log fallback (demo mode + sanitize) | PASS | Demo mode always available; `_sanitize_log()` used |

One medium-severity finding: HTML template interpolation should use `html.escape()` for
the `name` field to prevent potential XSS in email clients rendering HTML. Not a blocker
for the password reset flow, but should be addressed before production.

---

## Neo4j-Thread → Qdrant-Async Review

**Reviewer:** Neo4j-Thread Lead
**Date:** 2026-03-29
**Files reviewed:**
- `backend/app/db/qdrant_client.py`
- `backend/app/services/rag_service.py` (own file, for context)

---

### AsyncQdrantClient import — PASS

`qdrant_client.py` line 12:
```python
from qdrant_client import AsyncQdrantClient
```
`AsyncQdrantClient` is the correct async-native client. The sync `QdrantClient` is NOT
imported anywhere. Full marks.

---

### All public operations are awaited — PASS

Every method in `QdrantService` that touches `self.client` uses `await`:

| Method | Await pattern | Status |
|--------|---------------|--------|
| `_create_collection_if_not_exists` | `await self.client.get_collections()` / `await self.client.create_collection(...)` | PASS |
| `upsert_vectors` | `await self.client.upsert(...)` | PASS |
| `search` | `await self.client.search(...)` | PASS |
| `delete_by_user` | `await self.client.delete(...)` | PASS |
| `delete_collection` | `await self.client.delete_collection(...)` | PASS |
| `get_collection_info` | `await self.client.get_collection(...)` | PASS |

No blocking calls detected. Event loop will not be stalled.

---

### `connect()` / explicit connect step — N/A (not applicable)

`AsyncQdrantClient` connects lazily on first network call. There is no explicit
`connect()` step in the Qdrant async client API, so the absence of one is correct
by design.

---

### Lazy singleton — PASS

`_LazyQdrantProxy` defers `QdrantService.__init__()` (and therefore the
`AsyncQdrantClient` constructor + network connection) until first actual use.
The `threading.Lock` protecting singleton creation is appropriate because the
initialisation path is synchronous; no asyncio.Lock is needed here.

---

### GDPR `delete_by_user` — MINOR NOTE

`delete_by_user` is an `async` method and correctly uses `await self.client.delete(...)`.
However, the method catches all exceptions with a broad `except Exception` and only logs
a warning, silently swallowing errors mid-loop. If a deletion fails, the loop continues
and the caller receives the count of *attempted* collections, not the count of
*successfully deleted* ones. Not a thread-safety issue, but worth noting for auditability.

**Recommendation:** Track failed collections and include them in the return value or raise
after the loop if any deletions failed.

---

### Summary

| Check | Status | Notes |
|-------|--------|-------|
| `AsyncQdrantClient` imported | PASS | Correct async-native client |
| All search/upsert/create await-ed | PASS | No blocking calls |
| `connect()` method needed | N/A | Qdrant async client connects lazily |
| `asyncio.to_thread` misuse on async methods | PASS | None found |
| Singleton thread safety | PASS | `threading.Lock` for sync init path |
| `delete_by_user` error handling | MINOR | Silently swallows per-collection errors |

**No blocking issues.** The Qdrant-Async pair's implementation is correct and fully
async throughout. One minor recommendation for `delete_by_user` error reporting.

---

## SSE-Frontend → SSE-Backend Review

**Reviewer:** SSE-Frontend Lead
**Date:** 2026-03-29
**Files reviewed:**
- `backend/app/services/streaming_service.py`
- `backend/app/api/v1/endpoints/diagnosis.py` (streaming route `/analyze/stream`)

---

### Endpoint URL — PASS

Route registered as `POST /analyze/stream` under the `/api/v1/diagnosis` router prefix.
Combined path: `/api/v1/diagnosis/analyze/stream`.
Frontend `diagnosisService.ts` calls `${VITE_API_URL}/api/v1/diagnosis/analyze/stream`. Exact match.

---

### SSE Format — PASS

`_format_sse_event()` emits:
```
event: {type}\ndata: {json}\n\n
```
This is well-formed SSE. The frontend `parseSSEEvents()` correctly handles `event:` + `data:` lines, extracting only the JSON payload. No mismatch.

---

### `done: true` + `full_result` in final event — DESIGN NOTE (not a bug)

`streaming_service.py` `stream_result_as_chunks()` correctly emits a final event:
```json
{"chunk": "", "done": true, "full_result": {...}}
```
This matches the `StreamChunk` interface in `useStreamingDiagnosis.ts`.

However, the **main streaming endpoint** (`analyze_vehicle_stream`) does NOT use
`stream_result_as_chunks()`. Its final event is a `complete`-typed `StreamingEvent`:
```json
{"event_type": "complete", "data": {"confidence_score": ..., "diagnosis_id": ..., ...}, "progress": 1.0}
```
There is no top-level `done: true` field or `full_result` key in this event.

The hook handles this correctly via `onComplete` → `fullResult`. Consumers using
`streamDiagnosisGenerator` expecting `event.done === true` will only see it from
the `stream_result_as_chunks()` helper, not the main endpoint.

**Status:** No blocking issue. Design boundary is intentional; should be documented for future consumers.

---

### `complete` event payload — PASS

The `complete` event at `progress=1.0` carries `confidence_score`, `urgency_level`,
`probable_causes_count`, `repairs_count`, etc. Frontend `onComplete` stores this in
`fullResult`. Sufficient for UI rendering.

---

### Semaphore + capacity error — PASS

`MAX_CONCURRENT_STREAMS = 10`, 10-second acquisition timeout. If capacity is full,
endpoint yields `error` event with `error_type: "capacity"`. Frontend `onError` handles
all `error` event types uniformly. No gap.

---

### CSRF Token — PASS

`streamDiagnosis` attaches `X-CSRF-Token` header via `getCsrfToken()`. Correct.

---

### `analysis` events — no `text` field (confirmed from backend review)

Both leads independently identified this. Backend `analysis` events carry `stage` + `message`
fields only — no incremental LLM token streaming. `fullText` in the hook stays empty.
All actionable data arrives via `cause`, `repair`, and `complete` events.

**Status:** Known limitation, not a bug. `fullText` accumulation will activate once the
backend adds LLM token-level streaming to `analysis` events (future sprint).

---

### Summary

| Check | Status | Notes |
|-------|--------|-------|
| Endpoint URL `/api/v1/diagnosis/analyze/stream` | PASS | Exact match |
| SSE format `event:/data:/\n\n` | PASS | Correct |
| `done: true` in main endpoint final event | NOTE | Uses `complete` event type; handled correctly by hook |
| `full_result` in main endpoint | NOTE | Via `complete.data`; `stream_result_as_chunks()` uses top-level field |
| Error event capacity handling | PASS | Frontend `onError` catches uniformly |
| Semaphore protection | PASS | 10 concurrent max, fail-safe error event |
| CSRF token in streaming request | PASS | `getCsrfToken()` attached |
| `analysis` text accumulation | NOTE | Confirmed known limitation; no `text` field yet |

**Conclusion:** No blocking issues. The main endpoint and `streaming_service.py` helper
use two complementary output shapes — both handled correctly by the frontend.
The `fullText` feature awaits future LLM token streaming work on the backend.
