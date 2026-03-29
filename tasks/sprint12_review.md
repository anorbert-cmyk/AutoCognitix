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
