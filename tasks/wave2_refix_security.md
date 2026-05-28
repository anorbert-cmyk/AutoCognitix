# Wave2 Refix — Security Lead

Audit target: HEAD = 9ad6470 on `claude/sprint13-bugfixes-wave2`.
Focus: migration 019 orphan purge atomicity, Sentry PII in extras, HUBERT prod default.

## A) Migration 019 orphan purge / FK atomicity
- **severity:** HIGH (correctness + race), MEDIUM (lock)
- **finding 1 (TOCTOU / race window):** `019_fix_diagnosis_archive_indexes_and_fk.py:41-78`
  The DELETE (step 1) and the `ALTER TABLE … ADD CONSTRAINT … NOT VALID` +
  `VALIDATE CONSTRAINT` (steps 3a/3b) are three separate statements with no
  explicit table-level lock between them. Alembic runs the upgrade in a
  transaction by default, BUT under online-DDL configs (`transactional_ddl=False`)
  or with autocommit blocks the gap is exploitable: an INSERT into
  `diagnosis_archive` referencing a user deleted seconds earlier can land
  between DELETE and VALIDATE, causing the deploy to crash on the
  `VALIDATE CONSTRAINT` step (FK violation on the new row). Even under default
  transactional DDL, concurrent transactions writing to `diagnosis_archive`
  with `READ COMMITTED` snapshot can sneak in before the FK catalog update
  becomes visible.
  **Recommendation:** acquire `LOCK TABLE diagnosis_archive IN SHARE MODE`
  inside the same transaction as DELETE, or perform DELETE in a loop until
  zero rows affected immediately before VALIDATE.
- **finding 2 (NULL handling — already correct, confirm):** line 43-46.
  `WHERE user_id IS NOT NULL AND user_id NOT IN (SELECT id FROM users)` is
  correct: the explicit `IS NOT NULL` guard prevents the classic
  `NULL NOT IN (...)` → UNKNOWN → row skipped pitfall. `users.id` is the
  PK so no NULL on the right side. Verified safe. No action.
- **finding 3 (NOT IN subquery NULL trap on inner):** line 45. If a future
  refactor made `users.id` nullable (extremely unlikely PK), every row
  would silently survive purge. Defense-in-depth: replace with
  `NOT EXISTS (SELECT 1 FROM users u WHERE u.id = diagnosis_archive.user_id)`
  — same plan, NULL-safe regardless of inner-column nullability.
- **finding 4 (no SAVEPOINT, partial-failure recoverability):** lines 41-78.
  If VALIDATE fails (e.g. inserts during the gap), the whole transaction
  rolls back including the orphan DELETE → operator re-runs and re-deletes
  the same orphan set. Idempotent in this case, but consider wrapping
  steps 3a/3b in a SAVEPOINT so an operator can retry just VALIDATE after
  cleaning interferring rows.
- **finding 5 (downgrade non-atomic):** lines 81-90. `drop_constraint` then
  two `drop_index` calls; if the FK drop succeeds and a later index drop
  fails, the migration is left half-rolled-back with no FK but indexes still
  present. Acceptable since `if_exists=True` makes re-run idempotent.

## B) Sentry `set_extra("raw_path")` PII leak
- **severity:** HIGH (GDPR)
- **finding 1 (UUID/VIN leakage in extras):** `error_handlers.py:336`.
  `scope.set_extra("raw_path", request.url.path)` ships the FULL URL path
  to Sentry — including UUIDs (`/api/v1/garage/vehicles/<uuid>`), VINs
  (`/vehicles/decode-vin/WVWZZZ1JZXW000001`), and any token-bearing path
  segment. Sentry stores extras unindexed but VISIBLE in the event JSON in
  the UI; combined with the `request_id` tag they are trivially correlatable
  to a natural person → GDPR Art.4(1) personal data. The PII-safe-tag
  refactor is correct on the TAG axis but **leaks the same PII via extras**,
  defeating the purpose.
  **Recommendation:** redact UUID/VIN-shaped segments before sending:
  ```python
  import re
  _UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
  _VIN  = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
  def _redact(p: str) -> str:
      p = _UUID.sub("<uuid>", p)
      return _VIN.sub("<vin>", p)
  scope.set_extra("raw_path", _redact(request.url.path))
  ```
  Also drop the query string (currently `request.url.path` excludes it, OK)
  and consider redacting numeric IDs ≥6 digits.
- **finding 2 (request_id correlation):** line 333. `request_id` tagged
  alongside `raw_path` makes the extra searchable-by-correlation even if
  not indexed. Acceptable for ops; document that Sentry project must be in
  EU region for GDPR.

## C) HUBERT prod default `"main"`
- **severity:** MEDIUM (supply-chain / silent drift)
- **finding 1 (no runtime enforcement):** `config.py:156`.
  `HUBERT_REVISION: str = "main"` default + comment-only "REQUIRED IN PROD"
  in `.env.railway.example:81-82`. Nothing prevents a prod deploy from
  silently running on whatever HuggingFace currently has as `main` — a
  compromised SZTAKI account or rebase could swap the model weights, with
  no visible change to operators. Cache invalidation (`embedding_cache_key`
  hashing `HUBERT_REVISION`) only helps AFTER drift is detected.
  **Recommendation:** add to `Settings.model_post_init` (or a
  `@field_validator`) in `config.py`:
  ```python
  @field_validator("HUBERT_REVISION")
  @classmethod
  def _warn_or_fail_on_main(cls, v, info):
      env = info.data.get("ENVIRONMENT", "development")
      if env == "production" and v == "main":
          import warnings
          warnings.warn(
              "HUBERT_REVISION='main' in production — pin a commit hash",
              RuntimeWarning, stacklevel=2,
          )
          # Stricter: raise ValueError(...) to fail-fast on deploy.
      return v
  ```
  Stricter alternative: hard-fail on prod boot (recommended for
  supply-chain critical inference path).
- **finding 2 (no integrity check on download):** `embedding_service.py`
  pulls the model via `transformers.AutoModel.from_pretrained(...,
  revision=...)`. Even with a pinned hash, no SHA verification on the
  downloaded shards happens beyond HuggingFace's own. Mitigated if the
  revision is a commit hash (immutable), unmitigated for tag/branch refs.

## Olvasott fájlok
- `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py`
- `backend/app/core/error_handlers.py` (l. 300-360)
- `backend/app/core/config.py` (l. 140-180)
- `.env.example` (diff)
- `.env.railway.example` (diff)
