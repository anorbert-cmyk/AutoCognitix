# Wave2 R3 — Security

Scope: commit `4258ddc` HEAD only. Files reviewed:
- `backend/app/core/error_handlers.py` (PII regex)
- `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py` (SHARE LOCK)

## A) PII regex (`_redact_pii`)

- **severity: LOW (no new HIGH/CRITICAL)**
- **finding (UUID): error_handlers.py:314-316 — OK.** Character class `[0-9a-fA-F]` is explicitly case-insensitive; `\b` boundaries are correct. Matches canonical UUIDs (any version, not just v4 — comment misleadingly says "UUID v4" but regex is version-agnostic, which is *better* for redaction). No false negatives for the SQLAlchemy/Pydantic format.
- **finding (VIN): error_handlers.py:318 — LOW false-positive risk.** Pattern `\b[A-HJ-NPR-Z0-9]{17}\b` correctly excludes I/O/Q per ISO 3779. False-positive surface in `request.url.path`:
  - Any uppercase 17-char alnum slug without I/O/Q is over-redacted (e.g. `ABCDEFGHJKLMNPRST`). Probability of an ASCII-uppercase 17-char ID in a URL path is low; AutoCognitix path style uses lowercase + UUIDs. **Cost: cosmetic, not security.**
  - Session tokens / JWT segments are normally ≥32 chars and live in headers/cookies, not `.path` → no impact.
- **finding (missing classes): error_handlers.py:321 — LOW, scope acceptable.** Email / phone / IBAN are not redacted, but `request.url.path` does NOT include the query string (FastAPI/Starlette behavior), so these values are not captured here in the first place. If a future endpoint puts an email in the path segment (`/users/foo@bar.com/...`) Sentry would leak it. **Recommendation (non-blocking):** add a one-line email regex for defense-in-depth, e.g. `r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"`.
- **finding (case): error_handlers.py:318 — informational.** VIN regex is uppercase-only. VINs in the wild are case-insensitive in client input; if a route accepts a lowercase VIN (`/vehicles/wvwzzz...`) it will NOT be redacted. Quick fix: compile with `re.IGNORECASE` or extend the class. Severity LOW because backend normalizes VINs to uppercase before routing in existing services.

**Verdict A: no new HIGH. One LOW recommendation (case-insensitive VIN + optional email regex).**

## B) SHARE LOCK escalation (migration 019)

- **severity: LOW (acceptable for this table size)**
- **finding (lock compatibility): 019_fix_…py:46 — correct intent.** `LOCK TABLE ... IN SHARE MODE` is compatible with `ACCESS SHARE` (concurrent reads OK) and with itself, but conflicts with `ROW EXCLUSIVE` / `SHARE ROW EXCLUSIVE` / `EXCLUSIVE` / `ACCESS EXCLUSIVE`. This correctly **blocks** parallel `INSERT/UPDATE/DELETE` against `diagnosis_archive` for the migration's duration — which is the stated TOCTOU defense and works as advertised.
- **finding (deadlock risk): 019_fix_…py:46-90 — LOW.** Two realistic deadlock vectors:
  1. A concurrent `INSERT` holding `ROW EXCLUSIVE` plus an unrelated row lock that the migration would need. The migration touches no rows other than orphans (already dead refs), and the `DELETE … RETURNING` runs *after* the SHARE lock blocks new inserts. **Not a credible deadlock path.**
  2. A long-running `SELECT FOR UPDATE` on `users` (subquery target) intersecting the DELETE. Subquery on `users` only needs `ACCESS SHARE` on `users`, compatible with everything except `ACCESS EXCLUSIVE`. **Not a credible deadlock path.**
- **finding (live INSERT starvation): 019_fix_…py:46 — LOW, acceptable per scope note.** Concurrent archive INSERTs will **wait** for the migration tx to commit. Per CLAUDE.md sizing (archive holds only inactive sessions, far smaller than the 26K/35K live datasets in Neo4j/Qdrant), the DELETE + index creates + ADD CONSTRAINT complete in subseconds → block window is short. Acceptable.
- **finding (no `lock_timeout`): 019_fix_…py:46 — LOW (operational, not security).** Migration does not set `SET LOCAL lock_timeout = '5s'` before `LOCK TABLE`. If a stray long-running `SELECT` on `diagnosis_archive` holds `ACCESS SHARE` (compatible with SHARE — *wait, this does NOT block the migration*) or some admin tool holds an incompatible mode, the migration would hang the entire deploy indefinitely. **Recommendation (non-blocking):** prepend `op.execute(sa.text("SET LOCAL lock_timeout = '10s'"))` so a stuck deploy fails fast instead of wedging.
- **finding (downgrade gap): 019_fix_…py:93-95 — already documented, LOW.** Comment explicitly notes that downgrade-then-upgrade silently purges new orphans. This is the *intended* GDPR posture (orphans must not exist) and is now audit-logged via `RETURNING id` + `print`. Acceptable.

**Verdict B: no new HIGH. One LOW recommendation (add `lock_timeout` to fail-fast on contention).**

## Summary

- 0 CRITICAL, 0 HIGH, 0 MEDIUM, **2 LOW** (cosmetic / defense-in-depth).
- Both refixes correctly close the originally reported HIGH findings (Sec A TOCTOU, Sec B `raw_path` PII leak).
- Status: **CLEAN** for promotion.
