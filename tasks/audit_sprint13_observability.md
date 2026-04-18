# Sprint 13 Observability Audit

_Scope: Error tracking, metrics, audit log. Max 8 files read._

## A) Error tracking
- **severity: MEDIUM** (infra exists but gaps remain)
- findings:
  - `backend/app/core/logging.py:617-647` — Sentry SDK conditionally initialized if `SENTRY_DSN` set; uses FastApi + SQLAlchemy + Logging integrations. Lazy `import sentry_sdk` inside try/except → silent fallback (line 647 warns only).
  - `backend/app/core/error_handlers.py:1-100` — Global exception handlers present with request ID tracing and structured logging, but **no explicit `sentry_sdk.capture_exception()` calls** anywhere (grep of whole file = 0 hits). Relies solely on LoggingIntegration auto-forwarding; unhandled non-logged errors may be missed for custom exception paths.
  - `frontend/src/config/sentry.ts:1-18` — Sentry React SDK wired, gated on `VITE_SENTRY_DSN`. `enabled: import.meta.env.PROD` → DSN only active in prod build; dev errors never reported.
  - `frontend/src/components/ErrorBoundary.tsx:50-64` — `componentDidCatch` logs only to console in DEV; **no `Sentry.captureException(error)` call in catch handler**. In prod, caught React errors are swallowed unless the injected `onError` prop wires Sentry — grep of `App.tsx`/`main.tsx` needed to confirm wiring; not confirmed here.
  - Gap: DSN environment variable is not documented in `CLAUDE.md` Railway env vars list → likely unset in production.

## B) Metrika
- **severity: LOW** (well-instrumented)
- findings:
  - `docker-compose.monitoring.yml:1-50` — Full Prometheus+Grafana+Alertmanager stack defined (v2.48.0, ports 9090/3001/9093). Healthchecks configured.
  - `backend/app/api/v1/endpoints/metrics.py:32-57` — `/metrics` endpoint returns Prometheus text format via `generate_latest()`; `/metrics/summary` offers JSON.
  - `backend/app/core/metrics.py` — custom counters (`DTC_CODES_TOTAL`, `update_system_metrics`) exported.
  - `backend/app/middleware/metrics.py` — request-level middleware present.
  - Gap: `prometheus-fastapi-instrumentator` library not used; manual wiring. Verify all routers mounted before middleware; verify `/metrics` is unauthenticated (Prometheus cannot scrape otherwise) or allowlisted.

## C) Audit log
- **severity: CRITICAL** (GDPR + security)
- findings:
  - `backend/app/db/postgres/models.py` — grep `AuditLog|ActionLog|audit_log|action_log` → **0 matches**. Only `NHTSASyncLog` at line 649 exists (external data sync log, not a user/admin action audit).
  - `backend/alembic/versions/` — grep `audit|Audit` across 20 migration files → **0 matches**. Latest 3: `018_fix_diagnosis_session_fk_and_expires_index.py`, `017_add_password_reset_tokens.py`, `016_add_garage_tables.py` — none audit-related.
  - Impact: No persistent record of login attempts, password changes, data exports, vehicle CRUD, or admin actions. GDPR Art. 30 (records of processing) and Art. 32 (security logging) obligations unmet. Security incident forensics impossible.
  - Recommendation: Add `AuditLog(id, user_id, action, resource_type, resource_id, ip, user_agent, metadata, created_at)` model + migration; emit from auth, garage CRUD, diagnosis analyze, admin endpoints. Append-only table, never UPDATE/DELETE.

## Olvasott fájlok
- /home/user/AutoCognitix/backend/app/core/logging.py (grep only, lines 589-647)
- /home/user/AutoCognitix/backend/app/core/error_handlers.py (1-100)
- /home/user/AutoCognitix/frontend/src/config/sentry.ts (full)
- /home/user/AutoCognitix/frontend/src/components/ErrorBoundary.tsx (full)
- /home/user/AutoCognitix/docker-compose.monitoring.yml (1-50)
- /home/user/AutoCognitix/backend/app/api/v1/endpoints/metrics.py (1-60)
- /home/user/AutoCognitix/backend/app/db/postgres/models.py (grep only)
- /home/user/AutoCognitix/backend/alembic/versions/ (glob listing, grep)
