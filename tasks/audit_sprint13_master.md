# Sprint 13 Master Audit Összesítő

**Dátum:** 2026-04-18
**Lefedettség:** 7 specialist (Security, Database, Performance, Observability, Code Quality, Data Integrity, Product/Vision)
**Olvasott fájlszám:** ~60 fájl + 20+ grep/glob
**Tartalom:** 24 finding (1 CRITICAL, 9 HIGH, 10 MEDIUM, 4 LOW)

Ez a dokumentum összegzi a Sprint 13 párhuzamos kódbázis audit eredményeit. A **javítások nem ebben a sprintben** történnek — az itt azonosított CRITICAL + HIGH tételek a Sprint 13-14 backlog-jába mennek, a MEDIUM-ok a Sprint 15-be, a LOW-ok a backlog mélyére.

---

## 1. Severity áttekintés

| Severity | Db | Akció |
|----------|----|----|
| CRITICAL | 1 | Azonnali hotfix a következő sprintben |
| HIGH | 9 | Sprint 13 elsődleges scope |
| MEDIUM | 10 | Sprint 14-15 scope |
| LOW | 4 | Backlog, opportunisztikus fix |

---

## 2. CRITICAL (1)

| # | Terület | Probléma | Fájl | Javaslat |
|---|---------|----------|------|----------|
| C1 | Observability | **Nincs audit log tábla** — 0 forensikus nyoma auth/CRUD/admin akcióknak. GDPR Art. 30/32 megsértése. | `backend/app/db/postgres/models.py` (nincs `AuditLog`), `backend/alembic/versions/` (0 audit migration) | Új `AuditLog` modell + migration: `(id, user_id, action, resource_type, resource_id, ip, user_agent, metadata, created_at)` append-only. Emit auth/garage/diagnosis/admin végpontokról. |

---

## 3. HIGH (9)

### Security (3)
| # | Probléma | Fájl | Javaslat |
|---|----------|------|----------|
| H1 | JWT `iss`/`aud` claim nem ellenőrzött — cross-service token reuse elleni védelem hiányzik | `backend/app/core/security.py:173-178` | `jwt.decode(..., issuer=..., audience=..., options={"require": ["exp","iat","sub","type","jti","iss","aud"]})` |
| H2 | **Login endpoint-on NINCS rate limit** — csak account lockout, elosztott credential stuffing nem fékezhető | `backend/app/api/v1/endpoints/auth.py:463-561` | `@limiter.limit("5/minute")` login + forgot-password végpontokra |
| H3 | Refresh token rotation — régi refresh explicit `blacklist_token()` hívás nem látható → replay ablak | `backend/app/api/v1/endpoints/auth.py:583-599` | Refresh-kor `await blacklist_token(old_refresh_jti, old_exp)` a kiadás előtt |

### Data Integrity (3)
| # | Probléma | Fájl | Javaslat |
|---|----------|------|----------|
| H4 | HuggingFace **`revision=` pin hiányzik** — huBERT modell silent drift ellen nincs védelem 35k élő embedding fölött | `backend/app/services/embedding_service.py:212-221`, `config.py:152` | `AutoModel.from_pretrained(MODEL, revision="<commit_sha>")`, env `HUBERT_REVISION` |
| H5 | **Embedding cache kulcs nem tartalmaz modell-verziót** — ha a modell frissül, stale vektorok szivárognak | `backend/app/db/redis_cache.py:567-574` | Kulcs: `f"{model}:{revision}:{sha256(text)}"` |
| H6 | **Nincs automatizált restore test, nincs RTO/RPO** — DR nem bizonyított | `docs/BACKUP.md`, `scripts/backup_data.py` | Heti automatizált restore smoke-test (staging), RTO/RPO dokumentálása |

### Product/Vision (3)
| # | Probléma | Bizonyíték | Javaslat |
|---|----------|------------|----------|
| H7 | **Fizetési integráció teljes hiánya** (Stripe/SimplePay) — `/pricing` oldal marketing-only, revenue lehetőség = 0 | `frontend/src/pages/PricingPage.tsx` hard-coded tier-ek, `grep stripe\|SimplePay` = 0 találat backend+frontend | Stripe checkout integráció, `Subscription` modell PG-ben, webhook endpoint, billing oldal |
| H8 | **Nincs i18n / angol lokalizáció** — vízió "both Hungarian and English" | `grep i18next\|react-i18next` = 0 találat, minden magyar hard-coded | `react-i18next` + translation JSON-ok, locale switcher |
| H9 | **Nincs admin panel** — user/subscription management nem megoldott | `grep admin` api-ban csak role-említés, nincs admin route frontend pages-ben | Admin endpoint (`/api/v1/admin/*`) + `/admin` route + role guard |

---

## 4. MEDIUM (10)

| # | Terület | Probléma | Fájl |
|---|---------|----------|------|
| M1 | Database | Alembic drift: `DiagnosisArchive.user_id` FK a modellben, migrációban nincs | `db/postgres/models.py:281` vs `alembic/versions/013:29` |
| M2 | Database | Alembic drift: `DiagnosisArchive.original_id` és `user_id` `index=True` no-op migrációban | `alembic/versions/013:28-29` |
| M3 | Database | JSONB vs ARRAY típus-keveredés `dtc_codes`-nál (archive JSONB, session ARRAY) | `db/postgres/models.py:249 vs 288` |
| M4 | Database | `016` migration: `server_default` hiányzik `is_active`, `is_completed`, stb. oszlopokra | `alembic/versions/016_add_garage_tables.py:38,68` |
| M5 | Performance | `preprocess_hungarian` default executor-ban → CPU-heavy thread pool starvation | `services/diagnosis_service.py:427` |
| M6 | Observability | Sentry SDK wired, de `sentry_sdk.capture_exception()` hívás sehol | `core/error_handlers.py:1-100`, `frontend/ErrorBoundary.tsx:50-64` |
| M7 | Observability | Sentry DSN env var **nincs dokumentálva** Railway setup-ban → valószínűleg prod-ban unset | `CLAUDE.md` Railway env list |
| M8 | Code Quality | 2x bare `# type: ignore` (error-code nélkül, túl széles letiltás) | `core/logging.py:548-549` |
| M9 | Code Quality | Teszt fájl-duplikáció: `api/`, `integration/`, `e2e/` ugyanazokat az endpoint-okat fedi | `backend/tests/` |
| M10 | Code Quality | **Frontend 0 Playwright E2E teszt** — csak 19 Vitest komponens teszt | `frontend/src/` |
| M11 | Security | `.env.example` `DEBUG=true` + `ENVIRONMENT=development` default → prod-másolás-veszély | `.env.example:49-50` |
| M12 | Security | `.gitleaks.toml` allowlist túl megengedő (`.*test.*\.py$`, `example.*token`) | `.gitleaks.toml:11-26` |
| M13 | Data Integrity | Seed race condition: nincs `ON CONFLICT DO NOTHING` / `MERGE` | `scripts/seed_database.py:292-296,323-327,345-349,446-449` |
| M14 | Data Integrity | `requirements.prod.txt` torch/transformers kikommentelve → prod nem tud embedding-et újragenerálni | `backend/requirements.prod.txt:48-50` |

---

## 5. LOW (4)

| # | Terület | Probléma | Fájl |
|---|---------|----------|------|
| L1 | Security | Bcrypt work factor nem explicit konfigurálva (passlib default=12, verzióváltás csendben csökkentheti) | `core/security.py:26` |
| L2 | Database | `_save_diagnosis_session` hiba-ág némán `False`-t ad vissza, rollback nélkül → background task-ban elveszthet diagnózis | `services/diagnosis_service.py:1194-1199` |
| L3 | Performance | `get_vehicles` 2 round-trip (count + select) a list endpoint-on | `services/vehicle_garage_service.py:91-113` |
| L4 | Performance | Frontend: `minify: 'terser'` → válts `'esbuild'`-ra (30-50% gyorsabb build); Sentry eager import (~60 kB) | `frontend/vite.config.ts:49`, `App.tsx:3` |
| L5 | Observability | `/metrics` endpoint auth verify (Prometheus scrape kell, de allowlist nem dokumentált) | `api/v1/endpoints/metrics.py:32-57` |

---

## 6. Ami JÓL működik (no action)

- **PG mint source of truth** egyértelmű (`data_sync.py` egyirányú PG→Neo4j/Qdrant)
- **Injection védelem**: SQLAlchemy paraméterezett query-k, `escape_ilike` 11 helyen, Cypher label whitelist (`health.py:207`)
- **Secret management**: Pydantic `BaseSettings` konzisztens, nincs szóródó `os.getenv`
- **Prometheus stack** (`docker-compose.monitoring.yml`, `/metrics` endpoint, custom counter-ek) kész
- **Backup lefedettség**: PG + Neo4j + Qdrant + JSON mind backupolva, restore scripted
- **Python 3.9 kompatibilitás**: 0 `zip(strict=)`, 0 PEP 604 pipe union
- **Frontend bundle**: React.lazy 19 oldalra, manualChunks 7 vendor csomag
- **CI/CD pipeline**: Ruff, MyPy, pytest, Vitest, CodeQL, Bandit, npm audit — mind él
- **JWT type confusion védelem** (`expected_type` ellenőrzés), fail-closed `jti` blacklist

---

## 7. Javasolt Sprint allocation

### Sprint 13 (2 hét) — CRITICAL + top HIGH (biztonsági + infra)
- C1 (audit log) — 3 nap
- H1-H3 (JWT claim + login rate limit + refresh blacklist) — 2 nap
- H4-H5 (embedding pin + cache kulcs) — 1 nap
- H6 (restore smoke-test + RTO/RPO doc) — 2 nap
- M1-M4 (Alembic drift fix 013/016) — 1 nap
- M6 (Sentry capture_exception wiring) — 1 nap

### Sprint 14 (2 hét) — Vízió gap (monetization + i18n)
- H7 (Stripe integráció, Subscription modell, webhook) — 5 nap
- H8 (i18n alap HU/EN) — 3 nap
- H9 (admin panel v1) — 3 nap

### Sprint 15 (1 hét) — Quality + perf MEDIUM fixek
- M5 (dedicated executor)
- M8-M10 (bare type:ignore, test consolidation, Playwright E2E alap)
- M11-M14 (secret hygiene, seed idempotency, prod requirements)

### Backlog — LOW tételek
- L1-L5 opportunisztikus javítások következő touchpoint-nál

---

## 8. Új, kötelező follow-up fájlok

- `tests/test_sprint13_audit.py` — unit teszt minden HIGH + CRITICAL javításhoz (CLAUDE.md Post-Sprint Review protokoll)
- `docs/DR_PLAYBOOK.md` — RTO/RPO + restore runbook (H6 deliverable)
- `docs/ADMIN_PANEL_SPEC.md` — H9 előfeltétele

---

## 9. Részletes riportok hivatkozása

- `tasks/audit_sprint13_security.md`
- `tasks/audit_sprint13_database.md`
- `tasks/audit_sprint13_performance.md`
- `tasks/audit_sprint13_observability.md`
- `tasks/audit_sprint13_quality.md`
- `tasks/audit_sprint13_data.md`
- `tasks/audit_sprint13_product.md`

Onboarding csomag új szakértőknek:
- `docs/ONBOARDING.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA_FLOW.md`
- `docs/DATABASE_MAP.md`
