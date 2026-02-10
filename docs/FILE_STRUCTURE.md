# AutoCognitix - Teljes Fájlstruktúra
# Complete File Structure

---

## Projekt Gyökér / Project Root

```
AutoCognitix/
├── .claude/                          # Claude Code konfiguráció
│   └── settings.local.json
│
├── .github/                          # GitHub CI/CD
│   ├── dependabot.yml                # Automatikus függőség frissítés
│   └── workflows/
│       ├── cd.yml                    # Continuous Deployment (Railway)
│       ├── ci.yml                    # Continuous Integration (Lint, Test, Build)
│       └── security.yml              # Security scanning (CodeQL, Bandit)
│
├── backend/                          # FastAPI Backend Alkalmazás
│   ├── alembic/                      # PostgreSQL migrációk
│   ├── app/                          # Fő alkalmazás kód
│   └── tests/                        # Pytest tesztek
│
├── frontend/                         # React Frontend Alkalmazás
│   ├── dist/                         # Build output
│   └── src/                          # Forráskód
│
├── data/                             # Adatfájlok (nem verziókövetett)
│   ├── dtc_codes/                    # DTC kód JSON-ok
│   ├── nhtsa/                        # NHTSA export
│   └── obdb/                         # OBDb jármű adatok
│
├── docs/                             # Dokumentáció
│   ├── API*.md                       # API dokumentáció
│   ├── COWORK_BRIEF.md               # Pályázat összefoglaló
│   ├── BUDGET_AND_RESOURCES.md       # Költségvetés
│   └── ...
│
├── docker/                           # Docker konfiguráció
│   ├── alertmanager/                 # Alertmanager config
│   ├── grafana/                      # Grafana dashboards
│   └── prometheus/                   # Prometheus rules
│
├── scripts/                          # Utility scriptek
│   ├── cli/                          # CLI eszközök
│   ├── export/                       # Export scriptek
│   └── *.py                          # Import, sync, scrape scriptek
│
├── tasks/                            # Task tracking
│   ├── lessons.md                    # Tanulságok
│   └── todo.md                       # TODO lista
│
├── traefik/                          # Traefik reverse proxy
│
├── .env                              # Környezeti változók (lokális)
├── .env.example                      # Env template
├── .env.production.example           # Production env template
├── .env.railway.example              # Railway env template
├── .gitleaks.toml                    # Secret scanning config
├── .pre-commit-config.yaml           # Pre-commit hooks
├── CLAUDE.md                         # Claude Code projekt kontextus
├── docker-compose.yml                # Lokális fejlesztés
├── docker-compose.prod.yml           # Production
├── docker-compose.monitoring.yml     # Monitoring stack
├── PROJECT_OVERVIEW.md               # Projekt áttekintés
├── railway.json                      # Railway deployment config
└── README.md                         # Projekt README
```

---

## Backend Struktúra

```
backend/
├── alembic/                          # PostgreSQL Migrációk
│   ├── env.py                        # Alembic konfiguráció
│   └── versions/
│       ├── 001_initial_schema.py     # Alap séma
│       ├── 002_add_dtc_sources_column.py
│       ├── 003_vehicle_recalls.py
│       ├── 004_perf_indexes.py       # Teljesítmény indexek
│       ├── 005_vehicle_schema.py
│       ├── 006_nhtsa_sync.py
│       ├── 007_soft_delete.py        # Soft delete oszlopok
│       └── 008_add_fk_constraints.py # Foreign key constraints
│
├── app/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry point
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── router.py             # API router összeállítás
│   │       │
│   │       ├── endpoints/            # API végpontok
│   │       │   ├── __init__.py
│   │       │   ├── auth.py           # Authentikáció (login, register, logout)
│   │       │   ├── diagnosis.py      # Diagnosztika végpontok
│   │       │   ├── dtc_codes.py      # DTC kód keresés, CRUD
│   │       │   ├── health.py         # Health check
│   │       │   ├── metrics.py        # Prometheus metrics
│   │       │   └── vehicles.py       # Jármű adatok, VIN dekódolás
│   │       │
│   │       └── schemas/              # Pydantic schemák
│   │           ├── __init__.py
│   │           ├── auth.py           # User, Token schemák
│   │           ├── diagnosis.py      # DiagnosisRequest/Response
│   │           ├── dtc.py            # DTCCode, DTCDetail
│   │           └── vehicle.py        # Vehicle, VIN schemák
│   │
│   ├── core/                         # Core modulok
│   │   ├── __init__.py
│   │   ├── config.py                 # Központi konfiguráció (Settings)
│   │   ├── csrf.py                   # CSRF védelem
│   │   ├── error_handlers.py         # Globális hibakezelők
│   │   ├── etag.py                   # ETag cache
│   │   ├── exceptions.py             # Egyedi kivételek
│   │   ├── log_sanitizer.py          # Log adatok tisztítása
│   │   ├── logging.py                # Strukturált logolás
│   │   ├── metrics.py                # Prometheus metrics
│   │   ├── rate_limit.py             # Rate limiting
│   │   ├── rate_limiter.py           # Rate limiter impl
│   │   ├── retry.py                  # Retry logic
│   │   └── security.py               # JWT, password hashing
│   │
│   ├── db/                           # Adatbázis réteg
│   │   ├── __init__.py
│   │   ├── neo4j_models.py           # Neo4j node/relationship modellek
│   │   ├── qdrant_client.py          # Qdrant vektor kliens
│   │   ├── redis_cache.py            # Redis cache wrapper
│   │   │
│   │   └── postgres/
│   │       ├── __init__.py
│   │       ├── models.py             # SQLAlchemy ORM modellek
│   │       ├── repositories.py       # Repository pattern
│   │       └── session.py            # DB session management
│   │
│   ├── middleware/                   # FastAPI middleware-ek
│   │   ├── __init__.py
│   │   └── metrics.py                # Request metrics middleware
│   │
│   ├── prompts/                      # LLM prompt template-ek
│   │   ├── __init__.py
│   │   └── diagnosis_hu.py           # Magyar diagnosztika prompt
│   │
│   └── services/                     # Üzleti logika szolgáltatások
│       ├── __init__.py
│       ├── diagnosis_service.py      # Fő diagnosztika logika
│       ├── embedding_service.py      # HuBERT embedding generálás
│       ├── llm_provider.py           # LLM provider abstraction
│       ├── nhtsa_service.py          # NHTSA API kliens
│       ├── rag_service.py            # LangChain RAG pipeline
│       └── vehicle_service.py        # Jármű adatok kezelése
│
├── tests/                            # Tesztek
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   │
│   ├── api/                          # API tesztek
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_auth.py
│   │   ├── test_diagnosis.py
│   │   ├── test_dtc.py
│   │   ├── test_integration.py
│   │   └── test_vehicles.py
│   │
│   ├── e2e/                          # End-to-end tesztek
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_auth_flow.py
│   │   ├── test_data_integrity.py
│   │   ├── test_diagnosis_flow.py
│   │   └── test_dtc_api.py
│   │
│   ├── integration/                  # Integrációs tesztek
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_auth_api.py
│   │   ├── test_database_neo4j.py
│   │   ├── test_database_postgres.py
│   │   ├── test_database_qdrant.py
│   │   ├── test_diagnosis_api.py
│   │   ├── test_dtc_api.py
│   │   ├── test_e2e_diagnosis.py
│   │   ├── test_service_embedding.py
│   │   ├── test_service_rag.py
│   │   └── test_vehicles_api.py
│   │
│   ├── test_api_endpoints.py
│   ├── test_dtc_validation.py
│   ├── test_embedding_service.py
│   ├── test_rag_pipeline.py
│   └── test_translation_fixer.py
│
├── Dockerfile                        # Dev Dockerfile
├── Dockerfile.prod                   # Production Dockerfile
├── railway.toml                      # Railway konfig
├── requirements.txt                  # Dev függőségek
├── requirements.prod.txt             # Prod függőségek (torch nélkül)
├── ruff.toml                         # Ruff linter konfig
└── setup.cfg                         # Python tools konfig
```

---

## Frontend Struktúra

```
frontend/
├── dist/                             # Vite build output
│   ├── assets/
│   │   ├── index-DTQBSEXh.css        # Compiled CSS
│   │   └── js/                       # Code-split JS chunks
│   │       ├── index-*.js            # Main bundle
│   │       ├── vendor-*.js           # Vendor chunks
│   │       └── *Page-*.js            # Lazy-loaded pages
│   └── index.html                    # Entry HTML
│
├── src/
│   ├── App.tsx                       # Root component
│   ├── main.tsx                      # React entry point
│   ├── index.css                     # Global CSS
│   ├── vite-env.d.ts                 # Vite types
│   │
│   ├── components/
│   │   ├── DiagnosisCard.tsx         # Diagnózis kártya
│   │   ├── ErrorBoundary.tsx         # Error boundary
│   │   ├── Layout.tsx                # Fő layout
│   │   ├── VehicleSelector.tsx       # Jármű választó
│   │   │
│   │   ├── composite/                # Összetett komponensek
│   │   │   ├── Pagination/
│   │   │   ├── SearchInput/
│   │   │   ├── StatCard/
│   │   │   ├── Table/
│   │   │   ├── WizardStepper/
│   │   │   └── index.ts
│   │   │
│   │   ├── features/                 # Feature-specifikus
│   │   │   ├── diagnosis/
│   │   │   │   ├── AnalysisProgress.tsx
│   │   │   │   ├── DiagnosisForm.tsx
│   │   │   │   ├── RecentAnalysisList.tsx
│   │   │   │   └── index.ts
│   │   │   └── history/
│   │   │       ├── HistoryFilterBar.tsx
│   │   │       ├── HistoryStats.tsx
│   │   │       ├── HistoryTable.tsx
│   │   │       └── index.ts
│   │   │
│   │   ├── layouts/                  # Layout komponensek
│   │   │   ├── FloatingBottomBar/
│   │   │   ├── Header/
│   │   │   ├── PageContainer/
│   │   │   └── index.ts
│   │   │
│   │   ├── lib/                      # UI primitívek
│   │   │   ├── Badge/
│   │   │   ├── Button/
│   │   │   ├── Card/
│   │   │   ├── Input/
│   │   │   ├── Select/
│   │   │   ├── Textarea/
│   │   │   └── index.ts
│   │   │
│   │   └── ui/                       # Utility UI
│   │       ├── DTCAutocomplete.tsx
│   │       ├── ErrorMessage.tsx
│   │       ├── ErrorState.tsx
│   │       ├── LoadingSpinner.tsx
│   │       └── index.ts
│   │
│   ├── contexts/                     # React Context-ek
│   │   ├── AuthContext.tsx           # Auth state management
│   │   └── ToastContext.tsx          # Toast notifications
│   │
│   ├── hooks/                        # Custom React hooks
│   │   ├── index.ts
│   │   ├── useErrorHandler.ts
│   │   └── useVehicles.ts
│   │
│   ├── lib/                          # Utility függvények
│   │   └── utils.ts
│   │
│   ├── pages/                        # Oldal komponensek
│   │   ├── DTCDetailPage.tsx         # DTC részletek
│   │   ├── DiagnosisPage.tsx         # Diagnosztika
│   │   ├── ForgotPasswordPage.tsx    # Elfelejtett jelszó
│   │   ├── HistoryPage.tsx           # Előzmények
│   │   ├── HomePage.tsx              # Főoldal
│   │   ├── LoginPage.tsx             # Bejelentkezés
│   │   ├── NewDiagnosisPage.tsx      # Új diagnózis
│   │   ├── NotFoundPage.tsx          # 404
│   │   ├── RegisterPage.tsx          # Regisztráció
│   │   ├── ResetPasswordPage.tsx     # Jelszó reset
│   │   └── ResultPage.tsx            # Eredmény oldal
│   │
│   ├── services/                     # API szolgáltatások
│   │   ├── api.ts                    # Axios instance
│   │   ├── authService.ts            # Auth API
│   │   ├── diagnosisService.ts       # Diagnosis API
│   │   ├── dtcService.ts             # DTC API
│   │   ├── vehicleService.ts         # Vehicle API
│   │   ├── index.ts
│   │   └── hooks/                    # TanStack Query hooks
│   │       ├── index.ts
│   │       ├── useDTC.ts
│   │       ├── useDiagnosis.ts
│   │       └── useVehicle.ts
│   │
│   ├── styles/
│   │   └── design-tokens.css         # CSS custom properties
│   │
│   └── types/
│       └── vehicle.ts                # TypeScript típusok
│
├── Dockerfile                        # Dev Dockerfile
├── Dockerfile.prod                   # Prod Dockerfile
├── index.html                        # HTML template
├── package.json                      # NPM függőségek
├── package-lock.json                 # Lock file
├── postcss.config.js                 # PostCSS konfig
├── railway.toml                      # Railway konfig
├── tailwind.config.js                # TailwindCSS konfig
├── tsconfig.json                     # TypeScript konfig
├── tsconfig.node.json                # Node TypeScript konfig
└── vite.config.ts                    # Vite build konfig
```

---

## Dokumentáció Struktúra

```
docs/
├── API.md                            # API áttekintés
├── API_GUIDE.md                      # API használati útmutató
├── API_REFERENCE.md                  # API referencia
├── BACKUP.md                         # Backup stratégia
├── BUDGET_AND_RESOURCES.md           # Költségvetés és erőforrások
├── CLI.md                            # CLI eszközök dokumentáció
├── COWORK_BRIEF.md                   # Pályázat briefing (Cowork)
├── DEPLOYMENT.md                     # Deployment útmutató
├── DEVELOPMENT.md                    # Fejlesztői útmutató
├── GRANT_APPLICATION_SUMMARY.md      # Pályázati összefoglaló
├── HUNGARIAN_NLP.md                  # Magyar NLP dokumentáció
├── INSTALLATION.md                   # Telepítési útmutató
├── INTEGRATION_GUIDE.md              # Integráció útmutató
├── MIGRATIONS.md                     # DB migrációk
├── MONITORING.md                     # Monitoring beállítás
├── RAILWAY_DEPLOYMENT.md             # Railway deploy guide
├── README.md                         # Docs index
├── TECHNICAL_DESCRIPTION.md          # Technikai leírás
├── USER_MANUAL_HU.md                 # Magyar felhasználói kézikönyv
├── FILE_STRUCTURE.md                 # Ez a fájl
│
└── postman/                          # Postman collections
    ├── postman_collection.json
    ├── postman_environment_local.json
    └── postman_environment_production.json
```

---

## Scripts Struktúra

```
scripts/
├── cli/                              # CLI eszközök
│   ├── __init__.py
│   ├── autocognitix_cli.py           # Fő CLI
│   ├── diagtool.py                   # Diagnosztika CLI
│   └── setup.py                      # CLI setup
│
├── export/                           # Export scriptek
│   ├── __init__.py
│   ├── export_dtc_database.py
│   ├── export_full_backup.py
│   ├── export_neo4j_graph.py
│   └── export_qdrant_vectors.py
│
├── utils/                            # Utility modulok
│   ├── __init__.py
│   └── url_validator.py
│
├── checkpoints/                      # Checkpoint fájlok (gitignored)
│
├── # Import scriptek
├── import_back4app_vehicles.py       # Back4App jármű import
├── import_data.py                    # Általános import
├── import_dtcdb.py                   # DTCDB import
├── import_nhtsa_complaints.py        # NHTSA panasz import
├── import_nhtsa_recalls.py           # NHTSA recall import
├── import_obd_codes.py               # OBD kód import
├── import_obdb.py                    # OBDb import
├── import_obdb_github.py             # OBDb GitHub import
│
├── # Indexelő scriptek
├── index_all_to_qdrant.py            # Összes adat indexelés
├── index_qdrant.py                   # Alap indexelés
├── index_qdrant_full.py              # Teljes indexelés
├── index_qdrant_hubert.py            # HuBERT embedding indexelés
├── index_qdrant_robust.py            # Robust indexelés checkpoint-tal
├── init_qdrant.py                    # Qdrant inicializálás
│
├── # Neo4j scriptek
├── load_all_to_neo4j.py              # Összes adat betöltés
├── load_neo4j_robust.py              # Robust loader checkpoint-tal
├── seed_database.py                  # Alap seeding
├── seed_neo4j_aura.py                # Neo4j Aura seeding
├── seed_vehicles.py                  # Jármű seeding
├── setup_neo4j_indexes.py            # Neo4j index setup
├── expand_neo4j_graph.py             # Gráf bővítés
│
├── # Scraper scriptek
├── scrape_all_dtc_sources.py         # Összes DTC forrás
├── scrape_autocodes.py               # Autocodes.com
├── scrape_bbareman.py                # BBareman
├── scrape_dtcbase.py                 # DTCBase
├── scrape_engine_codes.py            # Motor kódok
├── scrape_klavkarr.py                # Klavkarr
├── scrape_obd_codes.py               # OBD kódok
├── scrape_repairpal.py               # RepairPal
├── scrape_troublecodes.py            # TroubleCodes
├── scrape_wikipedia_vehicles.py      # Wikipedia járművek
│
├── # Fordítás scriptek
├── translate_to_hungarian.py         # Magyar fordítás
├── fix_translations.py               # Fordítás javítás
├── fix_bad_translations.py           # Rossz fordítások javítása
├── continue_translations.py          # Fordítás folytatás
├── create_glossary.py                # Glossary generálás
│
├── # NHTSA scriptek
├── sync_nhtsa.py                     # NHTSA szinkronizálás
├── sync_nhtsa_complete.py            # Teljes NHTSA sync
├── sync_nhtsa_vehicles.py            # NHTSA jármű sync
│
├── # Utility scriptek
├── backup_data.py                    # Backup
├── benchmark.py                      # Teljesítmény teszt
├── data_sync.py                      # Adat szinkron
├── diagnose.py                       # CLI diagnózis
├── download_all_obdb.py              # OBDb letöltés
├── export_data.py                    # Adat export
├── export_openapi.py                 # OpenAPI export
├── generate_obdb_report.py           # OBDb report
├── health_check.py                   # Health check script
├── merge_dtc_sources.py              # DTC merge
├── validate_data.py                  # Adat validáció
├── add_more_symptoms.py              # Tünet bővítés
├── create_symptom_database.py        # Tünet DB
└── utils.py                          # Közös utility-k
```

---

## Docker & DevOps Struktúra

```
docker/
├── Dockerfile.backend.prod           # Backend prod image
├── Dockerfile.frontend.prod          # Frontend prod image
├── docker-compose.prod.yml           # Production compose
├── .env.production.example           # Prod env template
│
├── alertmanager/
│   └── alertmanager.yml              # Alert konfig
│
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   ├── dashboards.yml
│       │   └── json/
│       │       └── autocognitix-overview.json
│       └── datasources/
│           └── datasources.yml
│
└── prometheus/
    ├── prometheus.yml                # Prometheus konfig
    └── rules/
        └── alerts.yml                # Alert rules

traefik/
├── traefik.yml                       # Traefik konfig
└── dynamic.yml                       # Dynamic routing
```

---

## Fontos Konfigurációs Fájlok

| Fájl | Cél |
|------|-----|
| `.env` | Lokális környezeti változók |
| `.env.railway.example` | Railway env template |
| `CLAUDE.md` | Claude Code projekt kontextus |
| `railway.json` | Railway monorepo konfig |
| `docker-compose.yml` | Lokális fejlesztés |
| `backend/railway.toml` | Backend Railway deploy |
| `frontend/railway.toml` | Frontend Railway deploy |
| `backend/ruff.toml` | Python linter konfig |
| `backend/requirements.txt` | Python függőségek |
| `frontend/package.json` | NPM függőségek |
| `frontend/vite.config.ts` | Vite build konfig |
| `frontend/tailwind.config.js` | TailwindCSS konfig |

---

## Fájlok Száma (Összesítés)

| Kategória | Darab |
|-----------|-------|
| Backend Python fájlok | ~80 |
| Frontend TypeScript fájlok | ~70 |
| Teszt fájlok | ~25 |
| Dokumentáció (.md) | ~25 |
| Script fájlok | ~50 |
| Konfiguráció fájlok | ~30 |
| **Összesen** | **~280** |

---

*Dokumentum verzió: 1.0*
*Generálva: 2026-02-08*
