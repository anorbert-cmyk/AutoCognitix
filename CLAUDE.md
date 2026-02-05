# AutoCognitix - Claude Code Projekt Kontextus

## Projekt Áttekintés

**Cél:** AI-alapú gépjármű-diagnosztikai platform magyar nyelvtámogatással, hardver nélküli manuális DTC kód és tünet bevitellel.

**Státusz:** Sprint 2 befejezve, Railway deployment konfigurálva

**Deployment:** Railway (PostgreSQL + Redis) + Neo4j Aura + Qdrant Cloud

## Tech Stack

### Backend
- **Framework:** FastAPI + Pydantic V2
- **ORM:** SQLAlchemy 2.0 async + asyncpg
- **Adatbázisok:**
  - PostgreSQL 16 - strukturált adatok
  - Neo4j 5.x - diagnosztikai gráf (DTC → Symptom → Component → Repair)
  - Qdrant - vektor keresés (768-dim huBERT embeddings)
  - Redis - cache

### Frontend
- **Framework:** React 18 + TypeScript
- **Styling:** TailwindCSS
- **State:** TanStack Query
- **Build:** Vite

### AI/NLP
- **RAG:** LangChain
- **Magyar NLP:** huBERT (SZTAKI-HLT/hubert-base-cc), HuSpaCy
- **Embedding:** 768 dimenziós vektorok

## Projekt Struktúra

```
AutoCognitix/
├── backend/           # FastAPI alkalmazás
│   ├── app/
│   │   ├── api/v1/   # API végpontok
│   │   ├── core/     # Config, security, logging
│   │   ├── db/       # PostgreSQL, Neo4j, Qdrant
│   │   ├── services/ # Üzleti logika (KÉSZ)
│   │   └── nlp/      # Magyar NLP (services/embedding_service.py)
│   └── alembic/      # Migrációk
├── frontend/          # React alkalmazás
│   └── src/
│       ├── pages/    # Oldalak
│       ├── components/
│       └── services/ # API kliens
├── data/             # Adatfájlok (63 DTC kód KÉSZ)
└── scripts/          # Import scriptek (seed_database.py KÉSZ)
```

## Fontos Fájlok

- `docker-compose.yml` - Összes szolgáltatás
- `.env.example` - Environment változók template
- `backend/app/core/config.py` - Központi konfiguráció
- `backend/app/db/neo4j_models.py` - Gráf séma
- `backend/app/api/v1/schemas/diagnosis.py` - Fő API kontraktus

## Workflow Orchestration - MINDIG KÖTELEZŐ

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing new patterns

## Munkafolyamat Preferenciák

- **Párhuzamos ágensek:** Több Task agent egyidejű futtatása
- **Engedélyek:** Minden keresés/futtatás automatikusan engedélyezett
- **Todo lista:** Aktívan használva a haladás követésére

## API Végpontok

| Végpont | Státusz | Leírás |
|---------|---------|--------|
| `POST /api/v1/diagnosis/analyze` | Scaffold | Fő diagnosztika |
| `GET /api/v1/dtc/search` | Scaffold | DTC keresés |
| `POST /api/v1/vehicles/decode-vin` | Scaffold | VIN dekódolás |
| `POST /api/v1/auth/login` | Scaffold | Bejelentkezés |

## Adatforrások

### Ingyenes (implementálandó)
- NHTSA API (VIN, recalls, complaints)
- OBDb GitHub (738+ jármű repo)
- python-OBD könyvtár

### Fizetős (később)
- CarAPI, CarMD

## Tanulságok és Döntések

### 2024-02-03 - Projekt indítás
- Qdrant választva pgvector helyett (jobb teljesítmény)
- Neo4j gráf modell a diagnosztikai kapcsolatokhoz
- huBERT a magyar nyelvű embeddingekhez
- Monorepo struktúra (backend + frontend együtt)

## TODO - Következő Sprintek

### Sprint 2: Adatbázis réteg (FOLYAMATBAN)
- [ ] Alembic első migráció
- [x] Neo4j seed adatok (seed_database.py)
- [ ] Qdrant collection inicializálás

### Sprint 3: API implementáció
- [ ] Auth végpontok működőképessé
- [ ] DTC CRUD műveletek
- [ ] Vehicle repository
- [ ] API végpontok frissítése új szolgáltatásokkal

### Sprint 4: Adatforrás integráció (KÉSZ)
- [x] NHTSA API kliens (nhtsa_service.py)
- [ ] OBDb import script
- [x] Generikus DTC kódok importálása (63 kód)

### Sprint 5: AI/RAG (KÉSZ)
- [x] huBERT embedding service (embedding_service.py)
- [ ] Qdrant indexelés
- [x] LangChain RAG chain (rag_service.py)
- [x] Magyar prompt template
- [x] Diagnosis service (diagnosis_service.py)

## Deployment - Railway

### Architektúra

```
Railway Project
├── backend (FastAPI) ──────────┐
│   └── Dockerfile build        │
├── frontend (React) ───────────┤
│   └── Nixpacks build          │
├── PostgreSQL (Railway)        ├── Railway Private Network
├── Redis (Railway)             │
└── External Services           │
    ├── Neo4j Aura (cloud.neo4j.com)
    └── Qdrant Cloud (cloud.qdrant.io)
```

### Railway Services

| Service | Config File | Build |
|---------|-------------|-------|
| backend | `backend/railway.toml` | Dockerfile |
| frontend | `frontend/railway.toml` | Nixpacks |
| PostgreSQL | Railway Add-on | - |
| Redis | Railway Add-on | - |

### Deployment Lépések

1. **Railway Projekt létrehozása:**
   ```bash
   railway login
   railway init
   ```

2. **Adatbázisok hozzáadása:**
   - PostgreSQL: Railway Dashboard → New → Database → PostgreSQL
   - Redis: Railway Dashboard → New → Database → Redis

3. **Külső szolgáltatások:**
   - Neo4j Aura: https://cloud.neo4j.com (Free tier)
   - Qdrant Cloud: https://cloud.qdrant.io (Free tier)

4. **Environment Variables:**
   - Lásd: `.env.railway.example`
   - Railway Dashboard → Service → Variables

5. **Deploy:**
   ```bash
   # Backend
   cd backend && railway up

   # Frontend
   cd frontend && railway up
   ```

### Fontos Environment Variables

```
# Railway automatikusan beállítja
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
PORT=...

# Kézi beállítás szükséges
NEO4J_URI=neo4j+s://xxx.databases.neo4j.io
NEO4J_PASSWORD=...
QDRANT_URL=https://xxx.cloud.qdrant.io:6333
QDRANT_API_KEY=...
ANTHROPIC_API_KEY=... (vagy OPENAI_API_KEY)
JWT_SECRET_KEY=...
```

## Gyakori Parancsok

### Lokális Fejlesztés

```bash
# Fejlesztői környezet indítása
docker-compose up -d

# Backend futtatása (dev)
cd backend && uvicorn app.main:app --reload

# Frontend futtatása (dev)
cd frontend && npm run dev

# Migráció létrehozása
cd backend && alembic revision --autogenerate -m "description"

# Migráció futtatása
cd backend && alembic upgrade head
```

### Railway Deployment

```bash
# Railway CLI telepítés
npm install -g @railway/cli

# Bejelentkezés
railway login

# Projekt inicializálás
railway init

# Deploy
railway up

# Logok megtekintése
railway logs

# Environment változók
railway variables
```

## CI/CD Pipeline - KÖTELEZŐ ELLENŐRZÉSEK

### Commit Előtt MINDIG Futtasd:

```bash
# 1. Ruff linting
cd backend && python3 -m ruff check app tests

# 2. Ruff formatting
cd backend && python3 -m ruff format --check app tests

# 3. Ha hibák vannak, automatikus javítás:
cd backend && python3 -m ruff check app tests --fix --unsafe-fixes
```

### Ruff Konfiguráció (backend/ruff.toml)

A következő hibák IGNORÁLVA vannak:
- `UP035/UP006/UP045`: Modern typing syntax (Python 3.9+ dict/list)
- `PLC0415`: Lazy imports (FastAPI szükséges)
- `PLW0603`: Global statement (singleton pattern)
- `ERA001`: TODO comments
- `I001`: Import sorting (handled by formatter)

### GitHub Actions

| Workflow | Trigger | Cél |
|----------|---------|-----|
| `ci.yml` | push/PR | Lint, Type Check, Tests, Build |
| `cd.yml` | release/tag | Docker Build, Deploy to Railway |
| `security.yml` | daily/push | CodeQL, Bandit, npm audit |

### Ha CI Hibázik

1. Nézd meg a logokat: `gh run view <run-id> --log-failed`
2. Lint hibák: `ruff check app tests --fix`
3. Type hibák: `mypy app --ignore-missing-imports`
4. Test hibák: `pytest tests -v`

## Kapcsolódó Dokumentumok

- `AutoCognitix_Teljeskoeru_Elemzes.docx` - Részletes elemzés
- `MVP Definíció & Gyakorlati Megvalósítás.pdf` - MVP specifikáció
- `tasks/lessons.md` - Tanulságok és hibajavítások részletesen
