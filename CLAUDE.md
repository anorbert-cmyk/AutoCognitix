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

## Kapcsolódó Dokumentumok

- `AutoCognitix_Teljeskoeru_Elemzes.docx` - Részletes elemzés
- `MVP Definíció & Gyakorlati Megvalósítás.pdf` - MVP specifikáció
