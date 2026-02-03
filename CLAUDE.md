# AutoCognitix - Claude Code Projekt Kontextus

## Projekt Áttekintés

**Cél:** AI-alapú gépjármű-diagnosztikai platform magyar nyelvtámogatással, hardver nélküli manuális DTC kód és tünet bevitellel.

**Státusz:** Sprint 1 befejezve (Projekt scaffold)

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
│   │   ├── services/ # Üzleti logika (TODO)
│   │   └── nlp/      # Magyar NLP (TODO)
│   └── alembic/      # Migrációk
├── frontend/          # React alkalmazás
│   └── src/
│       ├── pages/    # Oldalak
│       ├── components/
│       └── services/ # API kliens
├── data/             # Adatfájlok (TODO)
└── scripts/          # Import scriptek (TODO)
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

### Sprint 2: Adatbázis réteg
- [ ] Alembic első migráció
- [ ] Neo4j seed adatok
- [ ] Qdrant collection inicializálás

### Sprint 3: API implementáció
- [ ] Auth végpontok működőképessé
- [ ] DTC CRUD műveletek
- [ ] Vehicle repository

### Sprint 4: Adatforrás integráció
- [ ] NHTSA API kliens
- [ ] OBDb import script
- [ ] Generikus DTC kódok importálása

### Sprint 5: AI/RAG
- [ ] huBERT embedding service
- [ ] Qdrant indexelés
- [ ] LangChain RAG chain
- [ ] Magyar prompt template

## Gyakori Parancsok

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

## Kapcsolódó Dokumentumok

- `AutoCognitix_Teljeskoeru_Elemzes.docx` - Részletes elemzés
- `MVP Definíció & Gyakorlati Megvalósítás.pdf` - MVP specifikáció
