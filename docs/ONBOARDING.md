# AutoCognitix - Onboarding Guide

Ez a dokumentum abban segít, hogy egy új vezető vagy szakértő **2 óra alatt** megértse az AutoCognitix rendszerét, lokálisan elindítsa, és lefuttassa az első diagnózist.

---

## 1. Belépési pont - olvasási sorrend

Az alábbi sorrendben olvasd el a dokumentumokat (kb. 45 perc):

1. **`CLAUDE.md`** (projekt root) - Projekt kontextus, sprint státusz, DB adatok, workflow szabályok. A legfrissebb forrás az aktuális állapotról.
2. **`PROJECT_OVERVIEW.md`** (projekt root) - Üzleti cél, funkcionális/nem-funkcionális követelmények, tech stack áttekintés.
3. **`README.md`** (projekt root) - Gyors start, tech stack tábla, API végpont lista, CI/CD áttekintés.
4. **`docs/ARCHITECTURE.md`** - Rétegek, 4 adatbázis, AI pipeline, mermaid diagram.
5. **`docs/DATA_FLOW.md`** - Egy konkrét diagnózis kérés útja frontend → backend → DB-k → LLM → válasz.
6. **`docs/DATABASE_MAP.md`** - Melyik adat melyik DB-ben, pontos tábla/node/collection nevek.

Támogató olvasmány (opcionális): `docs/INSTALLATION.md`, `docs/API_REFERENCE.md`, `docs/DEVELOPMENT.md`.

---

## 2. Repo térkép

| Útvonal | Tartalom egy mondatban |
|---------|------------------------|
| `backend/` | FastAPI backend (Python 3.11+), SQLAlchemy async, Pydantic V2, szolgáltatás-réteg, Alembic migrációk. |
| `backend/app/api/v1/` | HTTP végpontok (`endpoints/`), Pydantic sémák (`schemas/`), router aggregáció (`router.py`). |
| `backend/app/services/` | Üzleti logika: `diagnosis_service.py`, `rag_service.py`, `embedding_service.py`, `nhtsa_service.py`, `vehicle_garage_service.py`, stb. |
| `backend/app/db/` | Adatbázis-réteg: `postgres/`, `neo4j_models.py`, `qdrant_client.py`, `redis_cache.py`. |
| `backend/app/core/` | Config, biztonság, rate limit, logolás, naplózási szanitizálás. |
| `backend/alembic/versions/` | 18+ SQL migráció (initial schema -> garage táblák -> FK constraints). |
| `frontend/` | React 18 + TypeScript + Vite + Tailwind + TanStack Query SPA. |
| `frontend/src/pages/` | Oldal-komponensek (DiagnosisPage, ResultPage, GaragePage, DemoResultPage stb.). |
| `frontend/src/services/` | API kliens (`api.ts`, `diagnosisService.ts`, `garageService.ts`, hooks). |
| `scripts/` | Adatimport, scrape, Qdrant indexelő, Neo4j seed, fordítás és sync scriptek. |
| `data/` | DTC kódforrások (`dtc_codes/`), NHTSA dumpok, EPA fueleconomy adatok, backupok, `parts_mapping.json`. |
| `docs/` | Ez a mappa - onboarding, architektúra, data flow, DB map, API ref, deployment, stb. |
| `docker/` | Termelési Dockerfile-ok (backend, frontend), nginx, Prometheus, Grafana, Alertmanager config. |
| `docker-compose.yml` | Lokális fejlesztési stack: Postgres + Neo4j + Qdrant + Redis + backend + frontend. |
| `.github/workflows/` | CI/CD: `ci.yml` (lint + test + build), `cd.yml` (Docker + Railway deploy), `security.yml` (CodeQL, Bandit, Trivy). |
| `tasks/` | Sprint todo lista (`todo.md`) és tanulságok (`lessons.md`). |
| `n8n-workflows/` | Opcionális N8N automatizációs workflowk. |
| `traefik/` | Opcionális reverse-proxy config termelési deployhoz. |

---

## 3. Lokális futtatás 5 lépésben

> Feltétel: Docker + Docker Compose, Python 3.11+, Node.js 20+ telepítve.

### 1. Klónozás
```bash
git clone https://github.com/norbertbarna/AutoCognitix.git
cd AutoCognitix
```

### 2. `.env` konfigurálás
```bash
cp .env.example .env
# Kötelezően beállítandó:
#   SECRET_KEY               (openssl rand -hex 32 - min. 32 karakter!)
#   JWT_SECRET_KEY           (openssl rand -hex 32 - min. 32 karakter!)
#   ANTHROPIC_API_KEY        (vagy OPENAI_API_KEY az LLM hívásokhoz)
#   NEO4J_PASSWORD           (lokálisan bármi, pl. autocognitix_dev)
# Opcionális (Cloud használatakor):
#   NEO4J_URI, QDRANT_URL, QDRANT_API_KEY
```

### 3. Szolgáltatások indítása
```bash
docker-compose up -d
# Elindul: postgres:5432, neo4j:7687/7474, qdrant:6333, redis:6379,
#          backend:8000, frontend:3000
```

Ellenőrzés: `docker-compose ps` - minden service `healthy` kell legyen.

### 4. Adatbázis migráció
```bash
docker-compose exec backend alembic upgrade head
# Létrehozza a 20+ Postgres táblát (users, dtc_codes, diagnosis_sessions,
# user_vehicles, vehicle_recalls, stb.)
```

### 5. Seed adatok betöltése
```bash
# Postgres DTC kódok + jármű adatok:
docker-compose exec backend python /app/scripts/seed_database.py

# Neo4j gráf (26k+ node):
docker-compose exec backend python /app/scripts/seed_neo4j_aura.py

# Qdrant HuBERT vektorok (35k+ embedding, ~30 perc):
docker-compose exec backend python /app/scripts/index_qdrant_hubert.py
```

UI elérés: `http://localhost:3000`, API docs: `http://localhost:8000/docs`.

---

## 4. Első diagnózis kérés (curl)

```bash
curl -X POST http://localhost:8000/api/v1/diagnosis/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "dtc_codes": ["P0300", "P0301"],
    "symptoms": "A motor rázkódik alapjáraton, gyorsításnál teljesítményvesztés, égéskimaradás érzékelhető."
  }'
```

Várható válasz (kivonat):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "vehicle_make": "Volkswagen",
  "dtc_codes": ["P0300", "P0301"],
  "probable_causes": [ { "title": "...", "confidence": 0.85, ... } ],
  "recommended_repairs": [ { "title": "...", "estimated_cost_min": 5000, ... } ],
  "parts_with_prices": [ ... ],
  "total_cost_estimate": { ... },
  "confidence_score": 0.82,
  "sources": [ ... ]
}
```

Bemutató (pre-filled P0300 szimuláció): `http://localhost:3000/demo`.

---

## 5. Gyakori problémák (FAQ)

### P1. `Startup failed: SECRET_KEY must be at least 32 characters long`
**Ok:** A `.env` fájlban a `SECRET_KEY` vagy `JWT_SECRET_KEY` hiányzik / túl rövid.
**Megoldás:** `openssl rand -hex 32` paranccsal generálj értéket mindkét változónak, majd `docker-compose restart backend`.
**Forrás:** `backend/app/core/config.py` - `validate_secrets` validator.

### P2. `Neo4j unavailable - using PostgreSQL-only fallback` (warning)
**Ok:** A `neo4j` konténer még fel sem állt, vagy az `NEO4J_URI` / `NEO4J_PASSWORD` rosszul van beállítva. A rendszer **nem dől össze** - graceful degradation van beépítve.
**Megoldás:** `docker-compose logs neo4j` - ha healthcheck hibázik, töröld a volumeot: `docker-compose down -v && docker-compose up -d`.
**Forrás:** `backend/app/db/neo4j_models.py::is_neo4j_available()`.

### P3. Qdrant keresés 0 eredményt ad, pedig vannak vektorok
**Ok:** Az `index_qdrant_hubert.py` még nem futott le végig, vagy az embedding model verzió (`hubert-base-cc-v1`) nem egyezik. A collection nevek kötelezően `*_hu` szuffixummal végződnek (huBERT 768-dim).
**Megoldás:** Ellenőrizd: `docker-compose exec backend python -c "from app.db.qdrant_client import qdrant_client; import asyncio; print(asyncio.run(qdrant_client.get_storage_stats()))"`.
**Forrás:** `backend/app/db/qdrant_client.py` - `DTC_COLLECTION = "dtc_embeddings_hu"`.

### P4. `Rate limit exceeded` - 429-es válasz diagnózis indításakor
**Ok:** A rate limiter **fail-closed** (Sprint 9-es security hardening): ha Redis elérhetetlen, a kérés tiltva. Az alapértelmezett limit diagnózisra alacsony.
**Megoldás:** `docker-compose logs redis` ellenőrzés; fejlesztés közben állítsd át a `.env`-ben az `ENVIRONMENT=development`-et.
**Forrás:** `backend/app/core/rate_limit.py`, `backend/app/db/redis_cache.py::check_rate_limit()`.

### P5. `ModuleNotFoundError: huspacy` vagy HuBERT nem tölt be
**Ok:** A `SZTAKI-HLT/hubert-base-cc` model ~500MB, első indításkor tölt le a HuggingFace-ről. Internet hiányában nem megy.
**Megoldás:** `docker-compose exec backend python -c "from transformers import AutoModel; AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc')"` - kézi előtöltés. HuSpaCy nélkül is működik a preprocess (csak logban warning lesz).
**Forrás:** `backend/app/services/embedding_service.py::_load_hubert_model()`.

---

## Hasznos parancsok

```bash
# Logok figyelése
docker-compose logs -f backend

# Backend shell
docker-compose exec backend bash

# Lint + format (commit előtt kötelező)
cd backend && python3 -m ruff check app tests && python3 -m ruff format --check app tests

# Tesztek
cd backend && pytest tests -v
cd frontend && npm run test

# API docs böngészőben
open http://localhost:8000/docs
```
