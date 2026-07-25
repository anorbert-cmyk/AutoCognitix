# AutoCognitix - Railway Deployment Guide

## Architektúra Áttekintés

```
┌─────────────────────────────────────────────────────────────────┐
│                      Railway Project                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Backend   │    │  Frontend   │    │ PostgreSQL  │          │
│  │  (FastAPI)  │───▶│   (React)   │    │  (Railway)  │          │
│  │   :$PORT    │    │   :$PORT    │    │   :5432     │          │
│  └──────┬──────┘    └─────────────┘    └──────┬──────┘          │
│         │                                      │                 │
│         └──────────────────────────────────────┘                 │
│                            │                                     │
│                      ┌─────┴─────┐                               │
│                      │   Redis   │                               │
│                      │ (Railway) │                               │
│                      └───────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Services                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │    Neo4j Aura   │              │  Qdrant Cloud   │           │
│  │ (Graph Database)│              │ (Vector Search) │           │
│  │ cloud.neo4j.com │              │ cloud.qdrant.io │           │
│  └─────────────────┘              └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## Előfeltételek

1. **Railway Account:** https://railway.app
2. **GitHub Repository:** Projekt GitHub-ra push-olva
3. **Neo4j Aura Account:** https://cloud.neo4j.com
4. **Qdrant Cloud Account:** https://cloud.qdrant.io

## 1. Railway CLI Telepítés

```bash
# NPM-mel
npm install -g @railway/cli

# Homebrew (macOS)
brew install railway

# Ellenőrzés
railway --version
```

## 2. Railway Projekt Létrehozása

```bash
# Bejelentkezés
railway login

# Új projekt (a repository mappában)
cd AutoCognitix
railway init

# Vagy meglévő projekt összekapcsolása
railway link
```

## 3. Adatbázisok Létrehozása Railway-en

### 3.1 PostgreSQL

1. Railway Dashboard → Project → **New** → **Database** → **PostgreSQL**
2. Várj amíg a database elindul
3. A `DATABASE_URL` automatikusan elérhető lesz

### 3.2 Redis

1. Railway Dashboard → Project → **New** → **Database** → **Redis**
2. A `REDIS_URL` automatikusan elérhető lesz

## 4. Külső Szolgáltatások Beállítása

### 4.1 Neo4j Aura (Graph Database)

1. Menj a https://cloud.neo4j.com oldalra
2. Create **Free Instance**
3. Válaszd: **AuraDB Free** (50k nodes, 175k relationships)
4. Mentsd el:
   - **Connection URI:** `neo4j+s://xxxxxxxx.databases.neo4j.io`
   - **Username:** `neo4j`
   - **Password:** (generált jelszó)

### 4.2 Qdrant Cloud (Vector Database)

1. Menj a https://cloud.qdrant.io oldalra
2. Create **Free Cluster**
3. Válaszd: **Free** (1GB storage)
4. Mentsd el:
   - **Cluster URL:** `https://xxxxxxxx.cloud.qdrant.io:6333`
   - **API Key:** (generált kulcs)

## 5. Backend Deploy

### 5.1 Service Létrehozása

1. Railway Dashboard → Project → **New** → **GitHub Repo**
2. Válaszd ki az AutoCognitix repót
3. **Root Directory:** `backend`

### 5.2 Environment Variables

Railway Dashboard → Backend Service → **Variables** → Add:

```env
# Application
ENVIRONMENT=production
DEBUG=false
API_V1_PREFIX=/api/v1

# Security
JWT_SECRET_KEY=<generálj-egy-erős-kulcsot>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Cross-site auth (KÖTELEZŐ, ha a frontend és a backend külön domainen fut,
# pl. külön *.up.railway.app hostokon). Az auth 100%-ban httpOnly cookie-alapú:
# - SameSite=None + Secure nélkül a böngésző cross-site NEM küldi el az auth
#   cookie-kat -> a felhasználó login után azonnal kijelentkezettnek látszik.
# - A SameSite=None-hoz Secure=true KELL (a config validator ezt ki is kényszeríti).
COOKIE_SAMESITE=none
COOKIE_SECURE=true

# CORS - BIZTONSÁG-KRITIKUS. A CSRF-védelem header-only, aláírt tokenre épül,
# aminek a valódi garanciája a custom X-CSRF-Token header + CORS preflight.
# Ezért a BACKEND_CORS_ORIGINS-nek a PONTOS frontend origin(eke)t kell tartalmaznia
# (vesszővel elválasztva). SOHA ne legyen "*" wildcard - az feloldaná a CSRF-védelmet
# (és allow_credentials=True mellett a böngésző úgyis elutasítja).
BACKEND_CORS_ORIGINS=https://<frontend-service>.up.railway.app

# Database (Railway automatikusan beállítja, ha linkelve van)
DATABASE_URL=${{Postgres.DATABASE_URL}}
REDIS_URL=${{Redis.REDIS_URL}}

# Neo4j Aura
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<neo4j-aura-password>

# Qdrant Cloud
QDRANT_URL=https://xxxxxxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=<qdrant-api-key>

# AI Provider (válassz egyet)
ANTHROPIC_API_KEY=<anthropic-key>
# vagy
OPENAI_API_KEY=<openai-key>

# Magyar embedding (huBERT). A modell revíziója COMMIT SHA-ra van pinelve:
# a Qdrantban lévő ~54 000 vektor EZEKKEL a súlyokkal készült. Ha ezt a
# változót "main"-re állítod, felülírod a pint és szétcsúszik az embedding-tér.
HUBERT_REVISION=028baac7feb87a7b2f042bbdaa5deec6513c6060
# ONNX Runtime intra-op szálak SESSION-önként (minden workernek saját session-je van).
EMBEDDING_ORT_THREADS=1

# Gunicorn worker-ek száma. Lásd 5.3 – ez a RAM-szabályozó.
WEB_CONCURRENCY=2
```

### 5.3 Embedding Backend, Memória és Worker-ek

A backend a magyar embeddingeket **ONNX Runtime**-mal futtatja: a huBERT gráf a
`Dockerfile.prod` eldobott `onnx-export` stage-ében készül, és a runtime image
`onnxruntime` + `tokenizers` párost visz, **torch nélkül**.

#### RAM-igény

| Tétel | Mért érték |
|-------|-----------|
| Csúcs RSS / worker (ONNX út) | ~595 MB |
| `WEB_CONCURRENCY=2`, állandósult | ~1,2 GB |
| `WEB_CONCURRENCY=2`, tranziens (mindkét worker egyszerre tölti be a gráfot) | ~2 GB |

A modell **worker-enként** töltődik be, tehát a RAM lineárisan skálázódik a
worker-számmal. **Minimum 2 GB** RAM kell, **4 GB kényelmes**. A `/health`
statikus JSON, az ONNX session pedig lustán, az első embed híváskor épül fel
(~0,9 s) – a boot költségét továbbra is az `alembic upgrade head` dominálja.

#### Új environment változók

| Változó | Alapérték | Mire való |
|---------|-----------|-----------|
| `WEB_CONCURRENCY` | `2` | Gunicorn worker-ek száma. **Változó, nem beépített literál** – lásd a recovery runbookot. |
| `EMBEDDING_BACKEND` | `onnx` (az image állítja be) | `onnx` / `disabled`. **A `torch` a prod image-ben NEM működik** (nincs telepítve torch), tehát ez kill switch, nem backend-váltó. |
| `HUBERT_ONNX_PATH` | `/app/models/hubert_fp32.onnx` | Az exportált gráf helye. Env-ből állítható, hogy egy rossz útvonal Railway-változó-javítás legyen, ne rebuild. |
| `HUBERT_VOCAB_PATH` | `/app/models/vocab.txt` | WordPiece vocab (`lowercase=False`). |
| `EMBEDDING_ORT_THREADS` | `1` | ORT intra-op szálak **session-önként**. A teljes szálbüdzsé szorzat: `WEB_CONCURRENCY × embedding pool (2) × EMBEDDING_ORT_THREADS`. 2 vCPU-n az alapérték 4 szálat jelent. |
| `HUBERT_REVISION` | pinelt commit SHA | **Soha ne legyen `main`.** Változtatása = fixture-újragenerálás + teljes Qdrant reindex. |

#### Recovery runbook

A `railway.toml`-ban `restartPolicyMaxRetries = 3`: **három OOM-kill után a
Railway végleg abbahagyja az újraindítást**, tehát az „építsünk újat kevesebb
workerrel" nem recovery-opció. Sorrendben:

1. **OOM / memóriaszűke** → `WEB_CONCURRENCY=1` a Dashboard Variables alatt +
   restart. Nincs rebuild, ~30 s. Ha ez sem elég, nagyobb plan.
2. **Az embedding rossz eredményt ad, de a service él** →
   `EMBEDDING_BACKEND=disabled` + restart. A build megmarad, a szemantikus
   keresés kikapcsol, a lexikai és a Neo4j út tovább szolgál, és a
   `/health/detailed` `degraded`-ként jelzi. Ez **nem** a régi néma nullvektor:
   a rendszer tudja és jelenti, hogy degradált.
3. **A deploy egésze rossz** → **Railway Dashboard → Deployments → Rollback** az
   előző image-re. Ez az egyetlen valódi rebuild nélküli teljes visszaállás; a
   `EMBEDDING_BACKEND` csak a szemantikus ágat kapcsolja ki.
4. **Diagnózis** → `GET /health/detailed` (auth kell) megmondja, melyik backend
   aktív (`onnx` / `torch` / `none`) és lefuttat egy élő self-testet. Az
   embedding-próbának saját 5 s-os időkerete van, tehát egy lassú modellbetöltés
   nem rántja magával a többi adatbázis health-státuszát.

### 5.4 Deploy

```bash
cd backend
railway up
```

Vagy **Automatic Deploys** engedélyezése a Dashboard-on (push-ra automatikusan deploy-ol).

## 6. Frontend Deploy

### 6.1 Service Létrehozása

1. Railway Dashboard → Project → **New** → **GitHub Repo**
2. Válaszd ki ugyanazt a repót
3. **Root Directory:** `frontend`

### 6.2 Environment Variables

```env
VITE_API_URL=https://<backend-service>.railway.app/api/v1
```

### 6.3 Deploy

```bash
cd frontend
railway up
```

## 7. Database Migration

A backend első indulásakor futtasd:

```bash
# Railway shell-ben
railway run alembic upgrade head

# Vagy a backend service-ben
railway run --service backend alembic upgrade head
```

## 8. Seed Data (Opcionális)

```bash
# DTC kódok és alapadatok betöltése
railway run python scripts/seed_database.py

# Qdrant indexelés
railway run python scripts/index_qdrant.py
```

## 9. Domain és HTTPS

Railway automatikusan biztosít:
- **Subdomain:** `<service-name>.railway.app`
- **HTTPS:** Automatikus SSL

Egyedi domain:
1. Dashboard → Service → **Settings** → **Domains**
2. Add **Custom Domain**
3. Állítsd be a DNS CNAME rekordot

## 10. Monitoring

### Railway Dashboard

- **Logs:** Valós idejű logok
- **Metrics:** CPU, Memory, Network
- **Deployments:** Deploy history

### Hasznos Parancsok

```bash
# Logok megtekintése
railway logs

# Specifikus service logok
railway logs --service backend

# Environment változók listázása
railway variables

# Shell a service-ben
railway shell
```

## Költségek

### Railway
- **Free Tier:** $5 kredit/hó (elegendő fejlesztéshez)
- **Hobby Plan:** $5/hó (500 óra, jobb limitekek)

### Neo4j Aura
- **Free Tier:** 50k nodes, 175k relationships (elegendő MVP-hez)

### Qdrant Cloud
- **Free Tier:** 1GB storage (elegendő ~100k vektorhoz)

## Troubleshooting

### Build Hiba

```bash
# Lokális build teszt
docker build -t autocognitix-backend ./backend

# Railway build logok
railway logs --build
```

### Database Connection

```bash
# Connection string ellenőrzés
railway variables | grep DATABASE_URL

# PostgreSQL direkt kapcsolat
railway connect postgres
```

### Health Check Hiba

A backend `/health` végpontja kell működjön:
```bash
curl https://<backend>.railway.app/health
```

## Checklist

- [ ] Railway account létrehozva
- [ ] PostgreSQL service futtatva
- [ ] Redis service futtatva
- [ ] Neo4j Aura instance létrehozva
- [ ] Qdrant Cloud cluster létrehozva
- [ ] Backend service deploy-olva
- [ ] Frontend service deploy-olva
- [ ] Environment variables beállítva
- [ ] **Cross-site auth: `COOKIE_SAMESITE=none` + `COOKIE_SECURE=true`** (különben a login után azonnali kijelentkezés)
- [ ] **`BACKEND_CORS_ORIGINS` = a pontos frontend origin** (nincs `*` wildcard — a CSRF-védelem ezen múlik)
- [ ] **`JWT_SECRET_KEY` beállítva és stabil** (rotáció érvényteleníti a kint lévő CSRF tokeneket)
- [ ] Database migration lefutott
- [ ] Seed data betöltve
- [ ] Health check működik
- [ ] Smoke teszt: böngészőből login → 200, majd egy írás (pl. jármű hozzáadása) → 200 (nem 403)
