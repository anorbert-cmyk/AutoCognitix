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
```

### 5.3 Deploy

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
- [ ] Database migration lefutott
- [ ] Seed data betöltve
- [ ] Health check működik
