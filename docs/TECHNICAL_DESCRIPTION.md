# AutoCognitix - Reszletes Technikai Leiras
# Detailed Technical Description

---

## 1. Rendszer Architektura / System Architecture

### 1.1 Attekintes / Overview

Az AutoCognitix egy mikroszolgaltatas-orientalt architekturara epul, negy adatbazissal es AI-tamogatott diagnosztikai pipeline-nal:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LOAD BALANCER                                │
│                         (Railway Ingress)                              │
└────────────────────────────────┬───────────────────────────────────────┘
                                 │
         ┌───────────────────────┴───────────────────────┐
         │                                               │
         ▼                                               ▼
┌─────────────────────┐                     ┌─────────────────────┐
│    FRONTEND         │                     │     BACKEND         │
│    React 18 SPA     │◄───── REST API ────►│     FastAPI         │
│    11 oldal         │                     │     (Async)         │
└─────────────────────┘                     └──────────┬──────────┘
                                                       │
              ┌──────────┬──────────┬──────────────────┼──────────────┐
              │          │          │                   │              │
              ▼          ▼          ▼                   ▼              ▼
     ┌────────────┐ ┌────────┐ ┌────────┐     ┌──────────────┐ ┌──────────┐
     │ PostgreSQL │ │ Neo4j  │ │Qdrant  │     │  NHTSA API   │ │  Redis   │
     │ 790K+ rec  │ │ 65K+   │ │ 55K+   │     │  (Elo API)   │ │ (Cache)  │
     │ 18 tabla   │ │ node   │ │ vektor │     │              │ │          │
     └────────────┘ └────────┘ └────────┘     └──────────────┘ └──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │    HuBERT Model     │
                         │  (768-dim embedding) │
                         │   SZTAKI-HLT        │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         │   LLM Provider      │
                         │ (Claude / GPT-4)    │
                         └─────────────────────┘
```

### 1.2 Komponensek / Component Details

#### Frontend (React 18 + TypeScript)
- **11 oldal**: Home, Diagnosis, Result, DTCDetail, History, NewDiagnosis, Login, Register, ResetPassword, ForgotPassword, NotFound
- **Technologiak**: React 18, TypeScript 5, TailwindCSS 3, TanStack Query, React Router 6, Vite 5
- **Responsiv**: Mobil-optimalizalt felulet

#### Backend (FastAPI + Python 3.9)
- **7 API endpoint csoport**: auth, diagnosis, dtc_codes, vehicles, health, metrics
- **Technologiak**: FastAPI, Pydantic V2, SQLAlchemy 2.0 async, asyncpg, neo4j-driver, qdrant-client, redis-py
- **12 Alembic migracio**: users, vehicles, dtc_codes, diagnosis_sessions, complaints, recalls, epa_vehicles

---

## 2. Adatbazis Architektura / Database Architecture

### 2.1 PostgreSQL 16 - Strukturalt Adatok (790 000+ rekord)

| Tabla | Rekordszam | Funkcios |
|-------|-----------|----------|
| dtc_codes | 6 814 | DTC kodok magyar/angol leirassal |
| vehicle_makes | 89 | Jarmugyartok |
| vehicle_models | 2 192 | Jarmumodellek evjarattal |
| vehicle_complaints | 751 422 | NHTSA panaszok (indexelve) |
| epa_vehicles | 31 603 | EPA jarmutechnikai adatok |
| users | - | Felhasznalok (JWT auth) |
| diagnosis_sessions | - | Diagnosztikai munkamenetek |
| vehicle_engines | - | Motor konfiguraciok |
| vehicle_recalls | - | Visszahivasok |

**Indexek (15+ teljesitmeny index):**
- GIN index DTC kod tomb kereshez
- Composite index jarmu kereshez (make + model + year)
- Full-text index NHTSA panasz szovegkereshez
- B-tree indexek az EPA make/model/year-re

### 2.2 Neo4j 5.x - Diagnosztikai Tudagraf (65 207 node)

**Node tipusok es mennyisegek:**

| Tipus | Darabszam | Tartalom |
|-------|-----------|----------|
| DTCCode | 6 814 | Hibakodok (code, leiras, severity, category) |
| Vehicle | 7 289 | Jarmuvek (make, model, evjarat) |
| Engine | 1 104 | Motorkonfiguraciok (hengerszam, uzemanyag, turbo) |
| Complaint | 50 000 | NHTSA panaszok (sulyossag szerint szurve) |

**Kapcsolat tipusok:**
```cypher
(dtc:DTCCode)-[:MENTIONS_DTC]->(complaint:Complaint)   // 107 DTC-panasz kapcsolat
(vehicle:Vehicle)-[:HAS_COMPLAINT]->(complaint:Complaint) // Jarmu-panasz
(vehicle:Vehicle)-[:USES_ENGINE]->(engine:Engine)         // Jarmu-motor
(dtc:DTCCode)-[:CAUSES]->(symptom:Symptom)               // DTC-tunet
(dtc:DTCCode)-[:AFFECTS]->(component:Component)           // DTC-alkatresz
(component:Component)-[:REPAIRED_BY]->(repair:Repair)     // Alkatresz-javitas
```

### 2.3 Qdrant Cloud - Vektor Adatbazis (54 652 embedding)

- **Dimenzio**: 768 (SZTAKI HuBERT)
- **Tavolsag**: Cosine similarity
- **Collection**: `autocognitix`
- **Tartalom**: DTC leirasok, NHTSA panasz szovegek, EPA adatok
- **Kereses**: Szemantikus hasonlosag alapjan (magyar es angol)

### 2.4 Redis 7 - Cache Reteg

**Tiered TTL strategia:**

| Kategoria | TTL | Tartalom |
|-----------|-----|----------|
| DTC kod | 1 ora | Egyedi DTC lekerdezes |
| Kereses | 15 perc | DTC/tunet kereses |
| Embedding | 1 ora | HuBERT vektor cache |
| NHTSA recall | 6 ora | Visszahivas adatok |
| Jarmu info | 24 ora | Make/model cache |

---

## 3. AI Diagnosztikai Pipeline

### 3.1 Teljes Pipeline Folyamatabra

```
FELHASZNALOI BEMENET
  │  DTC kod(ok): pl. P0171, P0101
  │  Tunetek: "A motor nehezen indul hidegben"
  │  Jarmu: VW Golf 2018
  │  VIN (opcionalis)
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  1. BEMENET VALIDALAS ES ELOFELDOLGOZAS             │
│                                                      │
│  • VIN dekodolas (NHTSA API) → jarmuazonositas      │
│  • DTC kod validalas (PostgreSQL) → leiras, severity │
│  • Magyar tunet elofeldolgozas (HuSpaCy)            │
│    - Lemmatizacio                                    │
│    - Stopword szures                                 │
│    - Automotive terminologia normalizalas             │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  2. PARHUZAMOS ADATGYUJTES (async)                  │
│                                                      │
│  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ NHTSA API (elo)  │  │ Qdrant szemantikus      │  │
│  │ • get_recalls()  │  │ • DTC embedding kereses  │  │
│  │ • get_complaints()│  │ • Tunet embedding kereses│  │
│  │ → recall lista   │  │ → hasonlo esetek        │  │
│  └──────────────────┘  └─────────────────────────┘  │
│                                                      │
│  ┌──────────────────┐  ┌─────────────────────────┐  │
│  │ Neo4j graf       │  │ PostgreSQL full-text     │  │
│  │ • DTC→Component  │  │ • DTC reszletek         │  │
│  │ • Component→Repair│  │ • KnownIssue tabla      │  │
│  │ • Vehicle→Engine  │  │ • Hasonlo panaszok      │  │
│  │ → javitasi utvonal│  │ → strukturalt kontextus │  │
│  └──────────────────┘  └─────────────────────────┘  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  3. RAG KONTEXTUS OSSZEALLITAS                      │
│                                                      │
│  Reciprocal Rank Fusion (RRF) a kulonbozo           │
│  forrasok eredmenyein:                               │
│                                                      │
│  • dtc_context:     DTC leirasok + PostgreSQL match  │
│  • symptom_context: Szemantikus talaltok (Qdrant)   │
│  • repair_context:  Graf utvonalak (Neo4j)          │
│  • recall_context:  NHTSA recall + complaint adatok │
│                                                      │
│  Sulyzoas: Qdrant 0.6 + PostgreSQL 0.3 + Neo4j 0.1 │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  4. LLM GENERALAS (Claude / GPT-4)                  │
│                                                      │
│  Magyar nyelvu diagnosztikai prompt:                 │
│  • Direktiv utasitasok (tapasztalt szerelo szerepben)│
│  • Jarmu es DTC kontextus                           │
│  • RAG adatok (recall, panasz, javitas)             │
│  • Meresi ertekek es szerszamok szekciok            │
│  • Gyokerok elemzes (root cause analysis)           │
│                                                      │
│  Kimenet strukturalt JSON:                           │
│  • Valoszinu okok (confidence score)                │
│  • Erintett alkatreszek                              │
│  • Javitasi lepesek                                  │
│  • Sulyossagi szint                                  │
│  • Biztonsagi figyelmeztetesek                       │
│                                                      │
│  Fallback: Rule-based diagnozis ha LLM nem elerheto │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  5. ALKATRESZ ARAZAS (PartsPriceService)            │
│                                                      │
│  • DTC → alkatresz mapping (DTC_PARTS_MAP dict)    │
│  • Alkatresz kategorizalas (szenzor, szuro, fek...) │
│  • Min/max ar tartomany (HUF)                       │
│  • Munkadij becsles (nehezsegtol fuggoen)           │
│  • Osszkoltsg osszesites                             │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  6. VALASZ OSSZEALLITAS                              │
│                                                      │
│  DiagnosisResponse:                                  │
│  • probable_causes[] (ok, confidence, DTC)          │
│  • recommended_repairs[] (lepes, koltseg, ido)      │
│  • tools_needed[] (szerszam, leiras, ar)            │
│  • parts_with_prices[] (alkatresz, min/max ar)      │
│  • total_cost_estimate (alkatresz + munka)          │
│  • similar_complaints[] (NHTSA panaszok)            │
│  • related_recalls[] (visszahivasok)                │
│  • urgency_level (alacsony/kozepes/magas/kritikus)  │
│  • safety_warnings[]                                 │
└─────────────────────────────────────────────────────┘
```

### 3.2 HuBERT Embedding Szolgaltatas

A SZTAKI-HLT/hubert-base-cc modell magyar nyelvre optimalizalt BERT valtozat:

- **Dimenzio**: 768
- **Elotrenirozas**: Magyar kozossegi media szovegeken
- **Alkalmazas**: Panasz/tunet → hasonlo DTC/recall keresés
- **GPU tamogatas**: CUDA (Linux/Windows), MPS (Mac), CPU fallback
- **OOM vedelem**: Automatikus batch meret csokkentes

```python
class EmbeddingService:
    """Magyar nyelvu szoveg embedding HuBERT-tel."""

    def __init__(self):
        self.model = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        self.tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        # GPU/MPS/CPU autodetect
        self.device = self._detect_device()

    async def embed_text(self, text: str) -> List[float]:
        """768-dim embedding generalas magyar szoveghez."""
        # HuSpaCy elofeldolgozas (lemma, stopword)
        preprocessed = self.preprocess_hungarian(text)
        # Tokenizalas + inference
        inputs = self.tokenizer(preprocessed, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling → 768-dim vektor
        return outputs.last_hidden_state.mean(dim=1).tolist()[0]
```

### 3.3 Szemantikus Kereses (Qdrant)

```python
async def semantic_search(query: str, top_k: int = 10) -> List[SearchResult]:
    """Magyar nyelvu szemantikus kereses a Qdrant vektoradatbazisban."""
    # 1. Query embedding generalas (HuBERT)
    query_vector = await embedding_service.embed_text(query)

    # 2. Cosine similarity kereses
    results = qdrant_client.search(
        collection_name="autocognitix",
        query_vector=query_vector,
        limit=top_k,
        query_filter=Filter(must=[
            FieldCondition(key="language", match=MatchValue(value="hu"))
        ])
    )

    # 3. Relevancia szures (score > 0.65)
    return [r for r in results if r.score > 0.65]
```

### 3.4 Graceful Degradation

A rendszer robusztusan kezeli ha egyes adatforrasok nem elerhetoek:

| Forras | Kieses eseten | Hatas |
|--------|--------------|-------|
| Qdrant | PostgreSQL full-text search fallback | Kulcsszo-alapu kereses szoveg helyett |
| Neo4j | Ures graf adat, LLM csak DTC + NHTSA-bol dolgozik | Kevesbe reszletes javitasi javaslat |
| NHTSA API | Ures recall/complaint lista | Nincs visszahivas/panasz adat |
| LLM | Rule-based diagnozis (szabaly-alapu) | Strukturalt de kevesbe reszletes valasz |
| Redis | Kozvetlen DB lekerdezes (lassabb) | Magasabb latency |
| PostgreSQL | Hiba propagalodik | Kritikus — fo adatforras |

---

## 4. API Specifikacio

### 4.1 Vegpontok / Endpoints

| Vegpont | Metodus | Leiras | Auth |
|---------|---------|--------|------|
| `/api/v1/auth/register` | POST | Felhasznalo regisztracio | Nem |
| `/api/v1/auth/login` | POST | Bejelentkezes (JWT) | Nem |
| `/api/v1/auth/refresh` | POST | Token frissites | Igen |
| `/api/v1/auth/logout` | POST | Kijelentkezes (token blacklist) | Igen |
| `/api/v1/diagnosis/analyze` | POST | Fo diagnosztika (RAG + LLM) | Igen |
| `/api/v1/diagnosis/history` | GET | Elozmenyek listazasa | Igen |
| `/api/v1/dtc/search?q=` | GET | DTC kod kereses | Nem |
| `/api/v1/dtc/{code}` | GET | DTC reszletek | Nem |
| `/api/v1/vehicles/makes` | GET | Gyartok listaja | Nem |
| `/api/v1/vehicles/models?make_id=` | GET | Modellek listaja | Nem |
| `/api/v1/vehicles/decode-vin` | POST | VIN dekodolas (NHTSA) | Nem |
| `/health` | GET | Health check | Nem |
| `/metrics` | GET | Prometheus metrikak | Nem |

### 4.2 Diagnosztika Request/Response

**Request:**
```json
{
    "vehicle_make": "Volkswagen",
    "vehicle_model": "Golf",
    "vehicle_year": 2018,
    "vin": "WVWZZZ3CZWE123456",
    "dtc_codes": ["P0171", "P0101"],
    "symptoms": "A motor nehezen indul hidegben, egyenetlenul jar alapjaraton."
}
```

**Response:**
```json
{
    "id": "uuid",
    "probable_causes": [
        {
            "title": "MAF szenzor szennyezodes",
            "description": "A levegoetoemeg-mero szenzor szennyezett...",
            "confidence": 0.85,
            "related_dtc_codes": ["P0171", "P0101"]
        }
    ],
    "recommended_repairs": [
        {
            "title": "MAF szenzor tisztitas",
            "steps": ["Valassza le a csatlakozot", "Permetezze be MAF tisztitoval"],
            "difficulty": "beginner",
            "estimated_time_minutes": 30
        }
    ],
    "tools_needed": [
        {"name": "MAF szenzor tisztito spray", "estimated_price_huf": 3500}
    ],
    "parts_with_prices": [
        {"name": "MAF szenzor (uj)", "price_min_huf": 15000, "price_max_huf": 45000}
    ],
    "total_cost_estimate": {
        "parts_min": 0, "parts_max": 5000,
        "labor_min": 0, "labor_max": 5000,
        "total_min": 0, "total_max": 10000
    },
    "similar_complaints": [...],
    "related_recalls": [...],
    "urgency_level": "medium",
    "safety_warnings": [],
    "confidence_score": 0.82
}
```

---

## 5. Biztonsag / Security

### 5.1 Authentikacio es Authorizacio
- **JWT token**: Access (30 perc) + Refresh (7 nap)
- **Password**: bcrypt hash (auto rounds)
- **Token blacklist**: Redis-alapu logout
- **Account lockout**: 5 sikertelen proba → 15 perc zaroltas
- **Email verifikacio**: Token-alapu megerosites

### 5.2 Halozati Biztonsag
- HTTPS only (Railway TLS)
- CORS: konfiguralt origin lista
- CSRF middleware (double-submit cookie)
- Rate limiting: 100 req/perc/IP
- Security headers: X-Frame-Options, HSTS, XSS protection
- IDOR vedelem: ownership check diagnosztikai munkameneteken

### 5.3 Adatvedelmi Szabalyok
- GDPR kompatibilis adatkezeles
- Soft delete minden felhasznaloi adatra
- Nincs hardcoded credential a kodban
- Env variable-alapu konfiguracio

---

## 6. Teljesitmeny / Performance

| Metrika | Celertk | Megjegyzes |
|---------|---------|------------|
| DTC lookup | <50ms | PostgreSQL index |
| Szemantikus kereses | <200ms | Qdrant cosine similarity |
| Teljes diagnozis | <3s | RAG + LLM + Parts pricing |
| Cache hit rate | 90%+ | Redis tiered TTL |
| API rendelkezesre allas | 99.9% | Railway auto-restart |

**Optimalizaciok:**
- Connection pooling (10 base + 20 overflow)
- Redis embedding cache (elkeruli ujraszamitast)
- FP16 GPU inference (fele memoria, gyorsabb)
- Dinamikus batch meret (16-128 GPU alapjan)
- GZip kompresszio (60-80% csokkenes)
- Frontend code splitting (lazy loading)

---

## 7. Deployment & DevOps

### 7.1 Production Kornyezet

```
Railway Project
├── backend (FastAPI) ─────────────┐
│   └── Dockerfile                 │
├── frontend (React) ──────────────┤
│   └── Nixpacks build             │
├── PostgreSQL 16 (Railway)        ├── Railway Private Network
├── Redis 7 (Railway)              │
└── External Services              │
    ├── Neo4j Aura (cloud.neo4j.com)
    ├── Qdrant Cloud (cloud.qdrant.io)
    └── LLM API (Anthropic / OpenAI)
```

### 7.2 CI/CD Pipeline (GitHub Actions)

| Workflow | Trigger | Lepesek |
|----------|---------|---------|
| ci.yml | Push/PR | Ruff lint → Ruff format → MyPy → Pytest → Frontend build |
| security.yml | Daily + Push | CodeQL → Bandit → npm audit |
| Deploy | CI Success (main) | Railway auto-deploy |

### 7.3 Monitorozas
- **Logging**: Strukturalt JSON (correlation ID)
- **Metrics**: Prometheus endpoint (/metrics)
- **Health**: /health endpoint (DB connectivity check)
- **Error tracking**: Strukturalt exception hierarchy (50+ error code)

---

## 8. Adat Pipeline (Sprint 8-9)

### 8.1 Adatforrasok

| Forras | Tipus | Mennyiseg | Frissites |
|--------|-------|-----------|-----------|
| NHTSA Complaints | API + Flat files | 1.66M nyers → 751K szurt | Negyedevente |
| NHTSA Recalls | API | Elo lekerdezes | Valos ideju |
| EPA FuelEconomy | Flat files | 31 603 jarmu | Evente |
| DTC Adatbazis | CSV import | 6 814 kod | Manualis |

### 8.2 Import Scriptek

| Script | Funkcio | Kimenet |
|--------|---------|---------|
| sample_complaints.py | 1.66M → 751K intelligens mintavetelezes | sampled_200k.json, sampled_50k_embedding.json |
| sync_postgres_sprint9.py | Batch UPSERT PostgreSQL-be | 790K+ rekord |
| sync_neo4j_sprint9.py | MERGE + relationship building | 65K node |
| sync_qdrant_sprint9.py | HuBERT embedding + Qdrant upload | 54K+ vektor |

**Robusztussag:**
- Atomic checkpoint writes (tmp + rename pattern)
- Resume tamogatas (batch szintu checkpoint)
- Retry logika exponential backoff-fal
- OOM vedelem (automatikus batch meret csokkentes)

---

*Dokumentum verzio: 2.0*
*Utolso frissites: 2026-02-10 (Sprint 9 utan)*
