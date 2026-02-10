# AutoCognitix - Palyazati Briefing a Cowork Szamara

---

## GYORS ATTEKINTES

**Projekt neve:** AutoCognitix
**Tipus:** AI-alapu gepjarmu-diagnosztikai platform
**Celkituzes:** Hardver nelkuli jarmudiagnosztika magyar nyelven
**Fejlesztesi fazis:** MVP kesz, V1 produktum epul (Sprint 9 kesz, Q2 2026 target)
**Igenyelt tamogatas:** 30 000 000 HUF

---

## MI EZ A PROJEKT?

**Egyszeru magyarazat:**
Kepzeld el, hogy felvilagit a muszerfalon a "check engine" lampa. A hagyomanyos megoldas: elmesz egy szervizbe, ahol 5-15 ezer forintert leolvassak a hibakodot, aztan meg elmagyarazzak (remelhhetoleg magyarul), hogy mi a problema.

Az AutoCognitix ezt kiikusobolja:
1. A felhasznalo begepeli a DTC kodot (pl. P0171) VAGY leirja a tuneteket magyarul
2. Az AI elemzi a hibat a tudagraffal es a panaszadatbazissal
3. Visszaadja: mi a problema, mi okozhatja, hogyan javithato lepesrol-lepesre, mennyibe kerul az alkatresz

**Fo ertek:** Elso magyar nyelvu, AI-tamogatott autodiagnosztika, hardver nelkul.

---

## MIERT INNOVATIV?

### 1. Hardver Nelkuli Diagnosztika
- Nem kell OBD-II olvaso (50-500 EUR megtakaritas)
- Manualis kod vagy tunet bevitel elegseges
- Barki hasznalhatja, nem kell szaktudas

### 2. Magyar Nyelvu AI
- Elso gepjarmu-diagnosztika magyar NLP-vel
- SZTAKI huBERT modell (768-dim embedding)
- Megert magyar gepjarmu-terminologiat ("berreges", "kopogatas", "egyenetlen alapjarat")

### 3. Multi-Adatbazis Architektura
- PostgreSQL: felhasznalo adatok, munkamenetek, 790 000+ rekord
- Neo4j: diagnosztikai tudagraf (65 000+ node)
- Qdrant: szemantikus kereses (54 652 embedding)
- Redis: cache a gyors valaszokhoz

### 4. NHTSA Integracios Adatbazis
- 751 422 NHTSA panasz indexelve a PostgreSQL-ben
- 50 000 panasz a Neo4j grafban DTC kapcsolatokkal
- 31 603 EPA jarmutechnikai adat (motor, fogyasztas, emisszio)
- Valos visszahivas (recall) adatok eloben

---

## PIAC ES UZLETI MODELL

### Celcsoport

| Szegmens | Meret (HU) | Fajdalompont |
|----------|------------|--------------|
| Fuggetlen szervizek | 8 000+ | Draga diagnosztikai eszkozok |
| Autotulatdonosok | 4M jarmu | Nem ertik a hibakodokat |
| Gepjarmu oktatas | 200+ iskola | Nincs magyar kepzesi eszkoz |
| Flottakezeles | 500+ ceg | Gyors hibaelharitas |

### Uzleti Modell

| Csomag | Ar | Celcsoport |
|--------|-----|------------|
| B2B Szerviz | 19 990 - 29 990 Ft/ho | Autoszervizek - korlatlan elemzes |
| B2C Interaktiv | 9 900 Ft/elemzes | Otthoni szerelok - AI elemzes + kerdezhetsz az AI-tol |
| B2C Utmutato | 4 900 Ft/elemzes | DIY szerelok - AI vegigvezet a javitason |
| Enterprise | Egyedi | Flottak, szervizlancok |
| API | Pay-per-use | Fejlesztok, integraciok |

### Piaci Meret

- **TAM:** 2.5 Mrd EUR (EU digitalis aftermarket)
- **SAM:** 150M EUR (fuggetlen szerviz + DIY)
- **SOM:** 15M EUR (HU + KKE regios)

### Strategia

Eloszor B2B (autoszervizek) = stabil MRR + termekvalidalas, utana B2C (retail PWA app) = skalazas

---

## JELENLEGI ALLAPOT (2026-02-10, Sprint 9 utan)

### Kesz Funkciok - Production-ben Elve

| Funkcio | Allapot | Reszletek |
|---------|---------|-----------|
| Backend API (FastAPI) | ✅ Online | 7 endpoint, JWT auth, RAG diagnozis |
| Frontend (React + TS) | ✅ Online | 11 oldal, diagnosis wizard, auth flow |
| PostgreSQL adatok | ✅ Feltoltve | 6 814 DTC + 89 marka + 2 192 model + 751K panasz + 31K EPA |
| Neo4j tudagraf | ✅ Feltoltve | 65 207 node (DTC, Vehicle, Engine, Complaint) |
| Qdrant vektor kereso | ✅ Mukodik | 54 652 embedding (768-dim HuBERT) |
| NHTSA API | ✅ Elo | Recall + Complaint + VIN lekerdezes |
| Railway deployment | ✅ Online | Backend + Frontend + PostgreSQL + Redis |
| CI/CD pipeline | ✅ Automatikus | Lint, TypeCheck, Test, Build, Deploy |
| Biztonsagi audit | ✅ Kesz | JWT, CSRF, IDOR fix, rate limiting |
| Tesztekkel | ✅ 200+ teszt | Unit, integration, E2E |
| LLM diagnozis | ✅ Mukodik | Claude/GPT prompt, RAG context, Parts&Prices |

### Production URL-ek

| Szolgaltatas | URL |
|-------------|-----|
| Backend API | https://autocognitix-production.up.railway.app |
| Frontend | https://remarkable-beauty-production-8000.up.railway.app |
| API Docs | https://autocognitix-production.up.railway.app/docs |
| Health Check | https://autocognitix-production.up.railway.app/health |

### Adatbazis Telitettsg - Hol Tartunk

| Metrika | Aktualis | V1 Cel | Teljesites |
|---------|----------|--------|------------|
| DTC kodok | 6 814 | 11 000 | **62%** |
| Magyar DTC forditasok | ~923 | 9 000+ (80%) | **10%** |
| Neo4j node-ok | 65 207 | 160 000 | **41%** |
| Qdrant vektorok | 54 652 | 170 000 | **32%** |
| NHTSA panaszok (PG) | 751 422 | 750 000+ | **100%** |
| Jarmu markak | 89 | 90+ | **99%** |
| Jarmu modellek | 2 192 | 2 500+ | **88%** |
| EPA jarmuvek | 31 603 | 30 000+ | **100%** |

### Mi Hianyzik Meg a V1-hez

| Funkcio | Prioritas | Becsult munka | Fuggoseg |
|---------|-----------|---------------|----------|
| Magyar DTC forditasok (80%) | KRITIKUS | 2-3 het | LLM batch forditas |
| Fizetesi rendszer (Stripe) | KRITIKUS | 2 het | Stripe account |
| PWA mobil tamogatas | FONTOS | 1 het | Service Worker, manifest |
| Tunetes diagnozis bovites | FONTOS | 2 het | Tunetszotár, NLP |
| Qdrant reindex (110K uj) | FONTOS | 4-6 ora | Lokalis HuBERT futtatas |
| Szerelesi utmutatok | KOZEPES | 2-3 het | iFixit + sajat tartalom |
| Neo4j grafbovites (160K) | KOZEPES | 1 het | Recall + TSB import |
| B2B API tier | ALACSONY | 1 het | API key management |

---

## KOLTSEGVETES OSSZEFOGLALO

### Igenyelt Tamogatas: 30 000 000 HUF

| Kategoria | Osszeg | % |
|-----------|--------|---|
| Fejlesztes | 15 000 000 | 50% |
| Infrastruktura (12 ho) | 6 000 000 | 20% |
| Marketing | 5 000 000 | 17% |
| Jogi & Admin | 2 000 000 | 7% |
| Tartalek | 2 000 000 | 7% |

### Felhasznalasi Utemterv

| Negyedev | Osszeg | Tevekenyseg |
|----------|--------|-------------|
| Q2 2026 | 8M HUF | V1 befejezese, beta (forditas, fizetes, PWA) |
| Q3 2026 | 10M HUF | Launch, marketing, B2B pilot |
| Q4 2026 | 7M HUF | Bovites, B2B skalazas |
| Q1 2027 | 5M HUF | KKE terjeszkedés |

### Jelenlegi Infrastruktura Koltsegek (havi)

| Szolgaltatas | Jelenlegi | V1 Production |
|-------------|-----------|---------------|
| Railway (Backend+FE+PG+Redis) | Ingyenes/Hobby | ~$20/ho |
| Neo4j Aura | Free tier | ~$65/ho (Pro) |
| Qdrant Cloud | Free tier | ~$25/ho |
| GitHub | Free | Free |
| Domain + SSL | - | ~$15/ev |
| **Osszesen** | **~$0** | **~$110/ho** |

---

## TECHNOLOGIAI VEREM

```
Frontend:     React 18 + TypeScript + TailwindCSS + Vite
Backend:      FastAPI + Python 3.9 + SQLAlchemy 2.0 async
Adatbazisok:  PostgreSQL 16 + Neo4j 5.x + Qdrant + Redis 7
AI/NLP:       huBERT (SZTAKI) + LangChain + Claude API
Embedding:    768-dim HuBERT vektorok, Qdrant cosine similarity
Deployment:   Railway + Neo4j Aura + Qdrant Cloud
CI/CD:        GitHub Actions (lint, type-check, test, security scan, deploy)
Monitoring:   Prometheus metrics, structured JSON logging
```

---

## VERSENYELONY

| Jellemzo | AutoCognitix | Hagyomanyos | Generikus app |
|----------|--------------|-------------|---------------|
| Hardver kell | NEM | IGEN | IGEN |
| Magyar nyelv | NATIV | NEM | NEM |
| AI elemzes | IGEN (RAG + LLM) | NEM | Korlatozott |
| Tudagraf | 65K+ node | NEM | NEM |
| NHTSA panaszok | 751K indexelve | NEM | Ritka |
| EPA motor adatok | 31K jarmu | NEM | NEM |
| Javitasi javaslat | Reszletes + arazas | Alap | Korlatozott |
| Szerelesi utmutato | Lepesrol-lepesre | Nincs | Alap |
| Fizetesi modell | Per-elemzes | Elofizetes | Egyszeri |

---

## CSAPAT

**Jelenlegi:**
- Lead Developer (1 FTE) - Backend, AI/ML, DevOps, Frontend

**Szukseges bovites (V1-hez):**
- Frontend Developer (0.5-1 FTE) - UI polish, PWA
- Product Designer (0.5 FTE) - UX, mobil optimalizalas
- Fordito/Lektor (projekt alapu) - DTC magyar forditasok

---

## UTEMETRV

| Fazis | Idoszak | Statusz | Fo deliverable |
|-------|---------|---------|----------------|
| Fazis 1: Alapok | Q4 2025 - Q1 2026 | ✅ Befejezve | Architektura, API, DB, NLP |
| Sprint 1-6: Core | Q1 2026 | ✅ Befejezve | Auth, DTC, Vehicle, Diagnosis |
| Sprint 7-7.5: AI | 2026 Feb eleje | ✅ Befejezve | LLM prompt, RAG, Parts&Prices |
| Sprint 8-9: Data | 2026 Feb 8-10 | ✅ Befejezve | 751K panasz, 31K EPA, Neo4j sync |
| Sprint 10: Forditas | 2026 Feb-Mar | ⏳ Kovetkezo | 80% magyar DTC fedettsg |
| Sprint 11: Fizetes | 2026 Mar | 📋 Tervezett | Stripe integracios |
| Sprint 12: PWA | 2026 Mar-Apr | 📋 Tervezett | Mobil app |
| Fazis 2: V1 Launch | Q2 2026 | 📋 Tervezett | Beta launch, B2B pilot |
| Fazis 3: AI Evolucio | Q3-Q4 2026 | 📋 Tervezett | Foto, hang, prediktiv |
| Fazis 4: Piacbovites | 2027 | 📋 Tervezett | DE, PL, RO nyelvek |

---

## KOCKAZATOK ES KEZELESUK

| Kockázat | Valószínűség | Hatás | Kezelés |
|----------|-------------|-------|---------|
| LLM API koltseg | Kozepes | Magas | Lokalis modell fallback (HuBERT mar mukodik) |
| Neo4j Aura free tier limit | Magas | Kozepes | 400K rel limit elerve, Pro tier szukseges |
| Lassu adopcio | Kozepes | Magas | Freemium modell, B2B elso |
| Magyar forditas minosege | Alacsony | Magas | Lektor + kozossegi validalas |
| Versenytars | Alacsony | Kozepes | Gyors feature fejlesztes, magyar NLP elony |

---

## AMIT MAR BEMUTATHATUNK (DEMO)

1. **Regisztracio + Bejelentkezes** - Teljes auth flow JWT tokennel
2. **DTC kereses** - 6 814 hibakod kozott kereses
3. **Diagnozis varazslo** - DTC kod VAGY tunet bevitel → AI elemzes
4. **Eredmeny oldal** - Diagnosztikai jelentes alkatreszarakkal
5. **Jarmu adatbazis** - 89 marka, 2 192 modell
6. **NHTSA recall check** - VIN alapu visszahivas ellenorzes
7. **API dokumentacio** - Swagger UI interaktiv teszteleshez

---

## KAPCSOLODO DOKUMENTUMOK

A `docs/` mappaban:

1. **PROJECT_OVERVIEW.md** - Teljes projekt attekintes (HU/EN)
2. **GRANT_APPLICATION_SUMMARY.md** - Palyazati osszefoglalo
3. **TECHNICAL_DESCRIPTION.md** - Reszletes technikai leiras
4. **BUDGET_AND_RESOURCES.md** - Koltsegvetes es eroforras terv
5. **FILE_STRUCTURE.md** - Projekt fajlstruktura

---

## ELERHETO DEMOK

- **Live Backend:** https://autocognitix-production.up.railway.app
- **Live Frontend:** https://remarkable-beauty-production-8000.up.railway.app
- **API Docs:** https://autocognitix-production.up.railway.app/docs
- **Health Check:** https://autocognitix-production.up.railway.app/health
- **GitHub:** github.com/anorbert-cmyk/AutoCognitix (privat)

---

*Dokumentum: Cowork Briefing*
*Verzio: 2.0*
*Utolso frissites: 2026-02-10 (Sprint 9 utan)*
