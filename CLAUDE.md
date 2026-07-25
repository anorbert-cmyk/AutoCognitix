# AutoCognitix - Claude Code Projekt Kontextus

## Projekt Áttekintés

**Cél:** AI-alapú gépjármű-diagnosztikai platform magyar nyelvtámogatással, hardver nélküli manuális DTC kód és tünet bevitellel.

**Státusz:** Sprint S4 befejezve — Szemantikus keresés helyreállítva (ONNX embedding + Qdrant collection-drift + DTC szabály egységesítés)

**Deployment:** Railway (PostgreSQL + Redis) + Neo4j Aura + Qdrant Cloud

## Aktuális Adatbázis Állapot

> ⚠️ **A repó forrásai ELLENTMONDANAK egymásnak.** Ezek dokumentált értékek, nem élő mérések — egyik szám sem lett Qdrant/Neo4j lekérdezéssel megerősítve ebben a sprintben. Amíg valaki nem futtat egy `GET /collections`-t a Qdrant Cloudon és egy node-count Cyphert a Neo4j Aurán, mindkét oszlop bizonytalan. **Ne írd át egyiket sem "rendrakásból" — előbb mérd meg.**

| Adatbázis | Tartalom | Méret | Forrás |
|-----------|----------|-------|--------|
| **Neo4j Aura** | Vehicles, DTC, Complaints, Recalls | **26,816 node** *vagy* **65,207 node** | 26,816: ez a sor + `docs/DATABASE_MAP.md`, `docs/project-description-hu.md`, `docs/ARCHITECTURE.md` (mind ezt a táblát idézi vissza). 65,207: `docs/TECHNICAL_DESCRIPTION.md` §2.2 és `docs/COWORK_BRIEF.md` — **típusonkénti bontással** (DTCCode 6 814 + Vehicle 7 289 + Engine 1 104 + Complaint 50 000). |
| **Qdrant Cloud** | HuBERT embeddings (768-dim), **egyetlen `autocognitix` collection** | **35,000+ vector** *vagy* **54,652 vector** | 35,000+: ez a sor + `docs/project-description-hu.md`. 54,652: `docs/COWORK_BRIEF.md` és `backend/tests/unit/test_qdrant_client.py`; a kód `~54k`-ként hivatkozik rá (`backend/app/core/config.py` `HUBERT_REVISION` komment, `backend/app/services/embedding_service.py` modul-docstring). |
| **PostgreSQL** | Users, Sessions, History, NHTSA panaszok | Kész (`docs/COWORK_BRIEF.md`: 751 422 panasz) | — |
| **Redis** | Cache, Session | Kész | — |

**Amit tudni érdemes a bontásról:** a 65,207-es és az 54,652-es szám **típusonkénti bontással együtt** szerepel, a 26,816 / 35,000+ nem. Ez nem bizonyíték, csak jelzés arról, melyik forrás áll közelebb egy valódi lekérdezéshez. A `docs/EMBEDDING_ARCHITECTURE_DECISION.md` §8.6.2 nyitott kérdésként tartja nyilván.

**Egy külön szám, ami MÉRVE lett:** a Neo4j gráf ~**107** `MENTIONS_DTC` élt tartalmazott (complaint → DTC), ami miatt a `common-issues` DTC-ága gyakorlatilag üres volt. Forrás: `scripts/sync_neo4j_sprint9.py`. Ez volt a bizonyíték arra, hogy a feature egy hamis feltevésre épült (hogy a fogyasztói panaszok idéznek DTC kódokat).

### HuBERT Embedding - Miért használjuk?
A **SZTAKI-HLT/hubert-base-cc** modell magyar nyelvre optimalizált BERT változat:
- **Szemantikus keresés**: Panasz/tünet → hasonló DTC/recall keresés
- **768-dim vektorok**: Qdrant-ban tárolva, cosine similarity alapú keresés
- **Lokális futás**: Nincs API limit (Groq kimerült)
- **RAG alapja**: A diagnosztikai AI innen keres releváns információt

**Runtime (2026-07-25 óta):** a production **ONNX Runtime fp32**-vel futtatja a huBERT-et — torch és transformers **nélkül**. A modell commit SHA-ra van pinelve (`HUBERT_REVISION`), a gráfot a `Dockerfile.prod` `onnx-export` stage-e exportálja, és minden build bebizonyítja, hogy a gráf reprodukálja a `backend/tests/fixtures/hubert_reference_vectors.json` befagyasztott vektorait. Részletek: **`docs/EMBEDDING_ARCHITECTURE_DECISION.md`**.

**Qdrant collection-modell:** minden vektor a **`autocognitix`** collectionben van, `type` payload-diszkriminátorral (`dtc` / `complaint` / `recall`). Az öt `*_hu` collection létezik, de **üres**. `type = "symptom"` nem létezik — a RAG symptom-lába `type="complaint"`-re képez.

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

### 2. Subagent Strategy (Default Mode)
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
- **Subagent = alapértelmezett.** Agent Teams-re CSAK az alábbi triggerek esetén válts (ld. pont 7.)

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

### 7. Agent Teams - Intelligens Eszkaláció (Experimental)
**Engedélyezés:** `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` a settings.json-ban.

**Alapelv:** Subagent az alapértelmezett. Agent Teams CSAK akkor, ha az alábbi triggerek közül LEGALÁBB KETTŐ teljesül.

#### Automatikus Trigger Felismerés - MIKOR válts Agent Teams-re:

| Trigger | Leírás | Példa |
|---------|--------|-------|
| **MULTI_DB_SYNC** | 2+ adatbázis egyidejű módosítása szükséges | PostgreSQL + Neo4j + Qdrant szinkron, fordítás → reindex |
| **CROSS_LAYER** | Backend + Frontend együtt változik, API contract érintett | Új endpoint + UI oldal + schema módosítás |
| **COMPETING_HYPOTHESES** | Bug okát nem ismerjük, 3+ lehetséges root cause | Lassú query: index? connection pool? N+1? lock? |
| **PARALLEL_REVIEW** | Kód review több független szempontból | Security + Performance + Compatibility egyidejű review |
| **MULTI_MODULE_FEATURE** | Új feature 4+ független fájlcsoportot érint | Auth rendszer: models + API + frontend + tests |
| **DATA_PIPELINE** | Adatfeldolgozás több lépéssel, lépések közti kommunikáció kell | Scrape → Parse → Translate → Validate → Import → Index |

#### Döntési Fa:
```
Feladat érkezik
  └─ Hány trigger teljesül?
       ├─ 0-1 trigger → SUBAGENT (default)
       ├─ 2+ trigger  → AGENT TEAMS
       └─ Bizonytalan? → Subagent-tel kezdj, eszkalálj ha szükséges
```

#### Agent Teams Szereposztás Sablonok:

**Multi-DB Szinkron Team:**
```
Lead: Koordinátor - nem ír kódot, delegate mode
Teammate 1: PostgreSQL specialist (migrációk, modellek)
Teammate 2: Neo4j specialist (Cypher, gráf struktúra)
Teammate 3: Qdrant specialist (embeddings, indexelés)
→ Require plan approval for all teammates
```

**Cross-Layer Feature Team:**
```
Lead: Architektus - API contract definiálás, delegate mode
Teammate 1: Backend (FastAPI endpoints, services, models)
Teammate 2: Frontend (React pages, components, hooks)
Teammate 3: Tests & Integration (pytest, Playwright)
→ Backend teammate-nek kell elsőként befejezni (task dependency)
```

**Debug Team (Competing Hypotheses):**
```
Lead: Szintetizáló - összegyűjti az eredményeket
Teammate 1-N: Hipotézis vizsgálók (egymást cáfolják!)
→ Broadcast: "Challenge each other's findings"
→ A lead CSAK akkor zárja le, ha konszenzus van
```

**Parallel Review Team:**
```
Lead: Review coordinator
Teammate 1: Security (OWASP, injection, auth bypass)
Teammate 2: Performance (N+1 queries, memory leaks, bundle size)
Teammate 3: Compatibility (Python 3.9, browser support, Railway)
→ Minden teammate független report-ot készít
```

#### Agent Teams Szabályok:
- **Delegate mode:** Lead NE implementáljon, csak koordináljon
- **File ownership:** Egy fájlt CSAK egy teammate szerkeszthet - no overlap!
- **Plan approval:** Komplex feladatoknál teammate-ek plan mode-ban indulnak
- **Task granularity:** 5-6 task per teammate az optimális
- **Monitoring:** Rendszeresen ellenőrizd a teammate-ek haladását
- **Cleanup:** Mindig a lead végezze a team cleanup-ot a végén
- **Shutdown order:** Előbb teammate-ek leállítása, utána cleanup

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Pre-Push Review Protocol - KÖTELEZŐ MINDEN PUSH ELŐTT

### Mikor: Minden `git push` ELŐTT, kivétel nélkül (.md-only push is).

### Folyamat:

**Minimum 5 lead agent párhuzamosan**, mindegyik a STAGED + COMMITTED diff-en dolgozik (`git diff origin/<branch>...HEAD`). Rövid, fókuszált scope (max 3 szekció / agent, max 200 sor riport), hogy ne timeout-oljon.

#### Standard 5 lead role:

1. **Security Lead** — OWASP Top 10, auth bypass, injection (SQL/log/cmd), secrets exposure, GDPR boundary, CSRF, IDOR
2. **Logic/Correctness Lead** — off-by-one, null/undefined, edge case, fallback path, idempotency, contract drift
3. **Concurrency/Async Lead** — race condition, thread pool starvation, deadlock, await missing, event loop blocking
4. **Data/Migration Lead** — DB schema drift, missing FK/index, transaction boundary, alembic forward+downgrade, cache key versioning
5. **Operational/Observability Lead** — Sentry capture coverage, structured logging, retry/timeout config, rollback strategy

#### Eredmény:
- Mind az 5 agent ad CRITICAL/HIGH/MEDIUM/LOW besorolást
- CRITICAL/HIGH → push BLOKKOLVA, javítás kötelező, újra audit
- MEDIUM → engedélyezett push + új issue/TODO felvétele
- LOW → dokumentálandó, nem blokkoló

#### Szabály:
- A push BLOKKOLT, amíg legalább 5 lead agent végzett és nincs CRITICAL/HIGH találat
- `git push --no-verify` használata TILOS

---

## Post-Sprint Review Protocol - KÖTELEZŐ MINDEN SPRINT UTÁN

### Mikor: Minden sprint befejezése után, a következő sprint indítása ELŐTT.

### Folyamat:

1. **Parallel Review Team indítása** (4 specialist párhuzamosan):
   - **Security Specialist**: OWASP Top 10, auth bypass, injection, GDPR compliance
   - **Database Specialist**: Cross-DB konzisztencia, tranzakció határok, connection management
   - **Performance Specialist**: Async hibák, race condition, memory leak, N+1 query
   - **Code Quality Specialist**: Type safety, edge case, Python 3.9 kompatibilitás, teszt minőség

2. **Minden specialist teljes fájl-olvasással dolgozik** - nem csak diff-ek, hanem teljes kontextus.

3. **Találat osztályozás**: CRITICAL → HIGH → MEDIUM → LOW

4. **Javítási sorrend**:
   - CRITICAL: Azonnal javítandó, blokkolja a következő sprintet
   - HIGH: Sprint review részeként javítandó
   - MEDIUM: Következő sprint backlog-ba kerül
   - LOW: Dokumentálandó, de nem blokkoló

5. **Verifikáció**: Ruff lint + format check + összes teszt PASS szükséges.

6. **Audit teszt fájl**: `tests/test_sprint_review_audit.py` - a review során talált és javított hibák tesztjei.

7. **Dokumentáció kötelező frissítése** — MINDEN sprint után automatikusan:
   ```
   FRISSÍTENDŐ FÁJLOK (párhuzamosan):
   - CLAUDE.md          → Státusz sor, API Végpontok tábla, Tanulságok szekció, TODO lista
   - README.md          → Key Features, API Endpoints tábla
   - tasks/todo.md      → Sprint státusz táblázat, új sprint blokk hozzáadása
   - tasks/lessons.md   → Új tanulságok és javítások a sprint hibáiból
   ```
   **Ez NEM opcionális.** Minden sprint végén automatikusan végrehajtandó, kérés nélkül.

### Trigger: PARALLEL_REVIEW + bármely másik trigger → Agent Teams mód.

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing new patterns

## Munkafolyamat Preferenciák

- **Párhuzamos ágensek:** Több Task agent egyidejű futtatása
- **Agent Teams:** Engedélyezve - automatikus eszkaláció triggerek alapján (ld. Workflow Orchestration #7)
- **Token költség:** NEM akadály - ha Agent Teams jobb eredményt ad, használd
- **Engedélyek:** Minden keresés/futtatás automatikusan engedélyezett
- **Todo lista:** Aktívan használva a haladás követésére

## API Végpontok

| Végpont | Státusz | Leírás |
|---------|---------|--------|
| `POST /api/v1/diagnosis/analyze` | ✅ Kész | Fő diagnosztika (LLM + RAG + PartsPriceService). A RAG a unified `autocognitix` collectiont kérdezi `type=dtc` és `type=complaint` lábbal |
| `GET /api/v1/dtc/search` | ✅ Kész | DTC keresés. A találatok ugyanazon a SAE J2012 validátoron mennek át, mint a részletek-endpoint — nem lehet olyat találni, amit utána nem lehet megnyitni |
| `GET /api/v1/vehicles/{make}/{model}/common-issues` | ✅ Kész | Jármű gyakori problémái. **Két független rangsor:** `components` (NHTSA panasz-komponens gyakoriság PostgreSQL-ből, `share` + crash/fire/injury/death számokkal) és `issues` (DTC-k a Neo4j gráfból — ritkán van benne adat). Plusz `total_complaints` (a `share` nevezője) és `sources` (forrásonkénti betöltési státusz, hogy egy adattár-kiesés megkülönböztethető legyen a valódi üres eredménytől). `limit` query paraméter. Kiesésnél **200 + igazmondó üres lista**, nem 500 |
| `POST /api/v1/vehicles/decode-vin` | ✅ Kész | VIN dekódolás |
| `POST /api/v1/auth/login` | ✅ Kész | Bejelentkezés (JWT) |
| `GET /demo` | ✅ Kész | Demo bemutató oldal (P0300 szimuláció, valós árak) |
| `GET /api/v1/garage/vehicles` | ✅ Kész | Felhasználó járműveinek listázása (valós `health_score` + `upcoming_reminders_count`, list↔health paritás) |
| `POST /api/v1/garage/vehicles` | ✅ Kész | Jármű hozzáadása garázshoz |
| `GET /api/v1/garage/vehicles/{id}/recalls` | ✅ Kész | Jármű NHTSA visszahívásai |
| `GET /api/v1/garage/vehicles/{id}/health` | ✅ Kész | Jármű egészségi pontszám |
| `GET /api/v1/garage/reminders` | ✅ Kész | Karbantartási emlékeztetők |

## Frontend Demo Oldal

A `/demo` útvonalon elérhető bemutató oldal teljes diagnosztikai jelentést mutat:
- **Szimulált hiba:** P0300 + P0301 + P0304 (több hengeres égéskimaradás)
- **Jármű:** VW Golf VII 1.4 TSI (2018), 98.420 km
- **Alkatrész árak:** Bárdi Autó, Uni Autó, AUTODOC – valós 2026 márciusi árak
- **6 alkatrész kártyás megjelenítéssel:** gyújtógyertya, gyújtótekercs, levegőszűrő, üzemanyagszűrő, injektor, lambda szonda
- **Demó fájlok:**
  - `frontend/src/data/demoData.ts` – demó adatok és árak
  - `frontend/src/pages/DemoResultPage.tsx` – demó oldal
  - `frontend/src/components/features/diagnosis/PartStoreCard.tsx` – bolt-specifikus alkatrész kártya

## Adatforrások

### Ingyenes (implementálandó)
- NHTSA API (VIN, recalls, complaints)
- OBDb GitHub (738+ jármű repo)
- python-OBD könyvtár

### Fizetős (később)
- CarAPI, CarMD

## Tanulságok és Döntések

### 2026-07-25 - Sprint S4: Szemantikus keresés helyreállítása

- **A csendes fallback hónapokig elrejt egy zászlóshajó funkciót.** A prod image-ből hiányzó torch miatt az `embed_text()` `[0.0]*768`-at adott vissza. A nullvektor **típushelyes** (768 float), tehát minden dimenzió- és típusellenőrzés átengedte — cosine keresésnél viszont minden score 0.0, amit a `score_threshold` üres listává alakít. **Az üres lista pedig egy keresőben legitim válasz.** Három egymásra rakódó csendes fallback (embedding → RAG `except` → `_fallback_diagnosis()`) minden szinten hibát üres eredménnyé alakított. Szabály: **a hiba legyen megkülönböztethető a valóban-üres eredménytől, és a megkülönböztetés érjen el az API kontraktusig, ne csak egy log-sztringig.** Ezért kapott a `common-issues` egy `sources` státusz-objektumot, és ezért dob ma `EmbeddingUnavailableError`-t a hiányzó backend.
- **MINDEN mai javítás először FÉL javítás volt.** Nem egy-egy elnézés, hanem visszatérő minta: a retrieval-lábat átirányítottuk, de a **fogyasztóját** nem (a `similar_cases` payload `description` kulcsot olvasott, amit a complaint payloadok nem hordoznak → öt üres bejegyzés a magyar promptban); a DTC szabályt szigorítottuk a részletek-úton, de a **keresésen** nem (a user megtalálta a `PEACE`-t, aztán nem tudta megnyitni); az olvasási utakat átvittük a unified collectionre, de a **GDPR törlést** nem (a törlés semmit nem törölt, miközben sikert jelentett). Szabály: **a bug megtalálása nem a munka — minden hívó végigjárása az.** Grep az egész repóra, ne egy listából dolgozz.
- **Egy őr, ami soha nem fut le, nem őr.** A befagyasztott referenciavektor-teszt `skipif`-fel indult, ami **minden környezetben** skippelt (a `.onnx` sehol nincs meg CI-ben és dev gépen sem) — nulla környezetben futott. Egy másik drift-teszt egy hiányzó függőség miatt volt véglegesen skippelve. Javítás: az egyikből **Docker build-kapu** lett (a `Dockerfile.prod` `onnx-export` stage-e minden buildnél assertálja), a másikból **agreement-teszt**: a `scripts/` importerek ma valóban importálják a kanonikus `app.core.dtc_codes` modult (nincs másolt regex), és egy teszt assertálja, hogy a két független extraction pipeline azonos kimenetet ad.
- **A duplikáció volt a root cause, nem stíluskérdés.** A DTC szabály **tíz** független regexként létezett (két request-séma, metrics middleware, hat `scripts/` importer), és csak egy volt helyes; a collection-név szintén több helyen — ezért lett egyetlen fogalmi javításból három commit. Az ok, amiért a másolatok egyáltalán léteztek: az `app/core/__init__.py` **import-időben építette a `Settings` objektumot**, ami `SECRET_KEY` nélkül nem áll össze, így a `scripts/` alól a stdlib-only `app.core.dtc_codes` sem volt importálható. Javítás: PEP 562 lazy re-export → az `app.core` import mellékhatás-mentes, a scriptek a közös modult használják.
- **A feltevéseket a valódi korpuszon ellenőrizd, ne a kódból következtess.** A `common-issues` arra épült, hogy a fogyasztói panaszok idéznek DTC kódokat. A gráfban **~107** `MENTIONS_DTC` él volt összesen. 26 237 valódi NHTSA narratíván mérve: a szigorú regex 373 találat (15 hamis pozitív), a laza 478 (39 hamis), a helyes SAE J2012 szabály (2. karakter `0-3`) 443 találat **0 megfigyelt hamis pozitívval**. A ranking ezért a **panasz-KOMPONENS gyakoriságra** épült át, ami valódi lefedettségű adat.
- **Egy nullvektor sosem juthat be egy similarity search-be.** `qdrant_client._validate_query_vector()` `ValueError`-t dob üres vagy `norm < 1e-6` vektorra. Szándékosan `ValueError`, nem `QdrantException`: nem a Qdranttal van baj, a hívó adott át érvénytelen vektort.
- **ONNX Runtime = drop-in torch csere, ha a pooling a modellen kívül van.** A pooling és az L2 a gráfon KÍVÜL, numpy-ban fut, ezért az ONNX csak a forward passt cseréli. Mért paritás a pinelt környezetben: `cos ≥ 0,9999991`. Image −720 MB, RSS −208 MB/worker, query ~1,7× gyorsabb. **Feltétel:** a `transformers` NEM kerülhet be a prod image-be (behúzza a torchot, +367 MB RSS → rosszabb, mint a tiszta torch).
- **Verziózott cache-névtér manuális Redis-takarítás helyett.** `EMBEDDING_CACHE_VERSION` + backend név a kulcsban → a mérgezett nullvektor-bejegyzések a deploy pillanatában elérhetetlenné válnak. Nincs `SCAN`+`DEL`, nincs egyórás TTL-ablak, amiben a javítás törötten néz ki.
- **Health státusz ≠ belső próba-státusz.** A `self_test()` `"ok"`-ot ad, a `/health/detailed` viszont `"healthy"`-ra képezi (és `"unavailable"`-t `"degraded"`-re, mert az API a lexikai úton tovább szolgál). Runbookban `status == "healthy"`-ra ellenőrizz.
- **A rank fusion helyben írta felül a score-okat.** A fúzió a `1/(60+rank)` értéket az itemekre írta, felülírva a cosine similarityt, amit utána a confidence számítás olvasott — 36% helyett 0,7% jelent meg a felhasználónak. Javítás: a fúzió **másolatokat ad vissza**, így a korrupció strukturálisan lehetetlen, nem a hívó fegyelmén múlik.

### 2026-07-19 - Sprint S1/S2 + Header Refactor (#22/#23/#24)

- **Pydantic v2 nem koercál UUID→str**: ORM `Uuid` oszlop `str` mezőre validálva `ValidationError`-t / 500-at dob. Megoldás: közös `UUIDStrModel` bázis `@field_validator(..., mode="before", check_fields=False)`-szal (`schemas/garage.py`), ami minden ORM-alapú response modell őse. Öt élőben törött garázs-endpoint javult ettől (BONUS #24).
- **SQLAlchemy `Uuid(as_uuid=True)` bind UUID objektummal**: str bind eltörik a SQLite teszt-harness alatt (`'str' object has no attribute 'hex'`). Megoldás: `_as_uuid()` helper (`vehicle_garage_service.py`), minden bind-paraméter és WHERE-feltétel `UUID`-ra konvertálva. Hibás id → 404 (nem 500).
- **Teszt-app router hiányok elrejtik a törött endpointokat**: a garázs-endpointok 500-ai azért maradtak rejtve, mert a teszt-app nem regisztrálta minden routert. Szabály: `tests/api/conftest.py`-ban MINDEN router regisztrálva legyen, hogy a smoke-tesztek lássák őket.
- **AsyncSession tiltja a konkurrens query-t**: `asyncio.gather()` több `session.execute()`-tal `InterfaceError`-t okoz. Megoldás: egyetlen csoportosított feltételes aggregátum — `func.coalesce(func.sum(case((cond, 1), else_=0)), 0)` + `.group_by(vehicle_id)` — a per-jármű reminder számláláshoz (N+1 és gather helyett).
- **Kitalált UI-adat = bizalmi hiba**: a ResultPage fabrikált mondatot ("főtengely"), ONLINE badge-et és #4829 azonosítót jelenített meg valós adat nélkül. Szabály: soha ne renderelj kitalált értéket — igazmondó üres állapotok (empty state), csak a backend által ténylegesen visszaadott mezők. Külső linkek csak `http`/`https` allowlist után.
- **Publikus oldal ne hívjon védett endpointot**: a HomePage nem-bejelentkezett látogatóként reminder-hívást indított (401). Fix: a hívás csak `isAuthenticated` mögött fut.
- **Grouped aggregate = list↔health paritás**: a járműlista és a health-endpoint UGYANAZT a tiszta scoring függvényt használja ugyanabból az aggregátumból → nincs eltérés a két nézet pontszáma között.
- **Streaming parts enrichment izolációval**: az SSE pipeline parts-dúsítása 5s time-box-szal fut, a hiba izolálva (a stream nem esik el), és a perzisztált eredmény paritásban van a streamelttel.
- **Munkamodell**: Fable orchestrator + Opus 4.8 max-thinking implementer ágensek. Specifikáció → párhuzamos implementáció DISZJUNKT fájl-tulajdonlással → 5-lencsés review (Security / Logic / Concurrency / Data / Ops) → konszolidált javító kör.

### 2026-03-29 - Sprint 9/10 CI Javítások

- **Tuple destructuring**: `get_vehicles()` / `get_reminders()` `Tuple[List, int]`-et ad vissza → endpoint-okban `vehicles, total = await service.get_vehicles(...)` kötelező
- **Pydantic → dict**: Service metódusok `Dict[str, Any]`-t várnak → `data.model_dump(exclude_none=True)` konverzió szükséges átadás előtt
- **MyPy no-any-return**: SQLAlchemy `scalar_one_or_none()` és Pydantic `model_validate()` `Any`-t ad vissza MyPy szerint → `# type: ignore[no-any-return]` komment a sor végén
- **CodeQL log injection**: Minden felhasználói adat logba kerülés előtt `sanitize_log()` kötelező, számok esetén is: `sanitize_log(str(days_ahead))`
- **ruff format vs check**: CI mindkettőt futtatja (`ruff check` + `ruff format --check`). Lokálisan mindig futtasd mindkettőt!
- **ESLint exhaustive-deps**: `const shops = data?.shops || []` minden rendernél új referenciát hoz létre → memo mindig újraszámol. Fix: `data?.shops ?? []` a memo belsejében, `[data?.shops]` a deps-ben
- **Alembic revision/down_revision**: Ezek `# lgtm[py/unused-global-variable]` suppression kommenttel jelölendők (Alembic runtime-ban olvassa őket, CodeQL nem látja). `branch_labels` és `depends_on = None` biztonságosan eltávolítható
- **Vitest AuthProvider trap**: `test-utils.tsx`-hez NEM szabad `AuthProvider`-t adni, mert egyes tesztfájlok az egész `AuthContext` modult mockolják (Provider export nélkül). Per-file mock a helyes megközelítés.
- **react-leaflet Vite marker icon**: Vite asset hashing törli a default marker iconokat → CDN URL-ek + `delete L.Icon.Default.prototype._getIconUrl` fix szükséges

### 2026-03-20 - Post-Sprint Review Audit (Sprint 9/10)
- **Új kötelező workflow**: Minden sprint után Parallel Review Team (4 specialist) audit
- 3 CRITICAL + 6 HIGH hiba javítva (SQL injection, GDPR atomicity, JWT claim protection, stb.)
- `contextvars.ContextVar` használata singleton service-ek thread-safety-jéhez
- `embed_text_async()` az event loop blokkolás elkerülésére
- `_escape_ilike()` helper az SQL wildcard injection ellen
- Rate limiter: fail-closed policy (Redis kiesés → deny, nem allow)
- Post-Sprint Review Protocol hozzáadva a CLAUDE.md-hez

### 2026-03-01 - Demo Error Simulation (Sprint 8)
- Demo oldal: pre-filled P0300 szimuláció VW Golf VII 1.4 TSI-hez valós árakkal
- PartStoreCard: bolt-specifikus kártyás megjelenítés (Bárdi Autó, Uni Autó, AUTODOC)
- Árkutatás: Bárdi és Uni Autó nem indexeli nyilvánosan az árakat → AUTODOC + más webshopok referenciaárai
- DemoPartWithStores típus: StorePricing interface bolt-specifikus árakhoz (storeName, price, inStock, brand)
- Agent Teams 8 fővel: Lead Koordinátor + 7 specialist (Data, CardDesign, PageBuilder, Router, Docs, Types, Lint)

### 2026-02-09 - Enhanced Diagnosis Report (Sprint 7.5)
- PartsPriceService integráció: korábban létezett de nem volt bekötve a pipeline-ba
- LLM prompt újraírás: direktív utasítások, mérési értékek, szerszámok szekciók
- Agent Teams 8 fővel: delegate mode lead + file ownership = sikeres koordináció
- ruff PLR0912 fix: if/elif chain helyett dictionary lookup
- ruff RUF013 fix: `dict = None` helyett `Optional[dict] = None` (Python 3.9)

### 2026-02-08 - Adatbázis feltöltés befejezve
- HuBERT lokális embedding: Groq API limit helyett lokális modell (nincs rate limit)
- Python 3.9 kompatibilitás: `strict=False` zip()-ben nem támogatott
- NHTSA mezőnevek: normalizált format használ snake_case-t (`odi_number`, nem `ODI_ID`)
- Checkpoint rendszer: robusztus resume támogatás batch operációkhoz

### 2024-02-03 - Projekt indítás
- Qdrant választva pgvector helyett (jobb teljesítmény)
- Neo4j gráf modell a diagnosztikai kapcsolatokhoz
- huBERT a magyar nyelvű embeddingekhez
- Monorepo struktúra (backend + frontend együtt)

## TODO - Következő Sprintek

### Sprint 2-5: ✅ BEFEJEZVE
- [x] Neo4j seed adatok (26,816 node)
- [x] Qdrant HuBERT indexelés (35,000+ vector)
- [x] huBERT embedding service (embedding_service.py)
- [x] LangChain RAG chain (rag_service.py)
- [x] NHTSA API kliens (nhtsa_service.py)

### Sprint 6: ✅ BEFEJEZVE
- [x] Auth végpontok működőképessé
- [x] DTC keresés API endpoint
- [x] Vehicle lookup API
- [x] Frontend diagnosis wizard

### Sprint 7.5: ✅ BEFEJEZVE
- [x] LLM prompt újraírás (direktív utasítások, mérési értékek, szerszámok, gyökérok elemzés)
- [x] PartsPriceService integráció a diagnosis pipeline-ba (Step 5.5)
- [x] Frontend mock adatok eltávolítása, valódi API adatok megjelenítése
- [x] Új sémák: ToolNeeded, PartWithPrice, TotalCostEstimate
- [x] Alkatrészek és Árak táblázat + Összköltség becslés kártya (ResultPage)

### Sprint 8: ✅ BEFEJEZVE
- [x] Demo Error Simulation – P0300 (több hengeres égéskimaradás) VW Golf VII 1.4 TSI
- [x] DemoResultPage – teljes pre-filled bemutató oldal képpel, elemzéssel, javítási tervvel
- [x] PartStoreCard komponens – kártyás megjelenítés bolt-specifikus árakkal
- [x] Valós alkatrész árak: Bárdi Autó, Uni Autó, AUTODOC (2026. márciusi árak)
- [x] 6 alkatrész demó adattal: gyújtógyertya, gyújtótekercs, levegőszűrő, üzemanyagszűrő, injektor, lambda szonda
- [x] Demo route (/demo) + HomePage demo gomb
- [x] Agent Teams (7+ ügynök) implementáció: koordinátor + 7 specialist

### Sprint 9 - Security & CI Hardening: ✅ BEFEJEZVE
- [x] Tuple destructuring fix (`get_vehicles`, `get_reminders`)
- [x] Pydantic → dict konverzió (`create_vehicle`, `update_vehicle`, `create_reminder`)
- [x] Cascade invalidation fix (`useCompleteReminder` → targeted key)
- [x] Alembic migration downgrade fix (explicit index drops)
- [x] Post-Sprint Review: 3 CRITICAL + 6 HIGH biztonsági hiba javítva
- [x] Rate limiter fail-closed, JWT claim protection, SQL injection escape

### Sprint 10 - Leaflet Map + NHTSA Recalls: ✅ BEFEJEZVE
- [x] react-leaflet integráció ServiceComparisonPage-en (valós térkép, marker click, FlyToSelected)
- [x] NHTSA visszahívás badge ResultPage-en (piros kártya, campaign_number, summary, remedy)
- [x] VehicleDetailPage "Visszahívások" tab (useVehicleRecalls hook)
- [x] Backend recalls endpoint (`GET /garage/vehicles/{id}/recalls`)
- [x] RelatedRecall interface (`api.ts`), VehicleRecall interface (`garageService.ts`)
- [x] MyPy type: ignore fix (SQLAlchemy + Pydantic no-any-return)
- [x] CodeQL log injection fix (4 HIGH severity sanitize_log)
- [x] Alembic unused globals fix (branch_labels/depends_on eltávolítva, lgtm suppress)

### Header Refactor - Navigáció (#22): ✅ BEFEJEZVE
- [x] 11 elemű header → 4 intent-alapú dropdown (Diagnosztika / Garázs / Szerviz & Árak / Tudástár) + fiók menü
- [x] Akadálymentes `NavDropdown` (disclosure pattern)
- [x] Halott `/settings` link eltávolítva

### Sprint S1 - Igazmondó Frontend (#23): ✅ BEFEJEZVE
- [x] ResultPage truthfulness: fabrikált "főtengely" mondat / ONLINE badge / #4829 azonosító eltávolítva
- [x] ResultPage: `urgency`, `safety_warnings`, `diagnostic_steps`, `sources`, `similar_complaints` renderelése (http/https allowlist)
- [x] VehicleDetailPage "Panaszok" (complaints) tab
- [x] HomePage teljes redesign: marketing + bejelentkezett dashboard, őszinte stat strip (26 816 / 35 000+)
- [x] `NewDiagnosisPage.tsx` törölve
- [x] Publikus HomePage nem indít nem-hitelesített reminder-hívást

### Sprint S2 - Valós Garázs + Streaming Parts (#24): ✅ BEFEJEZVE
- [x] Valós `health_score` / `upcoming_reminders_count` egyetlen csoportosított aggregátumból (közös tiszta scoring, list↔health paritás)
- [x] Igazmondó HistoryPage: valós `vehicle_vin` / `symptoms_text`, élő szerver-oldali szűrők, `has_more` lapozás, törlés
- [x] Parts-dúsítás a streaming pipeline-ban (5s time-box, hiba-izoláció, perzisztálás-paritás)
- [x] BONUS: 5 élőben törött garázs-endpoint javítva (Pydantic UUID→str 500-ak) közös `UUIDStrModel` before-validator bázissal; hibás id → 404

### Sprint S3 - Settings/Profil + GDPR: ✅ BEFEJEZVE
- [x] Settings/Profil oldal (a #22-ben eltávolított `/settings` pótlása valós funkcióval)
- [x] GDPR: adat-export + fiók törlés UI a meglévő backend flow-ra kötve
- [x] Megosztott állapotok (shared loading/empty/error state komponensek) egységesítése + HistoryPage chrome-dedup
- [ ] Follow-up: app-szintű main→div landmark rendezés (5 további oldal)
- [ ] Follow-up: tegezés/magázás egységesítés

### Sprint S4 - Szemantikus keresés helyreállítása: ✅ BEFEJEZVE

**A sprint tárgya: három egymástól független hiba, amitől a zászlóshajó funkció (magyar szemantikus keresés) hónapok óta némán nem működött.**

- [x] **R1 — Production tud embedelni.** huBERT **ONNX Runtime fp32**-n (`EMBEDDING_BACKEND=onnx`), torch/transformers nélkül. Export egy eldobott `Dockerfile.prod` build-stage-ben, SHA-ra pinelt `HUBERT_REVISION`-ből
- [x] **R1b — A nullvektor-fallback megszüntetve.** Hiányzó backend → `EmbeddingUnavailableError` (`app/core/exceptions.py`, 503), soha nem `[0.0]*768`
- [x] **R2 — Qdrant collection-drift javítva.** A RAG a unified `autocognitix` collectiont kérdezi `type=dtc` / `type=complaint` lábbal; a halott `vehicle_make` szűrő eltávolítva
- [x] **R2b — A drift MARADÉK hívói is átirányítva:** `chat_service` (DTC-kontextus) és `consistency_service` (az admin konzisztencia-ellenőrzés maga is a drift áldozata volt — permanensen 0 vektort jelentett)
- [x] **R3 — Safety guardok.** Qdrant query-vektor norm guard (`ValueError` nullvektorra), `/health/detailed` embedding self-test, verziózott embedding cache-névtér (`EMBEDDING_CACHE_VERSION`)
- [x] **Build-kapu:** minden Docker build assertálja, hogy az exportált ONNX gráf reprodukálja a befagyasztott referenciavektorokat (`backend/tests/fixtures/hubert_reference_vectors.json` + `pooling_reference.json`)
- [x] **DTC szabály egységesítés:** `backend/app/core/dtc_codes.py` = single source of truth (SAE J2012, 2. karakter `0-3`). Tíz divergens regex kivezetve; `app/core/__init__.py` lazy re-exportra állítva, hogy a `scripts/` is importálhassa
- [x] **`common-issues` javítva:** 500 helyett 200; a rangsor NHTSA panasz-**komponens** gyakoriságra épül (`components`, `total_complaints`, `share`); `sources` státusz-objektum kiesés vs. valódi üres megkülönböztetésére; index a komponens-lekérdezéshez (`020_complaint_component_index.py`)
- [x] **Complaint → DTC import javítva:** a szigorú regex kihagyta a valódi hex kódokat, a mintavétel pedig a legkevésbé DTC-valószínű panaszokat választotta ki
- [x] **GDPR erasure javítva:** a `delete_by_user()` elsőként a unified collectiont törli és a hibákat **propagálja** (korábban csak üres legacy collectionöket söpört, minden hibát elnyelt, és sikert jelentett)
- [x] **Brand egységesítés:** MechanicAI / MechanicAI PRO → AutoCognitix
- [x] **10-szempontú adverzariális review** a sprint SAJÁT commitjain — confidence-score korrupció (rank fusion in-place írás), embedding singleton race, search↔detail DTC eltérés javítva
- [x] Dokumentáció: `docs/EMBEDDING_ARCHITECTURE_DECISION.md` tervből **ADR**-ré átírva (a szállított állapotot írja le), `ARCHITECTURE` / `DATABASE_MAP` / `DATA_FLOW` / `MIGRATIONS` collection-modell pontosítva

**Nyitott follow-upok (nem blokkoló):**
- [ ] `semantic_search_available` / `degraded_reason` a `/diagnosis/analyze` **válaszában** — ma a degradáció csak logban és a health endpointon látszik, a felhasználó felé nem
- [ ] Legacy `*_hu` collectionök kivezetése (az `initialize_collections()` még létrehozza őket üresen; `get_storage_stats()` **csak** ezeket nézi, a valódi vektortárolót nem)
- [ ] A `search_similar_symptoms()` / `search_components()` / `search_repairs()` halott kód eltávolítása (a legacy üres collectionökre mutatnak)
- [ ] **Élő adatbázis-számlálás** — a node/vektor számok forrásai ellentmondanak (ld. "Aktuális Adatbázis Állapot")
- [ ] Magyar retrieval kiértékelő halmaz (50–100 `query → várt DTC` pár) — enélkül semmilyen retrieval-minőség állítás nem bizonyítható

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

## CI/CD Tanulságok - KRITIKUS SZABÁLYOK

### SQLAlchemy Fenntartott Szavak
**SOHA** ne használd ezeket oszlopnévként:
- `metadata` → használj `sync_metadata`, `extra_data`
- `registry`, `query`, `columns`, `tables`

```python
# HELYTELEN
metadata: Mapped[dict | None] = mapped_column(JSONB)

# HELYES
sync_metadata: Mapped[dict | None] = mapped_column(JSONB)
```

### npm package-lock.json Sync
**MINDIG** futtasd `npm install`-t package.json változtatás után:
```bash
cd frontend && npm install
git add package.json package-lock.json
```

CI-ben **MINDIG** `npm ci`-t használj (nem `npm install`-t)!

### GitHub Actions Conditional Execution
```yaml
# Deployment CSAK ha minden check sikeres
deploy:
  needs: [lint, test, build]
  if: needs.lint.result == 'success' && needs.test.result == 'success'
```

### Railway Deployment
1. Dockerfile.prod tesztelése lokálisan
2. Health endpoint implementálása
3. Environment variables Railway Variables-ben
4. Alembic migráció CD workflow-ban

**Részletes dokumentáció:** `tasks/lessons.md`

## Kapcsolódó Dokumentumok

- `AutoCognitix_Teljeskoeru_Elemzes.docx` - Részletes elemzés
- `MVP Definíció & Gyakorlati Megvalósítás.pdf` - MVP specifikáció
- `tasks/lessons.md` - Tanulságok és hibajavítások részletesen
