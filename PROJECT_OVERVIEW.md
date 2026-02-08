# AutoCognitix - Project Overview for Grant Applications
# AutoCognitix - Projekt Attekintes Palyazati Anyagokhoz

---

## Executive Summary / Vezetoi Osszefoglalo

### English

**AutoCognitix** is an innovative AI-powered vehicle diagnostic platform that democratizes automotive troubleshooting by enabling accurate diagnosis without expensive OBD-II hardware. The platform combines cutting-edge natural language processing with a comprehensive knowledge graph to analyze diagnostic trouble codes (DTCs) and symptoms, providing expert-level repair recommendations in both Hungarian and English.

**Key Innovation Points:**
- **Hardware-Free Diagnostics**: Manual DTC code and symptom input eliminates the need for expensive diagnostic equipment
- **Native Hungarian NLP**: First-of-its-kind implementation using SZTAKI huBERT for Hungarian automotive terminology
- **Multi-Database AI Architecture**: Combines PostgreSQL, Neo4j graph database, and Qdrant vector search for comprehensive analysis
- **RAG-Powered Intelligence**: Retrieval-Augmented Generation provides contextual, accurate diagnostic recommendations

**Current Achievements:**
- 26,000+ nodes in Neo4j diagnostic knowledge graph
- 21,000+ semantic embeddings in Qdrant vector database
- 3,579+ DTC codes with Hungarian translations
- 738+ vehicle configurations from OBDb database
- 520+ vehicle components and 320+ repair procedures mapped

---

### Magyar

Az **AutoCognitix** egy innovativ, mesterseges intelligenciaval tamogatott gepjarmu-diagnosztikai platform, amely demokratizalja az autoszerelest azaltal, hogy draga OBD-II hardver nelkul tesz lehetove pontos diagnozizt. A platform elen jaro termeszetes nyelvfeldolgozast kombinal atfogo tudagraffal a diagnosztikai hibakodok (DTC) es tunetek elemzesehez, szakertoi szintu javitasi javaslatokat nyujtva magyarul es angolul.

**Fo innovacios pontok:**
- **Hardver nelkuli diagnosztika**: A manualis DTC kod es tunet bevitel kiikusobolja a draga diagnosztikai eszkozok szuksegesseget
- **Nativ magyar NLP**: Elsoként alkalmazott megoldas a SZTAKI huBERT modellel magyar gepjarmu-terminologiahoz
- **Tobbadatbazisos AI architektura**: PostgreSQL, Neo4j graf adatbazis es Qdrant vektor kereses kombinacioja atfogo elemzeshez
- **RAG-alapu intelligencia**: A Retrieval-Augmented Generation kontextualis, pontos diagnosztikai javaslatokat biztosit

**Jelenlegi eredmenyek:**
- 26 000+ csomoport a Neo4j diagnosztikai tudagrafban
- 21 000+ szemantikus embedding a Qdrant vektor adatbazisban
- 3 579+ DTC kod magyar forditassal
- 738+ jarmu konfiguracio az OBDb adatbazisbol
- 520+ jarmu-alkatresz es 320+ javitasi eljaras felterkepezve

---

## Innovation Description / Innovacio Leirasa

### The Problem / A Problema

**English:**
Vehicle diagnostics traditionally require expensive OBD-II scanners (50-500 EUR) and specialized knowledge to interpret error codes. Independent mechanics and car owners face significant barriers:
- High cost of professional diagnostic equipment
- Language barriers (most resources are English-only)
- Complexity of modern vehicle electronics
- Fragmented information sources for repair guidance

**Magyar:**
A gepjarmuvek hagyomanyos diagnosztikaja draga OBD-II leolvasokat (50-500 EUR) es szakismeretet igenyel a hibakodok ertelmezesehez. A fuggetlen szerelok es autotulatdonosok jelentos akadalyokkal szembesulnek:
- Professzionalis diagnosztikai eszkozok magas ara
- Nyelvi korlatozások (a legtobb forras csak angol nyelvu)
- Modern jarmu-elektronika bonyolultsaga
- Javitasi utmutatok szetszorodott informacioforrasai

### Our Solution / Megoldasunk

**English:**
AutoCognitix addresses these challenges through:

1. **Hardware-Free Input**: Users manually enter DTC codes displayed on their dashboard or describe symptoms in natural language
2. **Hungarian Language AI**: Native understanding of Hungarian automotive terminology using SZTAKI's huBERT model
3. **Knowledge Graph Intelligence**: Neo4j-based diagnostic paths connecting 26,000+ nodes (DTCs, symptoms, components, repairs)
4. **Semantic Search**: Qdrant vector database enables finding related issues through meaning, not just keywords
5. **Multi-Source Data Aggregation**: Integrates NHTSA recalls, complaints, OBDb specifications, and repair databases

**Magyar:**
Az AutoCognitix ezeket a kihivasokat a kovetkezokeppen kezeli:

1. **Hardver nelkuli bevitel**: A felhasznalok manuaisan adják meg a muszerfalon megjeleno DTC kodokat vagy termeszetes nyelven leirjak a tuneteket
2. **Magyar nyelvu AI**: A magyar gepjarmu-terminologia nativ megertese a SZTAKI huBERT modelljevel
3. **Tudagraf intelligencia**: Neo4j-alapu diagnosztikai utvonalak, amelyek 26 000+ csomopontot kotnek ossze (DTC-k, tunetek, alkatreszek, javitasok)
4. **Szemantikus kereses**: A Qdrant vektor adatbazis lehetove teszi a kapcsolodo problemak megtalalasat jelentestartalom alapjan, nem csak kulcsszavakkal
5. **Tobbforrasos adataggregacio**: Integralja az NHTSA visszahivasokat, panaszokat, OBDb specifikaciokat es javitasi adatbazisokat

---

## Technical Architecture / Muszaki Architektura

### System Overview / Rendszer Attekintes

```
+------------------------------------------------------------------+
|                        USER INTERFACE                             |
|            React 18 + TypeScript + TailwindCSS                    |
+------------------------------------------------------------------+
                                |
                                v
+------------------------------------------------------------------+
|                         API LAYER                                 |
|                    FastAPI + Pydantic V2                          |
|                   JWT Authentication                              |
+------------------------------------------------------------------+
                                |
        +----------+------------+-----------+----------+
        |          |            |           |          |
        v          v            v           v          v
+----------+  +----------+  +--------+  +-------+  +--------+
|PostgreSQL|  |  Neo4j   |  | Qdrant |  | Redis |  |  LLM   |
|   16     |  |   5.x    |  |  1.x   |  |   7   |  |Provider|
+----------+  +----------+  +--------+  +-------+  +--------+
| Structured|  |Knowledge |  | Vector |  | Cache |  |Claude/ |
| Data      |  | Graph    |  | Search |  |       |  |GPT-4   |
+----------+  +----------+  +--------+  +-------+  +--------+
```

### Technology Stack / Technologiai Verem

| Layer | Technology | Purpose / Cel |
|-------|------------|---------------|
| **Frontend** | React 18, TypeScript, Vite, TailwindCSS | Modern, responsive user interface / Modern, reszponziv felhasznaloi felulet |
| **Backend** | FastAPI, Python 3.11+, SQLAlchemy 2.0 | High-performance async API / Nagy teljesitmenyu aszinkron API |
| **Primary DB** | PostgreSQL 16 | Structured data, user management / Strukturalt adatok, felhasznalokezeles |
| **Graph DB** | Neo4j 5.x | Diagnostic knowledge graph / Diagnosztikai tudagraf |
| **Vector DB** | Qdrant | Semantic similarity search / Szemantikus hasonlosagi kereses |
| **Cache** | Redis 7 | Performance caching / Teljesitmeny cache |
| **NLP** | huBERT (SZTAKI-HLT), HuSpaCy | Hungarian language processing / Magyar nyelvfeldolgozas |
| **AI/RAG** | LangChain, Anthropic Claude / OpenAI | Intelligent response generation / Intelligens valaszgeneralas |

### Multi-Database Architecture / Tobbadatbazisos Architektura

**English:**
AutoCognitix employs a polyglot persistence architecture where each database type is optimized for its specific use case:

1. **PostgreSQL**: Handles structured data including users, diagnosis sessions, vehicle specifications, and NHTSA data. Optimized with 15+ performance indexes including GIN indexes for array operations and full-text search.

2. **Neo4j**: Stores the diagnostic knowledge graph with nodes for:
   - DTCCode (3,579 nodes)
   - Symptom (117+ nodes)
   - Component (520+ nodes)
   - Repair (320+ nodes)
   - Vehicle relationships

3. **Qdrant**: Vector database storing 768-dimensional huBERT embeddings for:
   - DTC code descriptions (Hungarian/English)
   - Symptom descriptions
   - Component descriptions
   - Repair procedures

4. **Redis**: Caching layer with tiered TTLs:
   - DTC codes: 1 hour
   - Search results: 15 minutes
   - Embeddings: 1 hour
   - NHTSA data: 6 hours

**Magyar:**
Az AutoCognitix poliglott perzisztencia architekturathasznál, ahol minden adatbazistipus az adott felhasznalasi esetre optimalizalt:

1. **PostgreSQL**: Strukturalt adatok kezelese, beleertve a felhasznalokat, diagnozis munkameneteket, jarmu specifikaciokat es NHTSA adatokat. 15+ teljesitmeny indexszel optimalizalva, beleertve GIN indexeket tombuműveletekhez es teljes szoveges kereséshez.

2. **Neo4j**: A diagnosztikai tudagraf tarolasa csomopontokkal:
   - DTCCode (3 579 csomopont)
   - Symptom (117+ csomopont)
   - Component (520+ csomopont)
   - Repair (320+ csomopont)
   - Jarmu kapcsolatok

3. **Qdrant**: Vektor adatbazis 768 dimenzios huBERT embeddingekkel:
   - DTC kod leirasok (magyar/angol)
   - Tunet leirasok
   - Alkatresz leirasok
   - Javitasi eljarasok

4. **Redis**: Cache reteg rangsorolt eljarasi idokkel:
   - DTC kodok: 1 ora
   - Keresesi eredmenyek: 15 perc
   - Embeddingek: 1 ora
   - NHTSA adatok: 6 ora

---

## Data & AI Capabilities / Adatok es AI Kepessegek

### Hungarian NLP Innovation / Magyar NLP Innovacio

**English:**
AutoCognitix is the first automotive diagnostic platform to implement native Hungarian language understanding using:

- **huBERT Model**: SZTAKI-HLT/hubert-base-cc - Hungarian BERT trained on Common Crawl
- **768-dimensional embeddings**: Semantic vector representations optimized for Hungarian text
- **HuSpaCy preprocessing**: Hungarian lemmatization and stopword removal
- **Custom automotive glossary**: 200+ Hungarian-English automotive term mappings

The system handles Hungarian automotive terminology including:
- Technical terms: "levegoetomeg-mero" (MAF sensor), "oxigenszenzor" (O2 sensor)
- Symptom descriptions: "egyenetlen alapjarat" (rough idle), "teljesitmenyvesztes" (power loss)
- Colloquial expressions: "berreges" (vibration), "kopogatas" (knocking)

**Magyar:**
Az AutoCognitix az elso gepjarmu-diagnosztikai platform, amely nativ magyar nyelvmegertest valősit meg a kovetkezok felhasznalasaval:

- **huBERT modell**: SZTAKI-HLT/hubert-base-cc - Magyar BERT, Common Crawl-on trenirozva
- **768 dimenzios embeddingek**: Szemantikus vektor reprezentaciok magyar szovegre optimalizalva
- **HuSpaCy elofeldolgozas**: Magyar lemmatizacio es stopszavak eltavolitasa
- **Egyedi gepjarmu-szoszedet**: 200+ magyar-angol gepjarmuipari kifejezesek felterkepezve

A rendszer kezeli a magyar gepjarmuipari terminológiat, beleertve:
- Muszaki kifejezesek: "levegoetomeg-mero" (MAF szenzor), "oxigenszenzor" (O2 szenzor)
- Tunet leirasok: "egyenetlen alapjarat", "teljesitmenyvesztes"
- Kozbeszedben hasznalt kifejezesek: "berreges", "kopogatas"

### Knowledge Graph Structure / Tudagraf Struktura

```
                    +-------------+
                    |   Vehicle   |
                    +------+------+
                           |
            +--------------+--------------+
            |              |              |
            v              v              v
      +---------+    +-----------+   +----------+
      | DTCCode |    |  Symptom  |   | Platform |
      +----+----+    +-----+-----+   +----------+
           |               |
           |   CAUSES      |
           +-------+-------+
                   |
                   v
            +------------+
            | Component  |
            +-----+------+
                  |
                  | REPAIRED_BY
                  v
            +-----------+
            |  Repair   |
            +-----------+
```

### Data Sources / Adatforrasok

| Source | Data Type | Volume | Status |
|--------|-----------|--------|--------|
| **OBDb GitHub** | Vehicle configurations | 738+ vehicles | Integrated |
| **NHTSA API** | Recalls, complaints, TSBs | 50,000+ records | Live API |
| **obd-trouble-codes** | Generic DTC codes | 11,000+ codes | Imported |
| **Klavkarr** | Extended DTC database | 11,000+ codes | Imported |
| **Custom translations** | Hungarian descriptions | 3,579+ codes | Translated |

### AI Pipeline / AI Csovezetek

```
User Input (Hungarian/English)
         |
         v
+------------------+
| Text Preprocessing|  <-- HuSpaCy lemmatization
+------------------+
         |
         v
+------------------+
| Embedding        |  <-- huBERT 768-dim vectors
| Generation       |
+------------------+
         |
         v
+------------------+
| Vector Search    |  <-- Qdrant similarity search
| (Top-K results)  |
+------------------+
         |
         v
+------------------+
| Graph Traversal  |  <-- Neo4j path finding
| (Related nodes)  |
+------------------+
         |
         v
+------------------+
| Context Assembly |  <-- LangChain RAG
+------------------+
         |
         v
+------------------+
| LLM Generation   |  <-- Claude/GPT-4
| (Diagnosis)      |
+------------------+
         |
         v
Structured Response (Hungarian/English)
```

---

## Market Potential / Piaci Potencial

### Target Market / Celpiax

**English:**

1. **Independent Auto Repair Shops**
   - EU: 350,000+ businesses
   - Hungary: 8,000+ businesses
   - Pain point: Expensive diagnostic equipment and training

2. **Car Owners (DIY Market)**
   - EU: 250 million registered vehicles
   - Hungary: 4 million registered vehicles
   - Pain point: Understanding error codes without mechanic visits

3. **Automotive Education**
   - Technical schools and vocational training
   - Pain point: Affordable training tools with local language support

4. **Fleet Management Companies**
   - Quick diagnostics for vehicle fleets
   - Pain point: Reducing downtime and repair costs

**Magyar:**

1. **Fuggetlen autoszervizek**
   - EU: 350 000+ vallalkozas
   - Magyarorszag: 8 000+ vallalkozas
   - Fajdalompont: Draga diagnosztikai eszkozok es kepzes

2. **Autotulatdonosok (barkacs piac)**
   - EU: 250 millio regisztralt jarmu
   - Magyarorszag: 4 millio regisztralt jarmu
   - Fajdalompont: Hibakodok megertese szervizlátogatas nelkul

3. **Gepjarmu oktatas**
   - Szakkozepiskolk es szakkepzes
   - Fajdalompont: Megfizheto kepzesi eszkozok helyi nyelvi tamogatassal

4. **Flottakezelo cegek**
   - Gyors diagnosztika jarmufottakhoz
   - Fajdalompont: Allasido es javitasi koltsegek csokkentese

### Competitive Advantage / Verseny elony

| Feature | AutoCognitix | Traditional Scanners | Generic Apps |
|---------|--------------|---------------------|--------------|
| Hardware Required | None | Yes (50-500 EUR) | Yes |
| Hungarian Language | Native | None | None |
| AI-Powered Analysis | Yes | No | Limited |
| Knowledge Graph | 26K+ nodes | No | No |
| Repair Recommendations | Detailed | Basic | Limited |
| NHTSA Integration | Yes | No | Rare |
| Cost Model | Subscription | One-time + updates | Subscription |

### Market Size / Piac merete

- **Total Addressable Market (TAM)**: 2.5B EUR (European automotive aftermarket digital tools)
- **Serviceable Addressable Market (SAM)**: 150M EUR (Independent shops + DIY diagnostics)
- **Serviceable Obtainable Market (SOM)**: 15M EUR (Hungarian market + CEE expansion)

---

## Development Roadmap / Fejlesztesi Utemterv

### Phase 1: Foundation (Completed / Befejezve)
**Q4 2025 - Q1 2026**

- [x] Core platform architecture
- [x] Multi-database integration (PostgreSQL, Neo4j, Qdrant, Redis)
- [x] Hungarian NLP pipeline with huBERT
- [x] Basic DTC lookup and search
- [x] NHTSA API integration
- [x] 3,579+ DTC codes with Hungarian translations
- [x] Knowledge graph with 26,000+ nodes
- [x] Vector embeddings for 21,000+ entries
- [x] Railway cloud deployment

### Phase 2: Enhancement (In Progress / Folyamatban)
**Q2 2026**

- [ ] Complete Hungarian translation coverage (80%+)
- [ ] Advanced symptom-based diagnosis
- [ ] Repair cost estimation
- [ ] Mobile-responsive interface
- [ ] User accounts and diagnosis history
- [ ] Extended vehicle database (1,500+ models)

### Phase 3: AI Evolution (Planned / Tervezett)
**Q3-Q4 2026**

- [ ] Multi-modal input (photo analysis of dashboard warnings)
- [ ] Voice input for hands-free diagnosis
- [ ] Predictive maintenance recommendations
- [ ] Integration with repair shop management systems
- [ ] API for third-party integrations
- [ ] B2B enterprise features

### Phase 4: Market Expansion (Planned / Tervezett)
**2027**

- [ ] German language support
- [ ] Polish language support
- [ ] Romanian language support
- [ ] EU-wide regulatory compliance data
- [ ] Electric vehicle diagnostics extension
- [ ] Hybrid powertrain specialization

---

## Technical Achievements / Muszaki Eredmenyek

### Performance Metrics / Teljesitmeny Mutatók

| Metric | Current Value | Target |
|--------|---------------|--------|
| DTC Lookup Latency | <100ms | <50ms |
| Semantic Search | <200ms | <100ms |
| Full Diagnosis | <2s | <1s |
| API Availability | 99.5% | 99.9% |
| Cache Hit Rate | 85% | 95% |

### Database Statistics / Adatbazis Statisztikak

| Database | Metric | Value |
|----------|--------|-------|
| PostgreSQL | Tables | 15+ |
| PostgreSQL | Indexes | 25+ |
| Neo4j | Nodes | 26,000+ |
| Neo4j | Relationships | 8,000+ |
| Qdrant | Vectors | 21,000+ |
| Qdrant | Dimension | 768 |
| Redis | Cache Keys | 10,000+ avg |

### Code Quality / Kod minoseg

- **Test Coverage**: Unit tests, integration tests, E2E tests
- **CI/CD Pipeline**: GitHub Actions with automated deployment
- **Security Scanning**: CodeQL, Bandit, npm audit, Trivy
- **Code Quality**: Ruff linting, MyPy type checking
- **Documentation**: API docs (OpenAPI), user manuals (HU/EN)

---

## Team & Resources / Csapat es Eroforrasok

### Current Infrastructure / Jelenlegi Infrastruktura

| Service | Provider | Tier |
|---------|----------|------|
| Backend Hosting | Railway | Production |
| Frontend Hosting | Railway | Production |
| PostgreSQL | Railway | Managed |
| Redis | Railway | Managed |
| Neo4j | Neo4j Aura | Cloud Free |
| Qdrant | Qdrant Cloud | Cloud Free |
| LLM | Anthropic Claude | API |
| CI/CD | GitHub Actions | Free |

### Resource Requirements / Eroforras igenyek

**For Production Scale / Termelesi mertekhez:**
- Compute: 4 vCPU, 8GB RAM (backend)
- Database: PostgreSQL 16GB, Neo4j 8GB, Qdrant 4GB
- GPU: Optional for local embedding (NVIDIA T4 or equivalent)
- Bandwidth: 100GB/month estimated

---

## Innovation Highlights / Innovacios Kiemelesek

### Novel Contributions / Uj Hozzajarulasok

1. **First Hungarian Automotive NLP System**
   - No existing solution provides native Hungarian language understanding for vehicle diagnostics
   - Custom training data and glossary for Hungarian automotive terminology

2. **Hardware-Free Diagnostic Paradigm**
   - Eliminates the traditional requirement for OBD-II readers
   - Enables diagnosis through symptom description alone

3. **Multi-Database AI Architecture**
   - Unique combination of relational, graph, and vector databases
   - Each database optimized for its specific query patterns

4. **Open Data Aggregation**
   - First platform to combine NHTSA, OBDb, and multiple DTC databases
   - Enriched with Hungarian translations and local repair costs

5. **RAG for Automotive Domain**
   - Specialized prompt engineering for vehicle diagnostics
   - Context-aware generation with knowledge graph integration

---

## Contact / Kapcsolat

**Project Repository**: [GitHub - AutoCognitix]
**Documentation**: `/docs` directory
**Technical Contact**: Development Team

---

*Document Version: 1.0*
*Last Updated: 2026-02-08*
*Language: Bilingual (English/Hungarian)*
