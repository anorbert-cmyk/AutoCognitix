# AutoCognitix - Palyazati Osszefoglalo
# Grant Application Summary

---

## 1. Projekt Osszefoglalo / Project Summary

### 1.1 Projekt neve / Project Name
**AutoCognitix** - AI-alapu Gepjarmu-diagnosztikai Platform

### 1.2 Rovid leiras / Brief Description

**Magyar:**
Az AutoCognitix egy innovativ, mesterseges intelligenciaval tamogatott gepjarmu-diagnosztikai platform, amely lehetove teszi a jarmuhibak pontos azonositasat draga OBD-II hardver nelkul. A platform egyesiti az elenjaró termeszetes nyelvfeldolgozast (NLP) egy atfogo tudasgraffal, hogy elemezze a diagnosztikai hibakodokat (DTC) es tuneteket, szakertoi szintu javitasi javaslatokat nyujtva magyar es angol nyelven. Legalabb azonban, a platform nem csak azonositja a problemat, hanem **lépésről-lépésre szerelési útmutatót** nyújt magyar nyelven, végigvezetive a felhasználókat az egész javítási folyamaton.

**English:**
AutoCognitix is an innovative AI-powered vehicle diagnostic platform that enables accurate vehicle fault identification without expensive OBD-II hardware. The platform combines cutting-edge natural language processing (NLP) with a comprehensive knowledge graph to analyze diagnostic trouble codes (DTCs) and symptoms, providing expert-level repair recommendations in Hungarian and English. Importantly, the platform not only identifies the problem but provides **step-by-step repair guidance in Hungarian**, walking users through the entire repair process.

---

## 2. Innovacios Tartalom / Innovation Content

### 2.1 Fo Innovaciok / Key Innovations

| Innovacio | Leiras | Versenyelony |
|-----------|--------|--------------|
| **Hardver nelkuli diagnosztika** | Manualis DTC kod es tunet bevitel, OBD-II olvasó nelkul | Elso ilyen megoldas a piacon |
| **Nativ magyar NLP** | SZTAKI huBERT modell magyar gepjarmu-terminologiahoz | Elso magyar nyelvu autodiagnosztikai AI |
| **Multi-adatbazisos architektura** | PostgreSQL + Neo4j + Qdrant + Redis | Egyedi kombinalas optimalis teljesitmenyhez |
| **RAG-alapu tudas** | Retrieval-Augmented Generation kontextualis valaszokhoz | 160 000+ csomopont tudagrafbol |

### 2.2 Szabadalmi Potencial / Patent Potential

1. **Magyar nyelvu gepjarmu-diagnosztikai NLP rendszer**
   - Elsoként alkalmazott huBERT embedding gepjarmu domenben
   - Egyedi magyar-angol gepjarmu terminologia szotar (500+ kifejezes)

2. **Hardver nelkuli diagnosztikai metodologia**
   - Tunet-alapu diagnosztika természetes nyelvi leirasból
   - Multi-modal bemenet (szoveg, kesobbi fazisban kep)

3. **Hibrid adatbazis architektura AI diagnosztikához**
   - Graf + vektor + relacios adatbazisok optimalis kombinacioja
   - Valós ideju kontextus-epitó algoritmus

---

## 3. Piaci Helyzet / Market Analysis

### 3.1 Celpiaci Szegmensek / Target Market Segments

| Szegmens | Meret (EU) | Meret (HU) | Fajdalompont |
|----------|------------|------------|--------------|
| **Fuggetlen autoszervizek (B2B)** ⭐ | **350 000+** | **8 000+** | **Draga diagnosztikai eszkozok** |
| Autotulatdonosok (DIY, B2C retail) | 250M jarmu | 4M jarmu | Hibakodok ertelmezese, DIY szerelesi utmutatas |
| Flottakezelo cegek (Enterprise) | 50 000+ | 500+ | Gyors hibaelhárítás |
| Gepjarmu oktatas | 15 000+ iskola | 200+ iskola | Megfizheto kepzesi eszkoz |

### 3.2 Piaci Meret / Market Size

- **TAM (Total Addressable Market)**: 2.5 milliard EUR (Europai gepjarmu utopiac digitalis eszkozok)
- **SAM (Serviceable Addressable Market)**: 150 millio EUR (Fuggetlen szervizek + DIY diagnosztika)
- **SOM (Serviceable Obtainable Market)**: 15 millio EUR (Magyar piac + KKE terjeszedes)

### 3.3 Versenyhelyzet / Competitive Landscape

| Jellemzo | AutoCognitix | Hagyomanyos szkenneren | Generikus appok |
|----------|--------------|------------------------|-----------------|
| Hardver szukseges | Nem | Igen (50-500 EUR) | Igen |
| Magyar nyelv | Nativ | Nincs | Nincs |
| AI elemzes | Igen | Nem | Korlatozott |
| Tudagraf | 160K+ node | Nem | Nem |
| Javitási javaslat | Részletes | Alap | Korlátozott |
| NHTSA integráció | Igen | Nem | Ritka |

---

## 4. Technikai Specifikació / Technical Specification

### 4.1 Rendszer Architektúra / System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FELHASZNALOI FELULET                     │
│              React 18 + TypeScript + TailwindCSS            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        API RETEG                            │
│                  FastAPI + Pydantic V2                      │
│                   JWT Authentikáció                         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────┬───────────┼───────────┬─────────┐
        ▼         ▼           ▼           ▼         ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ ┌─────────┐
   │PostgreSQL│ │  Neo4j  │ │ Qdrant  │ │ Redis │ │   LLM   │
   │    16   │ │   5.x   │ │   1.x   │ │   7   │ │Provider │
   └─────────┘ └─────────┘ └─────────┘ └───────┘ └─────────┘
   │Strukturalt│ │Tudagraf │ │Vektor   │ │Cache  │ │Claude/ │
   │adatok    │ │         │ │kereses  │ │       │ │GPT-4   │
   └─────────┘ └─────────┘ └─────────┘ └───────┘ └─────────┘
```

### 4.2 Technológiai Verem / Technology Stack

| Réteg | Technológia | Verzió | Cél |
|-------|-------------|--------|-----|
| Frontend | React, TypeScript, Vite | 18.x | Modern SPA |
| Backend | FastAPI, Python | 3.11+ | Aszinkron API |
| Primary DB | PostgreSQL | 16 | Strukturált adatok |
| Graph DB | Neo4j | 5.x | Diagnosztikai gráf |
| Vector DB | Qdrant | 1.x | Szemantikus keresés |
| Cache | Redis | 7.x | Teljesítmény cache |
| NLP | huBERT (SZTAKI-HLT) | - | Magyar embedding |
| AI | LangChain + Claude/GPT-4 | - | RAG generálás |

### 4.3 Adatbázis Statisztikák (V1 Cél) / Database Statistics (V1 Target)

| Adatbázis | Metrika | V1 Célérték |
|-----------|---------|-------------|
| Neo4j | Összes csomópont | 160 000+ |
| Neo4j | Járművek | 1 500+ |
| Neo4j | DTC kódok | 11 000+ |
| Neo4j | Panaszok | 146 000+ |
| Neo4j | Visszahívások | 2 100+ |
| Qdrant | Vektorok | 170 000+ |
| Qdrant | Dimenzió | 768 (HuBERT) |

---

## 5. Fejlesztési Ütemterv / Development Roadmap

### Fázis 1: Alap (Befejezve) - Q4 2025 - Q1 2026
- [x] Platform architektúra
- [x] Multi-adatbázis integráció
- [x] Magyar NLP pipeline huBERT-tel
- [x] NHTSA API integráció
- [x] Railway felhő telepítés
- [x] CI/CD pipeline biztonsági szkennelés

### Fázis 2: Bővítés (Folyamatban) - Q2 2026
- [ ] 80%+ magyar fordítás lefedettség
- [ ] Haladó tünet-alapú diagnózis
- [ ] Javítási költségbecslés (HUF)
- [ ] Per-analízis fizetési rendszer (pay-per-analysis)
- [ ] PWA mobil-reszponzív alkalmazás
- [ ] Felhasználói fiókok és diagnózis előzmények

### Fázis 3: AI Evolúció (Tervezett) - Q3-Q4 2026
- [ ] Natív mobil alkalmazás (iOS/Android)
- [ ] Fotó-alapú bemenet (műszerfal fotó elemzés)
- [ ] Hangalapú bemenet
- [ ] Prediktív karbantartás
- [ ] B2B API integráció

### Fázis 4: Piac Bővítés (Tervezett) - 2027
- [ ] Német nyelv támogatás
- [ ] Lengyel nyelv támogatás
- [ ] Román nyelv támogatás
- [ ] Elektromos jármű diagnózis

---

## 6. Pénzügyi Terv / Financial Plan

### 6.1 Üzleti Modell / Business Model

| Modell | Célcsoport | Árazás | Főbb jellemzők |
|--------|------------|--------|-----------------|
| **B2B (Autószervizek)** | Független szervizek | 19 990 - 29 990 Ft/hó | Korlátlan elemzés, teljes diagnosztikai tudásgráf, javítási útmutatók |
| **B2C Interaktív** | DIY felhasználók | 9 900 Ft/elemzés | Interaktív AI elemzés - diagnózis + kérdezhetsz az AI-tól, beszélgetés a problémáról |
| **B2C Útmutató** | DIY felhasználók | 4 900 Ft/elemzés | AI végigvezet a javítási folyamaton lépésről-lépésre, alkatrészlista, költségbecslés |
| **Enterprise (Flották)** | Flottakezelo cegek | Egyedi árazás | Magas volumen, dedikált support |
| **API** | Fejlesztők | Pay-per-use | Integrációs lehetőség harmadik féltől |

### 6.2 Bevételi Előrejelzés / Revenue Forecast

| Időszak | B2B szervizek | B2C użít | MRR (HUF) | ARR (HUF) |
|---------|--------------|---------|-----------|-----------|
| 2026 Q3 | 30 szerviz | Retail indulás | 750 000 | 9 000 000 |
| 2026 Q4 | 60 szerviz | Retail növekedés | 2 000 000 | 24 000 000 |
| 2027 Q1 | 120 szerviz | Retail bővülés | 4 200 000 | 50 400 000 |
| 2027 Q2 | 200 szerviz | Retail kiterjesztés | 9 000 000 | 108 000 000 |

### 6.3 Költségstruktúra / Cost Structure

| Költségelem | Havi (HUF) | Éves (HUF) |
|-------------|------------|------------|
| Felhő infrastruktúra (Railway) | 150 000 | 1 800 000 |
| Neo4j Aura (Professional) | 100 000 | 1 200 000 |
| Qdrant Cloud (Professional) | 80 000 | 960 000 |
| LLM API (Claude/GPT-4) | 200 000 | 2 400 000 |
| **Összesen infra** | **530 000** | **6 360 000** |

---

## 7. Csapat és Erőforrások / Team & Resources

### 7.1 Szükséges Kompetenciák / Required Competencies

| Pozíció | Felelősség | Státusz |
|---------|------------|---------|
| Lead Developer | Backend, AI/ML, DevOps | Aktív |
| Frontend Developer | React, UX/UI | Szükséges |
| NLP Specialist | Magyar nyelv, embedding | Konzultáns |
| Product Manager | Üzletfejlesztés | Szükséges |

### 7.2 Infrastruktúra (V1 Célállapot) / Infrastructure (V1 Target)

| Szolgáltatás | Provider | Szint |
|--------------|----------|-------|
| Backend & Frontend | Railway | Production |
| PostgreSQL | Railway | Managed |
| Redis | Railway | Managed |
| Neo4j | Neo4j Aura | Professional |
| Qdrant | Qdrant Cloud | Professional |
| LLM | Anthropic Claude | Enterprise |
| CI/CD | GitHub Actions | Pro |

---

## 8. Kockázatelemzés / Risk Analysis

### 8.1 Technikai Kockázatok / Technical Risks

| Kockázat | Valószínűség | Hatás | Kezelés |
|----------|--------------|-------|---------|
| LLM API költség növekedés | Közepes | Magas | Lokális modell fallback |
| Adatminőség (NHTSA) | Alacsony | Közepes | Validáció + tisztítás |
| Skálázhatóság | Alacsony | Magas | Horizontális skálázás |

### 8.2 Piaci Kockázatok / Market Risks

| Kockázat | Valószínűség | Hatás | Kezelés |
|----------|--------------|-------|---------|
| Versenytárs belépés | Közepes | Közepes | Első mozgó előny |
| Alacsony adopció | Alacsony | Magas | Freemium modell |
| Szabályozási változás | Alacsony | Közepes | Compliance monitoring |

---

## 9. Pályázati Igény Összefoglaló / Grant Request Summary

### 9.1 Igényelt Támogatás / Requested Funding

| Kategória | Összeg (HUF) | % |
|-----------|--------------|---|
| Fejlesztési költségek | 15 000 000 | 50% |
| Infrastruktúra (12 hónap) | 6 360 000 | 21% |
| Marketing & sales | 5 000 000 | 17% |
| Jogi, admin | 2 000 000 | 7% |
| Tartalék | 1 640 000 | 5% |
| **Összesen** | **30 000 000** | **100%** |

### 9.2 Felhasználási Ütemterv / Funding Timeline

| Negyedév | Összeg (HUF) | Fő tevékenység |
|----------|--------------|----------------|
| Q2 2026 | 8 000 000 | V1 befejezés, béta teszt |
| Q3 2026 | 10 000 000 | Publikus launch, marketing |
| Q4 2026 | 7 000 000 | Bővítés, B2B fejlesztés |
| Q1 2027 | 5 000 000 | KKE terjeszkedés előkészítés |

---

## 10. Kapcsolat / Contact

**Projekt Repository**: GitHub - AutoCognitix
**Technikai Dokumentáció**: /docs mappa
**API Dokumentáció**: /api/docs (OpenAPI/Swagger)

---

*Dokumentum verzió: 1.0*
*Utolsó frissítés: 2026-02-08*
*Célkiadás: V1 Production (Q2 2026)*
