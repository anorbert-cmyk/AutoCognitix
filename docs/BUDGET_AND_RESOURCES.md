# AutoCognitix - Koltsegvetes es Eroforras Terv
# Budget and Resource Plan

---

## 1. Projekt Idovonal / Project Timeline

### 1.1 Fazisok Attekintese / Phase Overview

| Fazis | Idotartam | Kezdet | Befejezés | Státusz |
|-------|-----------|--------|-----------|---------|
| **Fázis 1**: Alapok | 4 hónap | 2025 Q4 | 2026 Q1 | Befejezve |
| **Fázis 2**: Bővítés | 3 hónap | 2026 Q2 | 2026 Q2 | Folyamatban |
| **Fázis 3**: AI Evolúció | 6 hónap | 2026 Q3 | 2026 Q4 | Tervezett |
| **Fázis 4**: Piacbővítés | 12 hónap | 2027 Q1 | 2027 Q4 | Tervezett |

### 1.2 Reszletes Utemterv / Detailed Schedule

```
2025 Q4 ──────────────────────────────────────────────────────────
    │ ✅ Rendszer architektúra tervezés
    │ ✅ Multi-adatbázis infrastruktúra
    │ ✅ Backend alap API-k
    │ ✅ HuBERT NLP integráció

2026 Q1 ──────────────────────────────────────────────────────────
    │ ✅ Frontend React alkalmazás
    │ ✅ NHTSA API integráció
    │ ✅ Neo4j tudásgráf betöltés
    │ ✅ Qdrant vektor indexelés
    │ ✅ Railway deployment
    │ ✅ CI/CD pipeline

2026 Q2 ──────────────────────────────────────────────────────────
    │ ⏳ Magyar fordítás 80%+ lefedettség
    │ ⏳ Haladó tünet-alapú diagnózis
    │ ⏳ Javítási költségbecslés
    │ ⏳ Mobil-reszponzív UI
    │ ⏳ Publikus béta indítás
    │ ⏳ 200+ automatizált teszt

2026 Q3-Q4 ───────────────────────────────────────────────────────
    │ 📋 Multi-modal bemenet (fotó elemzés)
    │ 📋 Hangalapú bevitel
    │ 📋 Prediktív karbantartás
    │ 📋 B2B API integráció
    │ 📋 Enterprise funkciók

2027 ─────────────────────────────────────────────────────────────
    │ 📋 Német nyelv támogatás
    │ 📋 Lengyel nyelv támogatás
    │ 📋 Román nyelv támogatás
    │ 📋 EV diagnosztika bővítés
```

---

## 2. Koltsegvetes Reszletezese / Detailed Budget

### 2.1 Fejlesztési Költségek / Development Costs

#### Fázis 1 (Befejezve) - 2025 Q4 - 2026 Q1

| Tétel | Becsült óra | Óradíj (HUF) | Összeg (HUF) |
|-------|-------------|--------------|--------------|
| Backend fejlesztés | 400 | 15 000 | 6 000 000 |
| Frontend fejlesztés | 200 | 15 000 | 3 000 000 |
| DevOps & infrastruktúra | 100 | 15 000 | 1 500 000 |
| NLP/AI fejlesztés | 150 | 20 000 | 3 000 000 |
| Tesztelés & QA | 100 | 12 000 | 1 200 000 |
| **Összesen Fázis 1** | **950** | | **14 700 000** |

#### Fázis 2 (Folyamatban) - 2026 Q2

| Tétel | Becsült óra | Óradíj (HUF) | Összeg (HUF) |
|-------|-------------|--------------|--------------|
| Backend bővítések | 150 | 15 000 | 2 250 000 |
| Frontend optimalizáció | 100 | 15 000 | 1 500 000 |
| Magyar fordítások | 80 | 10 000 | 800 000 |
| Költségbecslő modul | 60 | 15 000 | 900 000 |
| Mobil UI fejlesztés | 80 | 15 000 | 1 200 000 |
| Integrációs tesztek | 60 | 12 000 | 720 000 |
| **Összesen Fázis 2** | **530** | | **7 370 000** |

#### Fázis 3 (Tervezett) - 2026 Q3-Q4

| Tétel | Becsült óra | Óradíj (HUF) | Összeg (HUF) |
|-------|-------------|--------------|--------------|
| Multi-modal AI (fotó) | 200 | 20 000 | 4 000 000 |
| Hangfelismerés integ. | 120 | 18 000 | 2 160 000 |
| Prediktív ML modellek | 180 | 20 000 | 3 600 000 |
| B2B API fejlesztés | 100 | 15 000 | 1 500 000 |
| Enterprise funkciók | 80 | 15 000 | 1 200 000 |
| Dokumentáció & SDK | 60 | 12 000 | 720 000 |
| **Összesen Fázis 3** | **740** | | **13 180 000** |

#### Fázis 4 (Tervezett) - 2027

| Tétel | Becsült óra | Óradíj (HUF) | Összeg (HUF) |
|-------|-------------|--------------|--------------|
| Német lokalizáció | 150 | 12 000 | 1 800 000 |
| Lengyel lokalizáció | 150 | 12 000 | 1 800 000 |
| Román lokalizáció | 150 | 12 000 | 1 800 000 |
| EV diagnosztika modul | 200 | 18 000 | 3 600 000 |
| Skálázás & optim. | 100 | 15 000 | 1 500 000 |
| **Összesen Fázis 4** | **750** | | **10 500 000** |

### 2.2 Infrastruktúra Költségek / Infrastructure Costs

#### Havi Működési Költségek (V1 Célállapot)

| Szolgáltatás | Provider | Szint | Havi (USD) | Havi (HUF) | Éves (HUF) |
|--------------|----------|-------|------------|------------|------------|
| Backend hosting | Railway | Pro | $50 | 18 000 | 216 000 |
| Frontend hosting | Railway | Pro | $20 | 7 200 | 86 400 |
| PostgreSQL | Railway | Managed | $30 | 10 800 | 129 600 |
| Redis | Railway | Managed | $20 | 7 200 | 86 400 |
| Neo4j | Neo4j Aura | Professional | $65 | 23 400 | 280 800 |
| Qdrant | Qdrant Cloud | Professional | $35 | 12 600 | 151 200 |
| LLM API (Claude) | Anthropic | Enterprise | $200 | 72 000 | 864 000 |
| Domain & SSL | Cloudflare | Pro | $20 | 7 200 | 86 400 |
| Monitoring | Sentry | Team | $26 | 9 360 | 112 320 |
| CI/CD | GitHub | Pro | $4 | 1 440 | 17 280 |
| **Összesen** | | | **$470** | **169 200** | **2 030 400** |

#### Skálázott Költségek (1000+ felhasználó)

| Szolgáltatás | Szint | Havi (HUF) | Éves (HUF) |
|--------------|-------|------------|------------|
| Backend (3x instance) | Scale | 54 000 | 648 000 |
| PostgreSQL (16GB) | Scale | 36 000 | 432 000 |
| Neo4j Aura (dedicated) | Enterprise | 180 000 | 2 160 000 |
| Qdrant Cloud (8GB RAM) | Professional | 50 000 | 600 000 |
| LLM API (increased) | Enterprise | 300 000 | 3 600 000 |
| CDN & Edge | Cloudflare | Business | 30 000 | 360 000 |
| **Összesen** | | **650 000** | **7 800 000** |

### 2.3 Marketing és Értékesítés / Marketing & Sales

| Tétel | Q2 2026 | Q3 2026 | Q4 2026 | Éves |
|-------|---------|---------|---------|------|
| Content marketing | 300 000 | 400 000 | 500 000 | 1 200 000 |
| Social media hirdetés | 200 000 | 300 000 | 400 000 | 900 000 |
| Google Ads | 300 000 | 400 000 | 500 000 | 1 200 000 |
| Szerviz partnerségek | 100 000 | 200 000 | 200 000 | 500 000 |
| Konferenciák, események | 0 | 300 000 | 300 000 | 600 000 |
| PR & kommunikáció | 100 000 | 150 000 | 200 000 | 450 000 |
| **Összesen** | **1 000 000** | **1 750 000** | **2 100 000** | **4 850 000** |

### 2.4 Jogi és Adminisztráció / Legal & Admin

| Tétel | Összeg (HUF) | Gyakoriság |
|-------|--------------|------------|
| Cégalapítás / bővítés | 200 000 | Egyszeri |
| GDPR compliance audit | 500 000 | Éves |
| Szerződések, ÁSZF | 300 000 | Egyszeri |
| Könyvelés, adózás | 600 000 | Éves |
| Biztosítások | 200 000 | Éves |
| **Összesen** | **1 800 000** | - |

---

## 3. Osszesitett Koltsegvetes / Total Budget Summary

### 3.1 Teljes Projekt Költségvetés (2026)

| Kategória | Q1 2026 | Q2 2026 | Q3 2026 | Q4 2026 | Éves Összeg |
|-----------|---------|---------|---------|---------|-------------|
| Fejlesztés | 7 350 000 | 7 370 000 | 6 590 000 | 6 590 000 | 27 900 000 |
| Infrastruktúra | 507 600 | 507 600 | 507 600 | 507 600 | 2 030 400 |
| Marketing | 0 | 1 000 000 | 1 750 000 | 2 100 000 | 4 850 000 |
| Jogi & Admin | 500 000 | 400 000 | 450 000 | 450 000 | 1 800 000 |
| **Összesen** | **8 357 600** | **9 277 600** | **9 297 600** | **9 647 600** | **36 580 400** |

### 3.2 Pályázati Igény Lebontása

**Igényelt támogatás: 30 000 000 HUF**

| Kategória | Összeg (HUF) | % | Felhasználás |
|-----------|--------------|---|--------------|
| **Fejlesztési költségek** | 15 000 000 | 50% | Backend, frontend, AI/ML fejlesztés |
| **Infrastruktúra** | 6 000 000 | 20% | Cloud szolgáltatások 12 hónap |
| **Marketing & Sales** | 5 000 000 | 17% | Piaci bevezetés, ügyfélszerzés |
| **Jogi & Admin** | 2 000 000 | 7% | GDPR, szerződések, könyvelés |
| **Tartalék** | 2 000 000 | 7% | Kockázatkezelés |
| **Összesen** | **30 000 000** | **100%** | |

### 3.3 Önrész és Társfinanszírozás

| Forrás | Összeg (HUF) | % |
|--------|--------------|---|
| Saját tőke (befektetett munka) | 14 700 000 | 33% |
| Pályázati támogatás | 30 000 000 | 67% |
| **Projekt összköltség** | **44 700 000** | **100%** |

---

## 4. Eroforras Terv / Resource Plan

### 4.1 Csapat Összetétel / Team Composition

#### Jelenlegi Csapat

| Pozíció | FTE | Felelősség | Státusz |
|---------|-----|------------|---------|
| Lead Developer | 1.0 | Backend, DevOps, AI/ML | Aktív |

#### Szükséges Bővítés (V1 Launch-ig)

| Pozíció | FTE | Prioritás | Becsült bér (HUF/hó) |
|---------|-----|-----------|----------------------|
| Frontend Developer | 1.0 | Magas | 800 000 - 1 000 000 |
| Product Designer | 0.5 | Közepes | 400 000 - 500 000 |
| QA Engineer | 0.5 | Közepes | 350 000 - 450 000 |
| Marketing Specialist | 0.5 | Alacsony (Q2) | 350 000 - 450 000 |

#### Konzultánsok / Alvállalkozók

| Specializáció | Becsült óra | Óradíj (HUF) | Összeg |
|---------------|-------------|--------------|--------|
| NLP/Magyar nyelv szakértő | 40 | 25 000 | 1 000 000 |
| Security auditor | 20 | 30 000 | 600 000 |
| Legal / GDPR | 15 | 40 000 | 600 000 |
| UX/UI konzultáns | 30 | 20 000 | 600 000 |

### 4.2 Technológiai Erőforrások / Technical Resources

#### Fejlesztési Környezet

| Eszköz | Típus | Havi költség (HUF) |
|--------|-------|-------------------|
| JetBrains IDEs | Development | 15 000 |
| Figma | Design | 18 000 |
| Linear | Project mgmt | 10 000 |
| Notion | Documentation | 12 000 |
| 1Password Teams | Security | 8 000 |
| **Összesen** | | **63 000** |

#### Tesztelési Erőforrások

| Eszköz | Cél | Költség |
|--------|-----|---------|
| Playwright Cloud | E2E testing | $50/hó |
| Percy | Visual regression | $75/hó |
| k6 | Load testing | Ingyenes (OSS) |

### 4.3 Adatforrások / Data Sources

| Forrás | Típus | Költség | Státusz |
|--------|-------|---------|---------|
| NHTSA API | Ingyenes | $0 | Integrálva |
| OBDb GitHub | Open source | $0 | Integrálva |
| obd-trouble-codes | Open source | $0 | Integrálva |
| Back4App Vehicles | Ingyenes tier | $0 | Integrálva |
| CarMD API | Fizetős (opció) | $100/hó | Tervezett |
| AllData (OEM) | Fizetős (opció) | $300/hó | Tervezett |

---

## 5. Kockázatkezelés Penzugyi Szempontbol / Financial Risk Management

### 5.1 Azonosított Kockázatok

| Kockázat | Valószínűség | Pénzügyi hatás | Kezelési stratégia |
|----------|--------------|----------------|---------------------|
| LLM API költség emelkedés | Közepes | +50% LLM költség | Lokális modell fallback |
| Lassú piaci adopció | Alacsony | -30% bevétel | Freemium erősítése |
| Fejlesztési csúszás | Közepes | +20% fejl. költség | Agilis iterációk |
| Infrastruktúra skálázás | Alacsony | +100% infra | Horizontális skálázás |
| Versenytárs belépés | Közepes | -20% piacrész | Gyors feature dev |

### 5.2 Tartalékképzés

| Tartalék típus | Összeg (HUF) | Cél |
|----------------|--------------|-----|
| Fejlesztési tartalék | 1 000 000 | Csúszás fedezése |
| Infrastruktúra tartalék | 500 000 | Váratlan skálázás |
| Jogi tartalék | 300 000 | Compliance költségek |
| Általános tartalék | 200 000 | Egyéb |
| **Összesen** | **2 000 000** | |

---

## 6. Bevetel Elorejelzes / Revenue Forecast

### 6.1 Üzleti Modell Árazás

| Csomag | Havi díj (HUF) | Éves díj (HUF) | Célcsoport |
|--------|----------------|----------------|------------|
| **Free** | 0 | 0 | Kipróbálás, magánszemélyek |
| **Pro** | 2 990 | 29 900 | DIY szerelők, hobbisták |
| **Business** | 9 990 | 99 900 | Kis szervizek |
| **Enterprise** | Egyedi | Egyedi | Flották, láncok |
| **API** | Pay-per-use | - | Fejlesztők, integrációk |

### 6.2 Felhasználó Növekedési Terv

| Időszak | Free | Pro | Business | Enterprise | Összesen |
|---------|------|-----|----------|------------|----------|
| 2026 Q2 | 500 | 50 | 10 | 0 | 560 |
| 2026 Q3 | 2 000 | 200 | 30 | 2 | 2 232 |
| 2026 Q4 | 5 000 | 500 | 80 | 5 | 5 585 |
| 2027 Q1 | 10 000 | 1 000 | 150 | 10 | 11 160 |
| 2027 Q2 | 20 000 | 2 500 | 300 | 20 | 22 820 |

### 6.3 Bevételi Előrejelzés

| Időszak | Pro MRR | Business MRR | Enterprise | Össz. MRR | Össz. ARR |
|---------|---------|--------------|------------|-----------|-----------|
| 2026 Q2 | 149 500 | 99 900 | 0 | 249 400 | 2 992 800 |
| 2026 Q3 | 598 000 | 299 700 | 200 000 | 1 097 700 | 13 172 400 |
| 2026 Q4 | 1 495 000 | 799 200 | 500 000 | 2 794 200 | 33 530 400 |
| 2027 Q1 | 2 990 000 | 1 498 500 | 1 000 000 | 5 488 500 | 65 862 000 |
| 2027 Q2 | 7 475 000 | 2 997 000 | 2 000 000 | 12 472 000 | 149 664 000 |

### 6.4 Break-Even Elemzés

**Havi fix költségek (V1 után):** ~2 500 000 HUF

**Break-even pont:**
- Pro: 836 felhasználó, VAGY
- Business: 250 felhasználó, VAGY
- Kombinált: 300 Pro + 100 Business

**Előrejelzett break-even időpont:** 2026 Q4

---

## 7. Pénzügyi Mérföldkövek / Financial Milestones

| Mérföldkő | Időpont | Kritérium | Státusz |
|-----------|---------|-----------|---------|
| MVP kész | 2026 Q1 | Működő platform | ✅ |
| Első fizető felhasználó | 2026 Q2 | 1+ Pro előfizető | ⏳ |
| 100 fizető felhasználó | 2026 Q3 | 100 Pro/Business | 📋 |
| Break-even | 2026 Q4 | MRR > havi költség | 📋 |
| 1M HUF MRR | 2026 Q4 | Stabil növekedés | 📋 |
| 5M HUF MRR | 2027 Q1 | Skálázás előtt | 📋 |
| 10M HUF MRR | 2027 Q2 | Series A ready | 📋 |

---

*Dokumentum verzió: 1.0*
*Utolsó frissítés: 2026-02-08*
*Megjegyzés: Az összegek tájékoztató jellegűek, a tényleges költségek eltérhetnek.*
