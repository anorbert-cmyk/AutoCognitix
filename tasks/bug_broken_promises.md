# Broken Promises Audit

Forrás: kód-vs-UI/docs ellentmondás vizsgálat (8 fájl olvasással).

## A) VW Golf hard-coding (demo univerzalitás)
- **severity:** HIGH (megtévesztő marketing, nem hibás kód)
- **ígéret (UI):** `/demo` címen "Bemutató mód – Így néz ki egy teljes diagnosztikai jelentés" + általános "Saját diagnózis indítása" CTA. CLAUDE.md a `/demo`-t mint általános "demo bemutató oldalt" hirdeti. A user joggal hiheti, hogy bármely jármű + DTC ugyanolyan részletességű, kártyás, bolt-specifikus eredményt kap.
- **valóság:**
  - `frontend/src/pages/DemoResultPage.tsx:14-18` — az egész oldal **statikus import**: `demoDiagnosisResponse, demoParts, demoVehicleImage, demoVehicleDetails` egyetlen fájlból. Nincs request, nincs paraméter.
  - `frontend/src/pages/DemoResultPage.tsx:37` — banner hard-coded szöveg: "Szimulált P0300 hibakód · VW Golf VII 1.4 TSI · Valós alkatrész árak".
  - `frontend/src/data/demoData.ts:42-92` — minden alkatrész **kézzel beírt** Bárdi/Unix/AUTODOC árakkal, OEM számokkal (`04E 905 612 C`), `compatibilityNote: 'VW Golf VII 1.4 TSI (CZCA/CMBA motor) 2012-2020'`. NEM dinamikus.
  - `backend/app/services/parts_price_service.py:42-150` — `STATIC_PARTS_PRICES` általános kategóriák ("spark_plug", "ignition_coil") **nem márka/modell specifikus**.
  - `parts_price_service.py:446-455` — egyetlen "vehicle-specific" logika: nyers `price_multiplier` (BMW=1.5, VW=1.0, Dacia=0.85). Nincs OEM szám, nincs bolt link, nincs készletinfó. A demo szintű kártyás megjelenítést a backend **nem tudja reprodukálni** valódi user inputra.
- **következmény:** A `/demo` cégér mögött **Golf-only kézi adat**. Valódi diagnózisra (`/diagnosis` flow) a user **lényegesen szegényesebb** PartWithPrice választ kap (statikus tábla × shop-független multiplier), bolt logók és OEM számok nélkül.

## B) NHTSA recalls magyar/EU járművekre
- **severity:** CRITICAL (alaptermék-funkció EU piacon nem működik, mégis hirdetjük)
- **ígéret (docs/UI):** CLAUDE.md "Aktuális Adatbázis Állapot": **Recalls** Neo4j-ben + Sprint 10 "NHTSA visszahívás badge ResultPage-en". API tábla: `GET /api/v1/garage/vehicles/{id}/recalls` "Jármű NHTSA visszahívásai". UI badge piros kártya. A magyar célközönségnek (lásd projekt cél: "magyar nyelvtámogatással") egyetlen szó sem szól arról, hogy ez **csak US piaci járművekre** ad eredményt.
- **valóság:**
  - `backend/app/services/nhtsa_service.py:256` — `RECALLS_BASE_URL = "https://api.nhtsa.gov/recalls"` — **U.S. National Highway Traffic Safety Administration**.
  - `nhtsa_service.py:521-526` — `recallsByVehicle` endpoint csak `make/model/modelYear` paramétert vesz, US flotta adat. EU-specifikus piacon eladott Golf/Skoda/Opel recall **nem szerepel**.
  - `nhtsa_service.py:563-567` — hibakezelés csak újra-emelést végez (`raise`). **Nincs EU-fallback** (RAPEX, KBA, EU Safety Gate). Ha üres listát ad: silent — a user azt látja, hogy "nincs visszahívás", pedig csak rossz forrás.
  - `nhtsa_service.py:53,453` — a `plant_country` mezőt dekódoljuk, de **nem használjuk** EU vs US route döntésre.
- **következmény:** Egy magyar Opel Astra tulajdonos a `/vehicles/{id}/recalls` lapon hamis biztonságérzetet kap ("nincs visszahívás"), miközben német KBA visszahívás létezhet. Termékkockázat: biztonsági ígéret, ami nem teljesül.

## C) HuBERT RAG fallback átláthatóság
- **severity:** MEDIUM (mechanika OK, de a user-felé kommunikáció hiányos)
- **ígéret:** CLAUDE.md: "HuBERT embedding ... RAG alapja: A diagnosztikai AI innen keres releváns információt." A user feltételezi: panasz → HuBERT → Qdrant → DTC → válasz.
- **valóság:**
  - `backend/app/services/diagnosis_service.py:538` — `from app.services.rag_service import diagnose` lazy import a try blokkban.
  - `diagnosis_service.py:643-649` — `except ImportError: ... return self._fallback_diagnosis(...)`. RAG service hiányakor csendben rollback.
  - `diagnosis_service.py:650-656` — `except Exception as e: ... _fallback_diagnosis(...)`. **Bármi** rosszul megy (Qdrant timeout, HuBERT model load, LLM hiba) → fallback, log warning szinten.
  - `diagnosis_service.py:731-840` — `_fallback_diagnosis`: csak a Neo4j DTC `possible_causes`-ből és NHTSA-ból barkácsolja össze a választ. **HuBERT/Qdrant nem fut.** `model_used: "fallback"`, `used_fallback: True` jelölve a backend response-ban.
  - `frontend/src/services/api.ts:315` — `used_fallback?: boolean` opcionálisan szerepel a typeban, **de** `grep -rn "used_fallback" frontend/src` egyetlen találata ez a deklaráció. **Nincs UI komponens, ami a usernek jelezné**, hogy "RAG offline, csökkentett minőségű elemzés".
- **következmény:** A user ugyanazt a "AI diagnosztikai jelentést" látja Qdrant-down esetén is, csak halkabb tartalommal — anélkül, hogy figyelmeztetést kapna. Megsérül a transzparencia-ígéret, és potenciálisan a fizetős user félrevezetése.

## Olvasott fájlok
- `frontend/src/pages/DemoResultPage.tsx` (1-100)
- `frontend/src/data/demoData.ts` (1-100)
- `backend/app/services/parts_price_service.py` (1-150, 440-525)
- `backend/app/services/nhtsa_service.py` (1-150, 484-567)
- `backend/app/services/diagnosis_service.py` (500-660, 731-840)
- `frontend/src/services/api.ts` (grep used_fallback)

## TL;DR top 3
1. **NHTSA US-only**, magyar/EU járművekre néma null-eredmény — `nhtsa_service.py:256` — CRITICAL.
2. **/demo statikus Golf-paste**, nem reprezentálja a valódi pipeline kimenetét — `demoData.ts:42` + `DemoResultPage.tsx:14` — HIGH.
3. **RAG-fallback rejtett**, user nem értesül a degradált válaszról — `diagnosis_service.py:643-656` + `api.ts:315` — MEDIUM.
