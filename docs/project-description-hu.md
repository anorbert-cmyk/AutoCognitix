# AutoCognitix — Projektleírás

## 1. Platform Áttekintés

### Cél
Az **AutoCognitix** egy mesterséges intelligenciával működő gépjármű-diagnosztikai platform, amely magyar nyelven nyújt segítséget járműtulajdonosoknak és szerelőknek. A rendszer hardver nélkül, manuális DTC hibakód- és tünetbevitellel működik, AI-alapú elemzéssel, alkatrészárakkal és javítási tervvel.

### Célcsoport
- Járműtulajdonosok, akik szeretnék megérteni autójuk hibakódjait
- Autószerelők, akik diagnosztikai támogatást keresnek
- Szervizek, amelyek hatékonyabb ügyfélkommunikációt szeretnének

### Technológiai Stack

| Réteg | Technológia |
|-------|-------------|
| **Backend** | FastAPI, Python 3.9, Pydantic V2, SQLAlchemy 2.0 async |
| **Frontend** | React 18, TypeScript, TailwindCSS, Vite |
| **Állapotkezelés** | TanStack React Query v5 |
| **Adatbázisok** | PostgreSQL 16, Neo4j 5.x (gráf), Qdrant (vektor), Redis (cache) |
| **AI/NLP** | HuBERT (768-dim magyar embedding), LangChain RAG, Groq/Anthropic LLM |
| **Ikonok** | lucide-react, Material Symbols |
| **Deployment** | Railway (backend + frontend), Neo4j Aura, Qdrant Cloud |

### Adatbázis Tartalom
- **Neo4j**: 26,816 node (járművek, DTC kódok, panaszok, visszahívások)
- **Qdrant**: 35,000+ vektor (HuBERT 768-dim embeddings)
- **PostgreSQL**: 3,716 DTC kód, felhasználók, diagnosztikai munkamenetek

---

## 2. Funkciók Részletes Leírása

### 2.1 AI Diagnosztika (`/diagnosis`)
**Fő funkció** — DTC hibakódok és tünetek bevitele → AI-alapú elemzés → részletes javítási terv.

- Jármű kiválasztás: márka, modell, évjárat (NHTSA adatbázisból)
- DTC kód bevitel (pl. P0300, P0420) + szabad szöveges tünetleírás
- Magyar nyelvű beszédfelismerés (Web Speech API, hu-HU)
- SSE streaming válasz (token-by-token megjelenítés)
- Eredmény: gyökérok elemzés, javítási lépések, alkatrészárak, konfidencia pontszám

### 2.2 Diagnosztikai Jelentés (`/diagnosis/:id`)
- Részletes AI-elemzés megjelenítése (navy téma, Space Grotesk font)
- Javítási terv priorizált lépésekkel, szerszámokkal, szakértői tippekkel
- Alkatrészek és árak táblázat (Bárdi Autó, Uni Autó, AUTODOC árak)
- Összköltség becslés (alkatrész + munkadíj)
- Cross-linkek: Kalkulátor, Chat, Műszaki vizsga, Szerviz keresés
- PDF/Nyomtatás funkció

### 2.3 DTC Kereső (`/dtc/:code`)
- 3,716 DTC kód keresése és részletes leírása
- Kategóriák: Powertrain (P), Chassis (C), Body (B), Network (U)
- Kapcsolódó visszahívások és panaszok megjelenítése
- Debounced kereső mező valós idejű találatokkal

### 2.4 VIN Dekóder (`/vehicles`)
- 17 karakteres VIN (alvázszám) dekódolás NHTSA API-val
- Jármű specifikációk: márka, modell, évjárat, motor, hajtás
- Visszahívások és panaszok automatikus lekérése
- Márkák és modellek böngészése

### 2.5 Demo Oldal (`/demo`)
- Előre kitöltött P0300 + P0301 + P0304 szimuláció
- Jármű: VW Golf VII 1.4 TSI (2018), 98,420 km
- Valós 2026 márciusi alkatrészárak 3 bolttól
- 6 alkatrész kártyás megjelenítéssel

### 2.6 Műszaki Vizsga Felkészítő (`/inspection`) ⭐ ÚJ
- DTC kódok → magyar MOT vizsgakategóriákra történő leképezés
- 10 vizsgakategória: emisszió, fékrendszer, világítás, futómű, stb.
- Kockázati szint: Magas / Közepes / Alacsony (kör alakú gauge)
- Kategóriánkénti MEGFELELT / NEM FELELT MEG / FIGYELMEZTETÉS
- Becsült javítási költség összesítéssel
- Magyar nyelvű ajánlások

**API:** `POST /api/v1/inspection/evaluate`

### 2.7 "Megéri megjavítani?" Kalkulátor (`/calculator`) ⭐ ÚJ
- Jármű értékbecslés magyar piaci adatok alapján
- Amortizációs modell: standard + prémium márkák (BMW, Audi, Mercedes)
- Kilométerállás korrekció (15,000 km/év magyar átlag)
- Állapot szorzók: kiváló (1.15x), jó (1.0x), elfogadható (0.88x), gyenge (0.70x)
- Javítási költség / jármű érték arány → ajánlás: Javítás / Eladás / Roncs
- Alternatív forgatókönyvek: eladás jelenlegi állapotban, javítás utáni eladás, roncsként
- Diagnosztikából átvehető javítási költség (`diagnosis_id`)

**API:** `POST /api/v1/calculator/evaluate`

### 2.8 AI Chat Asszisztens (`/chat`) ⭐ ÚJ
- Valós idejű SSE streaming magyar nyelven
- Magyar autószerelő AI persona
- Járműkontextus-tudatos válaszok (márka, modell, DTC kódok)
- RAG-alapú tudáslekérdezés (Qdrant vektor keresés)
- Követő kérdés javaslatok (3 gomb minden válasz után)
- Beszélgetés előzmény (max 10 üzenet)
- Prompt injection védelem

**API:** `POST /api/v1/chat/message` (SSE StreamingResponse)

### 2.9 Szerviz Összehasonlítás (`/services`) ⭐ ÚJ
- 30 valós magyar autószerviz országszerte
- Régió szűrő: Budapest + 19 megye
- Márka szűrő: elfogadott márkák alapján
- Szolgáltatás típus szűrő (általános, diagnosztika, karosszéria, stb.)
- Rendezés: értékelés, ár, távolság, név
- Szerviz kártyák: értékelés csillagokkal, árszint (€/€€/€€€), szolgáltatások
- Műszaki vizsga jelvény
- Térképes megjelenítés (Leaflet — placeholder)

**API:** `GET /api/v1/services/search`, `GET /api/v1/services/regions`, `GET /api/v1/services/{id}`

---

## 3. Felhasználói Folyamatok

### 3.1 Alap Diagnosztikai Folyamat
1. Felhasználó megnyitja a `/diagnosis` oldalt
2. Kiválasztja a járművet (márka → modell → évjárat)
3. Megadja a DTC kódokat és/vagy tüneteket (gépelés vagy beszéd)
4. "Elemzés indítása" gomb → streaming AI válasz
5. Átirányítás a `/diagnosis/:id` jelentés oldalra
6. Opcionálisan: Kalkulátor / Chat / Műszaki vizsga / Szerviz keresés

### 3.2 Műszaki Vizsgára Készülés
1. `/inspection` → Jármű adatok + DTC kódok megadása
2. "Értékelés" → Kockázati gauge + kategória kártyák
3. Piros (FAIL) kategóriáknál → "Szerviz keresése" gomb

### 3.3 Javítási Döntés
1. Diagnosztika eredmény → "Megéri megjavítani?" gomb
2. `/calculator` → Jármű + állapot + kilométer megadása
3. Automatikus értékbecslés + javítási költség összehasonlítás
4. Ajánlás megjelenítése + alternatív forgatókönyvek

### 3.4 AI Konzultáció
1. Bármely oldalról → `/chat` (opcionálisan járműkontextussal)
2. Kérdés beírása → streaming AI válasz
3. Követő kérdés javaslatok → kattintás → új válasz

---

## 4. Navigációs Struktúra (Sitemap)

```
/                       Főoldal (feature kártyák)
├── /diagnosis          AI Diagnosztika (beviteli űrlap)
│   └── /diagnosis/:id  Diagnosztikai Jelentés (eredmény)
├── /demo               Demo Oldal (P0300 szimuláció)
├── /inspection         Műszaki Vizsga Felkészítő
├── /calculator         "Megéri megjavítani?" Kalkulátor
├── /chat               AI Chat Asszisztens
├── /services           Szerviz Összehasonlítás
├── /dtc/:code          DTC Kód Részletek
├── /vehicles           VIN Dekóder / Járműkeresés
├── /history            Diagnosztikai Előzmények
├── /login              Bejelentkezés
├── /register           Regisztráció
├── /forgot-password    Elfelejtett jelszó
├── /reset-password     Jelszó visszaállítás
└── /*                  404 - Nem Található
```

### Navigációs Menü Elemek
| Ikon | Címke | Útvonal |
|------|-------|---------|
| Stethoscope | Diagnózis | `/diagnosis` |
| Search | DTC Keresés | `/dtc` |
| Car | Járművek | `/vehicles` |
| ClipboardCheck | Műszaki Vizsga | `/inspection` |
| Calculator | Kalkulátor | `/calculator` |
| MessageSquare | AI Chat | `/chat` |
| MapPin | Szervizek | `/services` |

---

## 5. Design Irányelvek

### 5.1 Színpaletta

| Szín | Kód | Használat |
|------|-----|-----------|
| **Navy (primary)** | `#0D1B2A` | Header, hero szekciók, CTA gombok, AI elemzés háttér |
| **Blue-600** | `#2563EB` | Akcentus, linkek, aktív állapotok, focus ring |
| **Slate-50** | `#F8FAFC` | Oldal háttér |
| **Slate-900** | `#0F172A` | Elsődleges szöveg |
| **Slate-500** | `#64748B` | Másodlagos szöveg |
| **Green-500** | `#22C55E` | Sikeres/alacsony kockázat |
| **Yellow-500** | `#EAB308` | Figyelmeztetés/közepes kockázat |
| **Red-500** | `#EF4444` | Hiba/magas kockázat |
| **Emerald-700** | `#047857` | Ár jelzés |
| **Amber-800** | `#92400E` | Specializáció badge-ek |

### 5.2 Tipográfia

| Betűtípus | Használat |
|-----------|-----------|
| **Space Grotesk** | Fejlécek (h1-h3), DTC kódok, nagy számok |
| **Noto Sans** | Fő szövegtörzs, bekezdések |
| **Inter** | UI elemek, gombok, badge-ek, menü |
| **Monospace (font-mono)** | VIN, DTC kódok, rendszám |

### 5.3 Ikonok
- **Elsődleges:** lucide-react (Stethoscope, Search, Car, Wrench, MapPin, Calculator, MessageSquare, ClipboardCheck, Shield, Phone, stb.)
- **Másodlagos:** Material Symbols (build_circle, psychology, assignment, warning, payments, print, stb.)

### 5.4 Lekerekítések és Árnyékok

| Elem | Stílus |
|------|--------|
| Kártyák | `rounded-2xl border border-slate-200 shadow-sm` |
| Gombok | `rounded-lg` vagy `rounded-xl` |
| Badge-ek | `rounded-full` vagy `rounded-md` |
| Input mezők | `rounded-lg border border-slate-300` |
| Modálok | `rounded-3xl shadow-xl` |
| Hero szekciók | `rounded-3xl` + blur háttér |

---

## 6. Responsive Breakpoints

| Breakpoint | Szélesség | Használat |
|------------|-----------|-----------|
| **sm** | 640px | Mobil tájkép |
| **md** | 768px | Tablet |
| **lg** | 1024px | Laptop |
| **xl** | 1280px | Desktop |
| **2xl** | 1536px | Nagy képernyő |

### Layout Viselkedés
- **Főoldal grid:** `grid-cols-1` → `md:grid-cols-2` → `lg:grid-cols-4`
- **Szerviz oldal:** Mobil: stack (szűrők → lista → térkép) | Desktop: `lg:flex-row` (sidebar 400px + térkép flex-1)
- **Chat oldal:** Teljes magasság `calc(100vh - 64px)`, flexbox column
- **Diagnosztikai jelentés:** `grid-cols-1` → `lg:grid-cols-12` (4+8 arány)
- **Max szélesség:** `max-w-screen-2xl` (1536px) középre igazítva

---

## 7. Komponens Könyvtár

### UI Alapkomponensek

| Komponens | Fájl | Leírás |
|-----------|------|--------|
| `MaterialIcon` | `components/ui/MaterialIcon.tsx` | Material Symbols ikon wrapper |
| `SectionErrorBoundary` | `components/SectionErrorBoundary.tsx` | Szekció szintű hibahatár |
| `AIDisclaimerBadge` | `components/features/diagnosis/AIDisclaimerBadge.tsx` | GDPR/EU AI Act nyilatkozat |
| `DiagnosticConfidence` | `components/features/diagnosis/DiagnosticConfidence.tsx` | Kör alakú konfidencia mutató |
| `RepairStep` | `components/features/diagnosis/RepairStep.tsx` | Javítási lépés kártya |
| `PartStoreCard` | `components/features/diagnosis/PartStoreCard.tsx` | Bolt-specifikus ár kártya |

### Feature Komponensek

| Funkció | Komponensek |
|---------|-------------|
| **Inspection** | `RiskGauge` (kör gauge), `InspectionCategoryCard` (kategória kártya) |
| **Calculator** | `ValueComparison` (érték sáv), `RecommendationCard` (ajánlás badge), `ConditionSlider` (állapot választó) |
| **Chat** | `ChatWindow` (üzenet konténer), `MessageBubble` (buborék), `ChatInput` (beviteli mező), `SuggestionChips` (javaslat gombok), `TypingIndicator` (gépelés animáció) |
| **Services** | `RegionSelector` (régió választó), `ShopCard` (szerviz kártya), `ShopFilters` (szűrők), `RatingStars` (csillagok) |

### Toast Rendszer
- `useToast()` hook: `toast.success()`, `toast.error()`, `toast.info()`, `toast.warning()`
- Jobb felső sarok, automatikus eltűnés 5 másodperc után

---

## 8. API Végpont Lista

### Meglévő Végpontok

| Metódus | Útvonal | Leírás |
|---------|---------|--------|
| `POST` | `/api/v1/diagnosis/analyze` | AI diagnosztika (SSE streaming) |
| `GET` | `/api/v1/diagnosis/{id}` | Diagnózis részletek |
| `GET` | `/api/v1/diagnosis/history` | Diagnosztikai előzmények |
| `DELETE` | `/api/v1/diagnosis/{id}` | Diagnózis törlése |
| `GET` | `/api/v1/dtc/search?q=` | DTC kód keresés |
| `GET` | `/api/v1/dtc/{code}` | DTC kód részletek |
| `GET` | `/api/v1/dtc/categories` | DTC kategóriák |
| `POST` | `/api/v1/vehicles/decode-vin` | VIN dekódolás |
| `GET` | `/api/v1/vehicles/years` | Elérhető évjáratok |
| `GET` | `/api/v1/vehicles/makes?year=` | Márkák évjárat szerint |
| `GET` | `/api/v1/vehicles/models?make=&year=` | Modellek |
| `GET` | `/api/v1/vehicles/recalls?make=&model=&year=` | Visszahívások |
| `GET` | `/api/v1/vehicles/complaints?make=&model=&year=` | Panaszok |
| `POST` | `/api/v1/auth/login` | Bejelentkezés (JWT) |
| `POST` | `/api/v1/auth/register` | Regisztráció |
| `GET` | `/api/v1/auth/me` | Aktuális felhasználó |

### Új Végpontok (Sprint 11)

| Metódus | Útvonal | Leírás |
|---------|---------|--------|
| `POST` | `/api/v1/inspection/evaluate` | Műszaki vizsga kockázatértékelés |
| `POST` | `/api/v1/calculator/evaluate` | Javítási kalkulátor |
| `POST` | `/api/v1/chat/message` | AI Chat (SSE streaming) |
| `GET` | `/api/v1/services/search` | Szerviz keresés (szűrőkkel) |
| `GET` | `/api/v1/services/regions` | Elérhető régiók |
| `GET` | `/api/v1/services/{id}` | Szerviz részletek |

### Hitelesítés
- JWT Bearer token (`Authorization: Bearer <token>`)
- Opcionális auth a legtöbb végponton (anonim hozzáférés rate-limittel)
- Rate limiting: 20/óra (szerviz, vizsga), 10/óra (kalkulátor), 5 üzenet (anonim chat)

---

## 9. Környezeti Változók

```env
# Backend
DATABASE_URL=postgresql+asyncpg://...
NEO4J_URI=neo4j+s://...
QDRANT_URL=https://...
REDIS_URL=redis://...
JWT_SECRET_KEY=...
ANTHROPIC_API_KEY=...

# Frontend
VITE_API_URL=http://localhost:8000
```

---

## 10. Fejlesztői Parancsok

```bash
# Backend indítása
cd backend && uvicorn app.main:app --reload

# Frontend indítása
cd frontend && npm run dev

# Lint ellenőrzés
cd backend && python3 -m ruff check app/
cd frontend && npx tsc --noEmit

# Build
cd frontend && npm run build

# Tesztek
cd frontend && npx vitest run
cd backend && pytest tests/ -v
```
