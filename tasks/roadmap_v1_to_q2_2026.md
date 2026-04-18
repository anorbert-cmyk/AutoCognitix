# Roadmap: Jelenlegi állapot → V1 Q2 2026

**Dátum:** 2026-04-18 (11 hét van Q2 2026 végéig)
**Vízióforrás:** `PROJECT_OVERVIEW.md` "V1 Production Target"
**Tényadat-forrás:** kódbázis + scriptek + `data/` mappa + CLAUDE.md adatbázis státusz sor

Ez a dokumentum ÖSSZEVETI a deklarált V1 célokat a TÉNYLEGES kódállapottal, és vázolja a zárási feladatokat. A `tasks/audit_sprint13_master.md` szerinti findingek függőségben vannak ezzel.

---

## 1. V1 mérőszám delta (számszerű)

| # | Cél | Vízió (Q2 2026) | Jelenlegi (2026-04-18) | Delta | Forrás |
|---|------|-----------------|------------------------|-------|--------|
| V1-1 | Neo4j gráf csomópontok | **160,000+** | 26,816 | **-83% (~133K hiány)** | `CLAUDE.md` DB státusz sor |
| V1-2 | Qdrant vektor embedding | **170,000+** | ~35,000 | **-79% (~135K hiány)** | `CLAUDE.md` + `scripts/index_qdrant_*.py` |
| V1-3 | DTC kódok HU fordítással | **11,000+ @ 80% HU** | 63 DTC | **-99%** | `data/dtc_codes/` méret |
| V1-4 | Jármű konfigurációk | **1,500+ EU piac** | NHTSA-alapú, ismeretlen pontos szám | számszerűsítetlen, nagyságrendi hiány | `scripts/import_back4app_vehicles.py`, `seed_vehicles.py` |
| V1-5 | NHTSA panasz index | **146,000+** | ~26K node összes (gráfban benne van) | nagyságrendi hiány | `scripts/import_nhtsa_complaints.py` |
| V1-6 | Automata teszt | **200+** | 74 (51 backend + 19 frontend + 4 audit) | **-63% (~126 hiány)** | `backend/tests/`, `frontend/src/**/*.test.*` |
| V1-7 | Cloud deploy | Railway + Neo4j Aura + Qdrant Cloud | **Mind él** | **✅ TELJESÍTVE** | `railway.toml`, `.env.railway.example` |
| V1-8 | CI/CD + security scan | Full pipeline | **Él** | **✅ TELJESÍTVE** | `.github/workflows/{ci,cd,security}.yml` |

**Összegzés:** Infrastruktúra-szinten 100%, de **tartalmi/adatmennyiség 20-25%** a célhoz képest.

---

## 2. Feature-szintű gap

### 2.1 ✅ Megvalósítva
- Hardver nélküli manuális DTC/tünet bevitel
- huBERT embedding + Qdrant semantic search
- LangChain RAG + Anthropic/OpenAI LLM
- VIN dekódolás (NHTSA)
- Jármű garage + maintenance reminder + health score
- NHTSA visszahívás badge (ResultPage) + Leaflet service map
- Demo P0300 szimuláció valós alkatrész-árakkal
- JWT auth + rate limiting + GDPR delete
- Strukturált JSON log + Prometheus /metrics

### 2.2 🟡 Részleges
- **Magyar fordítás** — 63 DTC van, 11,000+ kell → scriptek (`translate_to_hungarian.py`) léteznek, csak nagy LLM-budget kell a tömeges futtatáshoz
- **Sentry** — SDK wired mindkét oldalon, DE `capture_exception()` sehol nem hívott (`backend/app/core/error_handlers.py`, `frontend/src/components/ErrorBoundary.tsx`), DSN nincs dokumentálva Railway env-ben
- **Teszt lefedettség** — 74/200: backend-oldalon kb. 30 audit-teszt kellene még, frontend-oldalon Playwright E2E (ma 0)
- **Email auth flow** — Sprint 12 tervezi, password reset endpoint él, de email send-wiring nem teljes

### 2.3 ❌ Teljesen hiányzik (víziós célpont, kód nyom = 0)
- **Fizetési integráció** (Stripe/SimplePay) — `PricingPage.tsx` hard-coded tier-ek, `grep stripe|SimplePay` = 0. A "Választom" gomb simán `/auth/register`-re megy → **revenue-mentes rendszer**
- **Subscription backend** — nincs `Subscription` modell, nincs `user.plan` mező, nincs `/api/v1/subscriptions/*` endpoint
- **i18n / angol lokalizáció** — vízió "both Hungarian and English", de `grep i18next|react-i18next` = 0. Minden hard-coded magyar
- **Admin panel** — nincs `/admin` route, nincs `admin.py` endpoint, nincs superuser UI
- **Audit log** — **GDPR Art. 30/32 megsértés**: 0 audit tábla, 0 migration (ld. `audit_sprint13_master.md` C1)
- **PWA / offline** — nincs manifest.json, nincs service worker
- **Mobile app** — React Native / NativeScript nincs
- **CarAPI / CarMD** fizetős API integráció — CLAUDE.md "később"-nek jelöli, stub sincs

---

## 3. 11 hét cselekvési terv (Sprint 13-18)

> A Q2 2026 végéig ~11 hét van. Ez 5-6 kéthetes sprint.

### Sprint 13 — Biztonság + infra (2 hét)
- C1: Audit log modell + migration + emit az auth/garage/diagnosis endpoint-okról
- H1-H3: JWT iss/aud, login rate limit, refresh blacklist
- H4-H5: huBERT `revision=` pin + embedding cache key verzió
- H6: RTO/RPO doc + heti restore smoke-test staging-en
- M1-M4: Alembic drift fix (013, 016)
- M6: Sentry `capture_exception()` wiring

### Sprint 14 — Monetization + i18n (2 hét)
- H7: Stripe checkout + `Subscription` modell + webhook + billing oldal
- H8: `react-i18next` + HU/EN translation JSON (core oldalak)
- H9: Admin panel v1 (user list, subscription management)

### Sprint 15 — Adatfeltöltés PUSH #1 (2 hét)
- Tömeges HU fordítás (`translate_to_hungarian.py` + `fix_translations.py`): 63 → 3,000 DTC
- Neo4j seed bővítés: 26,816 → 80,000 node (OBDb + scrape)
- Qdrant újraindexelés új huBERT revízió-val: 35,000 → 100,000 vector

### Sprint 16 — Adatfeltöltés PUSH #2 (2 hét)
- DTC: 3,000 → 11,000 (vízió cél)
- Neo4j: 80,000 → 160,000
- Qdrant: 100,000 → 170,000
- Vehicle configs: 1,500 EU
- NHTSA complaints: 146,000 index

### Sprint 17 — Teszt + minőség (2 hét)
- Playwright E2E (5 kritikus flow): diagnosis, demo, garage, auth, pricing
- Backend audit teszt 10 új (Sprint 13-16 javításokhoz)
- Teszt konszolidáció: `api/` + `integration/` + `e2e/` összeolvasztás
- 74 → 200+ teszt

### Sprint 18 — Launch readiness (1 hét)
- Load test (k6/locust): p99 < 2s diagnosis
- Security sweep: CodeQL, Bandit, npm audit clean
- RTO/RPO drill live
- Prod DSN + alerting end-to-end
- Beta launch

---

## 4. Rizikók + feltételezések

| Rizikó | Hatás | Mitigálás |
|--------|-------|-----------|
| Tömeges LLM-fordítás költsége (11,000 DTC × ~200 token ≈ $50-200) | Budget feszes | Groq / local LLM (Qwen) fallback, caching |
| huBERT modell revízió változik | 35k+ vektor újraindexelése kell | Sprint 13 H4/H5 megoldja; 170k újraindexelése ~8 GPU-óra |
| Stripe integráció HU KKV compliance (ÁFA, számlázás) | Blokkoló | SimplePay alternatíva KKV tier-hez, Stripe marad B2B |
| Playwright CI időtartam | CI >10 perc | Parallel worker + csak main branch-en e2e |
| Neo4j Aura Free tier node limit | Blokkoló 160k node-nál | Paid tier ($65/hó) vagy self-hosted Railway Neo4j |

---

## 5. Q2 2026 go/no-go kritériumok

**Go, ha:**
- [ ] Minden V1 mérőszám ≥ 90% cél
- [ ] CRITICAL + HIGH audit találatok 100% javítva
- [ ] Revenue flow működik (Stripe end-to-end)
- [ ] 200+ teszt, CI zöld minden check-en
- [ ] RTO < 4h, RPO < 24h bizonyítottan

**No-go, ha:**
- Adatmennyiség < 60% (Neo4j < 100k, Qdrant < 100k)
- Revenue flow nem él
- CRITICAL finding nyitott

---

## 6. Kapcsolódó dokumentumok

- `tasks/audit_sprint13_master.md` — részletes finding lista
- 7 specialist audit: `tasks/audit_sprint13_{security,database,performance,observability,quality,data,product}.md`
- `docs/ONBOARDING.md`, `docs/ARCHITECTURE.md`, `docs/DATA_FLOW.md`, `docs/DATABASE_MAP.md` — új tagok belépési csomagja
- `CLAUDE.md` § Post-Sprint Review Protocol
- `PROJECT_OVERVIEW.md` § V1 Production Target
