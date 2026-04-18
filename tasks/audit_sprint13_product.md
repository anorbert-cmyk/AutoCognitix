# Sprint 13 Product/Vision Audit

## A) V1 Q2 2026 delta

| Cél | Vízió (Q2 2026) | Valóság (2026-04-18) | Gap |
|-----|------------------|----------------------|-----|
| Neo4j gráf csomópontok | 160,000+ | 26,816 | -83% (~133K hiány) |
| Qdrant vektor embedding | 170,000+ | ~35,000 | -79% (~135K hiány) |
| DTC kódok (HU fordítás) | 11,000+ @ 80% lefedettség | 63 DTC (`data/dtc_codes/`) | -99% |
| Jármű konfiguráció | 1,500+ európai piac | NHTSA-alapú (ismeretlen méret) | nagy, számszerűsítetlen |
| NHTSA panasz index | 146,000+ | `data/nhtsa/` mappa létezik, de a node-ok ~26K összesen | nagyságrendi hiány |
| Automata teszt | 200+ | 74 (backend+frontend fájl) | -63% |
| Cloud deploy | Railway + Neo4j Aura + Qdrant Cloud | Mind él | TELJESÍTVE |
| CI/CD + security scan | Full pipeline | `ci.yml`, `cd.yml`, `security.yml` él | TELJESÍTVE |

**Eredmény:** Infrastruktúra kész, **tartalmi/adatmennyiségi gap 80% körül**.

## B) Top 3 hiányzó feature

1. **Fizetési integráció (Stripe/SimplePay)** — Ígéret forrás: `PROJECT_OVERVIEW.md:354` "Cost Model: Subscription" + `PricingPage.tsx` élő árakkal (4.990 Ft / 14.990 Ft havonta). **Kód gap:** `grep stripe|SimplePay` → backend+frontend **0 találat**, csak `docs/COWORK_BRIEF.md` említi. A "Választom" gomb simán `/auth/register`-re navigál, nincs checkout flow, nincs payment webhook, nincs subscription model DB-ben.

2. **i18n / Angol lokalizáció** — Ígéret forrás: `PROJECT_OVERVIEW.md:10` "both Hungarian and English", Phase 4 bővítés (DE/PL/RO). **Kód gap:** `grep i18next|react-i18next|useTranslation` a `frontend/src`-ben → **0 találat**. A `PricingPage` teljesen hard-coded magyar szöveg. Nincs locale switcher, nincs translation JSON.

3. **Admin panel** — Ígéret forrás: Phase 3 "B2B enterprise features" (`PROJECT_OVERVIEW.md:399`), implicit az user/subscription management miatt. **Kód gap:** `grep admin` az `api/`-ban → csak `auth.py` user roles említés; nincs `admin.py` endpoint, nincs `/admin` route a frontend pages listában (BlogPage, CalculatorPage, stb. — admin page hiányzik).

## C) Üzleti modell

- **Tier-ek forrása:** 100%-ban **hard-coded React state** (`PricingPage.tsx:9-67` FeatureGroup tömbök, `:69-72` hard-coded árak: `STARTER_MONTHLY='4.990'`, `PRO_MONTHLY='14.990'`).
- **Backend subscription endpoint:** `grep -r "subscription" backend/app/api` → **nincs**. Csak `newsletter.py` tartalmaz véletlen match-et. Nincs `/api/v1/subscriptions`, nincs tier model, nincs user.plan mező.
- **CTA:** Mindkét terv gombja `window.location.href = '/auth/register'`-re megy (sor 157, 190). **Nincs Stripe/SimplePay checkout, nincs webhook, nincs billing oldal.**
- **Következmény:** A `/pricing` oldal jelenleg **marketing landing**, nem működő monetizációs flow. Revenue = 0 lehetőség a jelenlegi kódbázissal.

## Olvasott fájlok

- `/home/user/AutoCognitix/PROJECT_OVERVIEW.md` (részletek: 10-48, 340-409)
- `/home/user/AutoCognitix/frontend/src/pages/PricingPage.tsx` (teljes, 277 sor)
- `/home/user/AutoCognitix/CLAUDE.md` (context header: DB táblázat)
- `ls /home/user/AutoCognitix/data/` (6 alkönyvtár)
- `grep` hívások: stripe, i18next, admin, subscription (nem teljes fájl olvasások)
