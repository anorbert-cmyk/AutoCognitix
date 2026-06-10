# Pre-Push Audit — Operational/Observability Lead

Branch: `claude/code-review-fixes` vs `main` · Dátum: 2026-06-10 · Verdikt: **NINCS CRITICAL/HIGH — push engedélyezett** (3 MEDIUM TODO-val)

## A) cd.yml rollback logika — 4 forgatókönyv végigjátszva

YAML szintaxis: **valid** (PyYAML parse OK). `if: >-` folded multiline helyes.

| # | Forgatókönyv | needs eredmények | Rollback fut? | Issue ág |
|---|---|---|---|---|
| 1 | build-backend fail | migrations/deploy/smoke = skipped | ✅ IGEN (`always()` + `build-backend.result == 'failure'`) | "Build Failed" ✅ |
| 2 | run-migrations fail | deploy/smoke = skipped | ✅ IGEN | "Migration Failed — Deployment Blocked" ✅ (build success → első ág kihagyva) |
| 3 | migrations OK + deploy fail | smoke = skipped | ✅ IGEN | "Deploy Failed AFTER Successful Migration" + `schema-drift` label ✅ |
| 4 | minden OK + smoke fail | smoke `continue-on-error: true` → result = `success` | ❌ NEM (szándékos, smoke nincs az if-ben) | — |

`skipped ≠ failure` helyesen kezelve: skipped job result `'skipped'`, nem triggerel. A `needs`-be a run-migrations és deploy-railway felvétele helyes (enélkül a result nem olvasható). Az utolsó `else` ág ("Deploy Failed", deploy fail + migrations ≠ success) gyakorlatban elérhetetlen (deploy needs migrations) — ártalmatlan holt ág.

### Találatok

- **MEDIUM** `.github/workflows/cd.yml:362` — A rollback jobnak **nincs `permissions: issues: write`** deklarációja. A build jobok explicit permissions-t adnak (73-75, 135-137), a rollback a repo default token jogot örökli. Ha a repo "Workflow permissions" read-only, az `issues.create` 403-at kap, és a step `continue-on-error: true` (452) **némán elnyeli** — a schema-drift riasztás (az EGYETLEN jelzés) csendben elveszik. Fix: `permissions: { issues: write }` a rollback jobra.
- **MEDIUM** `.github/workflows/cd.yml:270` — Smoke-tests job-szintű `continue-on-error: true` + minden step `|| echo` → a smoke result MINDIG `success`; a notify táblázat (340) is success-t mutat valódi hibánál. Smoke hiba teljesen láthatatlan. (Korai fejlesztésben elfogadott, de dokumentálandó.)
- **LOW** `.github/workflows/cd.yml:371` — rollback `environment:` használ: ha az environmenthez később protection rule / required reviewer kerül, a riasztó job approvalra várna → késleltetett alert.

## B) Sentry coverage a konszolidáció után

`LoggingIntegration(level=INFO, event_level=ERROR)` **változatlan** (`backend/app/core/logging.py:665-668`). Új `before_send=_sentry_before_send` (671): PII redakció try/except-tel, **soha nem dobja el az eventet** ✅.

5xx útvonalak — mind a 6 logol `exc_info=True`-val (= LoggingIntegration teljes exception event stack trace-szel):

| Handler | Hely | exc_info |
|---|---|---|
| autocognitix 5xx | `error_handlers.py:157-161` | ✅ |
| sqlalchemy 5xx | `error_handlers.py:299-303` | ✅ |
| generic (unhandled) | `error_handlers.py:348-352` | ✅ |
| neo4j 503 | `error_handlers.py:444-450` | ✅ (új) |
| qdrant 503 | `error_handlers.py:494-500` | ✅ (új) |
| httpx 502/504 | `error_handlers.py:556-562` | ✅ (új) |

4xx ágak (autocognitix, IntegrityError 409, validation) → `logger.warning` → nem megy Sentry-be ✅. **Nincs maradék 5xx, ami exc_info nélkül logolna** a handlerekben. SSE stream hibák (HTTP 200 alatt) `logger.exception`-nel fedettek (`endpoints/diagnosis.py:1117`).

- **LOW** (pre-existing, nem regresszió): endpointból dobott nyers `HTTPException(500)` a FastAPI default handlerén megy át — se log, se Sentry, hacsak az endpoint maga nem logol (pl. `quick_analyze` `logger.exception`-t hív). A törölt `_capture_to_sentry` ezt korábban sem fedte.

## C) Frontend streaming éles viselkedés

Backend endpoint **létezik és teljes**: `POST /api/v1/diagnosis/analyze/stream` (`endpoints/diagnosis.py:805-844`), 300 s stream timeout + 10-es semaphore. A `complete` event **tartalmazza a `diagnosis_id`-t** (`diagnosis.py:1035`) → `streamDiagnosis` az `event.data`-t adja át (`diagnosisService.ts:471-474`) → `handleAnalysisComplete` a `result.diagnosis_id`-t olvassa (`DiagnosisPage.tsx:184-185`) → navigate. **Nincs ID-gap.**

Fallback lánc ✅: SSE `error` event / HTTP hiba / network hiba → `onError` → `handleStreamingError` (`DiagnosisPage.tsx:143-177`) → blocking `analyzeDiagnosis` → navigate vagy input+toast. A diffben javítva: a 2 perces frontend timeout most már `onErrorRef`-et hív (`AnalysisProgress.tsx:351-353`) → fallback fut, user nem ragad be ✅.

- **MEDIUM** `diagnosisService.ts:~496` — Ha a stream **complete/error event nélkül zárul tisztán** (pl. backend `_is_connected()` false-disconnect a `diagnosis.py:1087`-nél némán return-öl, vagy proxy zárja a kapcsolatot), a read-loop callback nélkül fejeződik be → a user a teljes 120 s timeoutot kivárja, mielőtt a fallback elindul. Javaslat: EOF után, ha nem jött `complete`, hívjon `onError`-t azonnal.
- **LOW** — Semaphore-telített backend `capacity` error eventje azonnali blocking `/analyze` fallbackot triggerel → túlterhelésnél a load-shedding hatástalan.
- **LOW** — Frontend 120 s abort < backend 300 s timeout: hosszú elemzésnél kliens abort után a fallback ÚJRA lefuttatja a teljes pipeline-t (dupla LLM költség, potenciálisan duplikált diagnosis session, ha a streamelt save épp az abort után fejeződik be).

## Összegzés

| Súlyosság | Db | Blokkol? |
|---|---|---|
| CRITICAL | 0 | — |
| HIGH | 0 | — |
| MEDIUM | 3 | Nem — issue/TODO felvétele kötelező |
| LOW | 5 | Nem — dokumentálva |
