# Pre-Push Logic/Correctness Lead Audit — claude/code-review-fixes

Scope: `git diff main` (working tree, uncommitted). Date: 2026-06-10.

## Verdikt: 1 HIGH (push-teljesség), 1 MEDIUM, 1 LOW — egyébként PASS

---

## HIGH-1 — Untracked új modulok, amikre a diff importjai épülnek

- Fájlok: `backend/app/core/vehicle_makes.py`, `backend/app/core/pii.py` — **`??` untracked** (`git status`).
- A diff importálja őket:
  - `backend/app/api/v1/schemas/diagnosis.py:12` → `from app.core.vehicle_makes import normalize_make`
  - `backend/app/services/nhtsa_service.py:34` → `from app.core.vehicle_makes import is_eu_only, normalize_make`
  - `backend/app/core/logging.py:33` → `from app.core.pii import redact_pii`
- Ha a push `git add` nélkül történik, a backend **ImportError-ral el sem indul**, és a Sentry redaction + make-normalizáció hiányzik.
- **Javítás:** `git add backend/app/core/vehicle_makes.py backend/app/core/pii.py` commit előtt. Push BLOKKOLT enélkül.

## MEDIUM-1 — Title-case fallback eltávolítása regresszió megszűnt US márkáknál

- `backend/app/core/vehicle_makes.py:147-160` — `normalize_make` az ismeretlen make-et változatlanul engedi át (szándékos: McLaren/RAM védelem).
- DE: a CANONICAL_MAKES-ből hiányoznak megszűnt US márkák: **Pontiac, Saturn, Mercury, Oldsmobile, Plymouth, Scion, Hummer, Geo, Eagle, Daewoo**.
- main-en `"pontiac".title()` → `"Pontiac"` → NHTSA találatok; most `"pontiac"` verbatim megy a case-sensitive NHTSA API-nak → **0 recall, ami cache-elődik**.
- Javítás: identity bejegyzések felvétele ezekre a márkákra (nem blokkoló, TODO/issue).

## LOW-1 — Érvénytelen DTC kettős validációs kör

- `DiagnosisPage.tsx:123` csak non-empty-t ellenőriz; érvénytelen DTC (pl. "P30") esetén
  `streamDiagnosis` validátora `queueMicrotask`-on onError-t hív → parent fallback
  `analyzeDiagnosis` ugyanazon a validáción újra elbukik → toast + vissza input-ra.
- Helyes végállapot, de felesleges kör. Nem blokkoló.

---

## A) AnalysisProgress konverzió — PASS

- `AnalysisProgress.tsx:237-246`: mind a 8 `DiagnosisRequest` mező átmegy a `DiagnosisFormData`-ba;
  `vehicle_engine`/`vin`/`additional_context` undefined-safe (validátor `data.vin && ...` skip,
  requestBody `?.trim() || undefined`, `diagnosisService.ts:355-366`).
- Trace DiagnosisPage → validátor (`diagnosisService.ts:88-124`):
  - `vehicle_make: 'Ismeretlen'` → non-empty ✓ (`DiagnosisPage.tsx:130`)
  - `symptoms: 'Nincs megadva'` = 13 char ≥ 10 ✓ (`DiagnosisPage.tsx:134`)
  - `vehicle_year` fallback `currentYear`=2026 ∈ [1900, 2030] ✓ (`DiagnosisPage.tsx:132`)
  - `dtc_codes`: 1 elem, uppercase+trim ✓; formátum-check ld. LOW-1.
- `abortRef: AbortController` típus egyezik a `streamDiagnosis` visszatérésével ✓.
- 120s timeout: complete-nél (`:286-289`) és error-nál (`:311-314`) törölve; timeout-abort →
  `AbortError` lenyelve a read-loopban (`diagnosisService.ts:499-501`) → **nincs dupla onError**;
  parent értesítés a timeout-ágon (`:351`) helyes új fix ✓. `handleRetry` (`:476-495`) timer-cleanup + direkt restart ✓.

## B) normalize_make ↔ EU_ONLY_MAKES — PASS

- `is_eu_only(normalize_make(x))` minden EU-brandre True:
  `'Škoda'` → lower `'škoda'` kulcs → `"Skoda"` → `"skoda"` ∈ EU set ✓;
  `'citroën'/'Citroën'` → `"Citroën"` → `"citroën"` ∈ set ✓; `'DS Automobiles'` → `"DS"` → `"ds"` ✓;
  Trabant/Wartburg/Moskvich nincs CANONICAL-ban → pass-through, lowercase match ✓.
- Idempotencia: minden kanonikus érték lowercase kulcsa önmagára képez
  (BMW, MINI, GMC, MG, RAM, FIAT, CUPRA, SEAT, DS, FCA, smart, McLaren, DeLorean,
  Citroën, Mercedes-Benz, Rolls-Royce, Land Rover, Alfa Romeo, AM General, General Motors —
  mind ellenőrizve) → az nhtsa_service újra-normalizálása no-op ✓.
- Kivétel: MEDIUM-1 (hiányzó megszűnt US márkák).

## C) error_handlers exc_info ágak — PASS

- `sqlalchemy_exception_handler` (`error_handlers.py:256-310`): isinstance sorrend helyes
  (OperationalError/IntegrityError a szülő DBAPIError ELŐTT). 503 OperationalError,
  504 Timeout, 500 DBAPIError/egyéb → `logger.error(exc_info=True)` ✓;
  409 IntegrityError → `logger.warning` (nem megy Sentry-be) ✓.
- `autocognitix_exception_handler` (`:154-164`): `>=500` → error+exc_info, különben warning ✓.
- Neo4j 503 (`:452`), Qdrant 503 (`:502`), httpx 502/504 (`:564`): `exc_info=True` hozzáadva ✓.
- generic (`:344-349`): error+exc_info ✓. A kézi `_capture_to_sentry` eltávolítása biztonságos:
  `LoggingIntegration(event_level=logging.ERROR)` aktív (`logging.py:665-668`), PII-redakció
  központilag a `_sentry_before_send`-ben (`logging.py:583-618`, message + logentry + request.url
  + extra lefedve).
