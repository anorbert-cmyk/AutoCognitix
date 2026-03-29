## Frontend Logic Audit
**Auditor:** Frontend-Logic Specialist
**Dátum:** 2026-03-29
**Branch:** `claude/ralph-loop-global-memory-lFEhR`

### Érintett fájlok

| Fájl | Státusz |
|------|---------|
| `frontend/src/hooks/useStreamingDiagnosis.ts` | Auditálva |
| `frontend/src/contexts/AuthContext.tsx` | Auditálva |
| `frontend/src/services/diagnosisService.ts` | Auditálva |
| `frontend/src/services/garageService.ts` | Auditálva |
| `frontend/src/pages/DiagnosisPage.tsx` | Auditálva |
| `frontend/src/pages/ResultPage.tsx` | Auditálva |
| `frontend/src/components/features/diagnosis/AnalysisProgress.tsx` | Auditálva |
| `frontend/src/services/authService.ts` | Auditálva |
| `frontend/src/services/api.ts` | Auditálva |

---

### Talált hibák — Osztályozva

---

#### [HIGH-1] `useStreamingDiagnosis` — `streamDiagnosisGenerator` generator nem hívja meg az `onStart`/`onProgress` callback-eket

**Fájl:** `frontend/src/hooks/useStreamingDiagnosis.ts` (sor 175–195)

**Leírás:**
A `streamDiagnosisGenerator` async generator a `streamDiagnosis()` service-t hívja meg, de csak `onAnalysis`, `onComplete` és `onError` callback-eket ad át. Az `onStart`, `onContext`, `onCause`, `onRepair`, `onWarning` és `onProgress` callback-ek hiányoznak. Ha a generator-t fogyasztó kód ezekre a közbülső eseményekre támaszkodik (pl. haladásjelző UI), soha nem fogja megkapni azokat.

**Súlyosság:** HIGH — A generator API hiányos; progress állapot sosem frissül a generator variansnál.

---

#### [HIGH-2] `AuthContext` — Token lejárat után nincs automatikus logout/redirect a React context szintjén

**Fájl:** `frontend/src/contexts/AuthContext.tsx` (sor 66–85), `frontend/src/services/api.ts` (sor 191–196)

**Leírás:**
Az `initAuth` csak egyszer fut mountkor. Ha az access token lejár a session közben, az axios interceptor (`api.ts` sor 161–198) megpróbálja a refresh token-nel megújítani. Ha a refresh is meghiúsul, az interceptor `window.location.href = '/login'`-ra redirect-el — ez azonban **bypass-olja a React router-t** és a teljes alkalmazás állapotot elveszíti. Nem hív `clearTokens()`-t az AuthContext-en belül, így az `isAuthenticated` / `user` állapot inkonzisztens marad a redirect előtt.

Ezen felül az `isAuthenticated` értéke `!!user` a context-ben (sor 178), de az `authService.isAuthenticated()` egy különálló in-memory flag (`authenticated`). Ha az axios interceptor az `authenticated` flaget `false`-ra állítja (`setCsrfToken(null)` → `clearTokens()`), a context `user` state még mindig nem-null értékű marad egészen a page reload-ig.

**Súlyosság:** HIGH — Az auth állapot felhasználói felületen kívüli meghajtású invalidáció esetén inkonzisztens.

---

#### [HIGH-3] `diagnosisService.ts` — SSE parser figyelmen kívül hagyja az SSE protokoll szintű `event:` sort

**Fájl:** `frontend/src/services/diagnosisService.ts` (sor 279–309)

**Leírás:**
A `parseSSEEvents()` függvény az SSE eseményeket a `data:` sor alapján parsing-olja, és a JSON-ból olvassa az `event_type`-ot. Ez azt feltételezi, hogy a backend az event type-ot kizárólag a JSON payloadban (`event.event_type`) küldi, nem az SSE `event:` header sorban. A parser a `event:` sort a for-ciklusban csendesen elveti (nincs `if (line.startsWith('event: '))` ág). Ha a backend a standard SSE formátumot használja (`event: analysis\ndata: {...}`) de nem teszi bele az `event_type`-ot a JSON-ba is, akkor a dispatch silent-fail-t szenved.

**Súlyosság:** HIGH — Nem-teljes SSE protokoll implementáció; backend SSE `event:` header sor elvész.

---

#### [MEDIUM-1] `DiagnosisPage` — Frontend validáció nem fut a submit előtt

**Fájl:** `frontend/src/pages/DiagnosisPage.tsx` (sor 121–140)

**Leírás:**
Az űrlap submit (`handleSubmit`) csak a `dtcCode.trim()` üresség-ellenőrzést végzi el, majd azonnal `setCurrentStep('analysis')` hívással átvált az `AnalysisProgress` komponensre. A `validateDiagnosisRequest()` (`diagnosisService.ts` sor 88–126) csak a service layer-ben fut le, miután az `AnalysisProgress` már megjelent és a streaming elindult. Ha a felhasználó szóközzel elválasztott kódokat gépel be (`P0300 P0301`), az teljes egészében egyetlen érvénytelen DTC kódként kerül elküldésre, a streaming indul, majd az error callback hívódik — ahelyett, hogy az input mezőnél azonnal hibát kapna.

**Súlyosság:** MEDIUM — Késői validáció rontja a UX-et; az érvénytelen adat a streaming start-ig nem derül ki.

---

#### [MEDIUM-2] `DiagnosisPage` — Submit gomb disabled állapota streaming módban soha nem aktiválódik

**Fájl:** `frontend/src/pages/DiagnosisPage.tsx` (sor 416–425)

**Leírás:**
A submit gomb `disabled={analyzeDiagnosis.isPending}` állapotban van. Az `analyzeDiagnosis.isPending` azonban soha nem lesz `true` a normál streamelt útvonalnál, mert `handleSubmit` közvetlenül `setCurrentStep('analysis')`-t hív, nem indít `analyzeDiagnosis.mutateAsync()`-t. Ennek következménye, hogy a gomb sosem kerül disabled állapotba streaming módban — ha az `AnalysisProgress` komponens visszanavigál (pl. hiba esetén), a felhasználó újra beküldheti az űrlapot a gomb vizuális visszajelzése nélkül.

**Súlyosság:** MEDIUM — Dupla submit lehetséges; hiányzó loading feedback streaming módban.

---

#### [MEDIUM-3] `AnalysisProgress` — `isMockMode` useMemo dependency array hiányos

**Fájl:** `frontend/src/components/features/diagnosis/AnalysisProgress.tsx` (sor 412–415)

**Leírás:**
```tsx
const isMockMode = useMemo(
  () => !streamingEnabled || !getStreamDiagnosisFn() || !diagnosisRequest,
  [streamingEnabled, diagnosisRequest]
);
```
A `getStreamDiagnosisFn()` egy runtime modul-lookup, amely potenciálisan eltérő értéket adhat vissza újrarenderelések között (pl. lazy import esetén). Mivel a `useMemo` dependency array-ben nem szerepel a függvény referenciája, a memo cache-elt értéket ad vissza és nem reagál a modul betöltési állapot változásaira. Ez azt okozhatja, hogy mock mode-ban ragad az oldal, miközben a streaming már elérhető.

**Súlyosság:** MEDIUM — Lazy-loaded modulok esetén az oldal mock mode-ban ragadhat.

---

#### [MEDIUM-4] `AnalysisProgress` — `handleRetry` manuálisan `mountedRef.current = true`-ra állít, de ez megkerüli a cleanup logikát

**Fájl:** `frontend/src/components/features/diagnosis/AnalysisProgress.tsx` (sor 493–513)

**Leírás:**
A `handleRetry` függvény (sor 509) `mountedRef.current = true`-ra állítja a ref-et. Ha viszont a useEffect cleanup már lefutott (pl. React Strict Mode kétszeres mount/unmount), a `mountedRef` visszaállítása megkerüli azt a védelmet, amelyet a ref nyújt az unmountolt komponensen végrehajtott setState hívások ellen. Strict Mode-ban ez "state update on unmounted component" warningot okozhat.

**Súlyosság:** MEDIUM — React Strict Mode / fejlesztési környezetben bugos viselkedés.

---

#### [MEDIUM-5] `ResultPage` — `handleSavePDF` és `handlePrintWorksheet` azonos `window.print()` hívást tartalmaz

**Fájl:** `frontend/src/pages/ResultPage.tsx` (sor 65–74)

**Leírás:**
Mindkét handler (`handleSavePDF` és `handlePrintWorksheet`) csak `window.print()`-et hív, más logika nélkül. A két gomb eltérő label-lel jelenik meg ("Nyomtatás / PDF" vs "Munkalap nyomtatása"), de viselkedésük azonos. Ha a felhasználó a "Munkalap nyomtatása" gombra kattint azzal a várakozással, hogy különálló munkalapformátumot kap, megtévesztő élményt tapasztal.

**Súlyosság:** MEDIUM — Félrevezető UX; a kettős gomb megtéveszti a felhasználót.

---

#### [LOW-1] `useStreamingDiagnosis` — `stopStreaming` nem nullázza a `progress` értékét

**Fájl:** `frontend/src/hooks/useStreamingDiagnosis.ts` (sor 108–112)

**Leírás:**
Ha a stream befejezés előtt `stopStreaming()`-et hívunk, az `isStreaming` `false`-ra kerül, de `progress` értéke megőrzi az utolsó értéket (pl. 0.82). Ha az UI ezt az értéket progress bar megjelenítéséhez használja, a következő `startStreaming()` hívásnál csak akkor kerül nullázásra, amikor az `INITIAL_STATE` spread lefut (sor 64) — ami elvárt viselkedés. Azonban ha a komponens a `progress`-t a `stopStreaming()` után is rendereli (pl. visszaküldi a parent-nek), vizuálisan félrevezető marad.

**Súlyosság:** LOW — Vizuális inkonzisztencia szélsőséges esetben.

---

#### [LOW-2] `AuthContext` — `refreshUser` csendes hibát produkál lejárt token esetén

**Fájl:** `frontend/src/contexts/AuthContext.tsx` (sor 165–175)

**Leírás:**
A `refreshUser` callback az `authService.isAuthenticated()` értékét ellenőrzi, ami egy in-memory flag, nem a valódi token érvényességét. Ha a token lejárt, de a flag még `true`, `refreshUser()` API hívást indít, ami 401-es hibát kaphat. A catch ágban `clearTokens()` és `setUser(null)` hívódik, de az `error` state nem kerül set-elésre. A hívó kód tehát nem tudja megkülönböztetni a sikeres és a silent-fail `refreshUser()` hívást.

**Súlyosság:** LOW — Csendes hiba, nehezen debuggolható.

---

#### [LOW-3] `diagnosisService.ts` — `formatCostRange` hamis negatívot ad `0` értéknél

**Fájl:** `frontend/src/services/diagnosisService.ts` (sor 609–629)

**Leírás:**
```ts
if (!min && !max) { return 'Nincs becslés' }
```
Ha `min = 0` (érvényes ár), az `!min` feltétel `true`, és ha `max` is `0` vagy undefined, a függvény `'Nincs becslés'`-t ad vissza. Ez JavaScript falsy értékkezelési antipattern — a `0` mint érvényes nullás ár nem kezelt. A helyes ellenőrzés: `if (min == null && max == null)`.

**Súlyosság:** LOW — Nullás/ingyenes árak esetén helytelen megjelenítés.

---

#### [LOW-4] `DiagnosisPage` — Szükségtelen `diagnosisId` state a navigáció előtt

**Fájl:** `frontend/src/pages/DiagnosisPage.tsx` (sor 82, 178–184)

**Leírás:**
A `diagnosisId` state (sor 82) `setDiagnosisId(resultId)` hívással frissül a `handleAnalysisComplete`-ben (sor 182), de ezt közvetlenül a `navigate()` hívás követi (sor 183). Az állapotfrissítés felesleges, mivel semmilyen render nem függ ettől az értéktől a navigáció előtt — az `AnalysisProgress` az `undefined`-ot kapja egészen az ID megérkeztéig. Ez szükségtelen setState → re-render ciklust idéz elő.

**Súlyosság:** LOW — Felesleges re-render, nincs funkcionális hatás.

---

### Összefoglalás

| Szint | Darab |
|-------|-------|
| HIGH | 3 |
| MEDIUM | 5 |
| LOW | 4 |
| **Összesen** | **12** |

### Kritikus figyelmet igénylő területek (prioritás sorrendben)

1. **Auth állapot szinkronizáció** (HIGH-2): A token lejárat kezelése két különböző rendszeren van elosztva (axios interceptor + AuthContext) és ezek nincsenek szinkronizálva. A `window.location.href` redirect az AuthContext állapotát inkonzisztensen hagyja.

2. **SSE parser robusztussága** (HIGH-3): A `parseSSEEvents` csak a JSON payload `event_type` mezőjét nézi, az SSE protokoll-szintű `event:` sort figyelmen kívül hagyja — nem-teljes SSE implementáció.

3. **Generator API hiányossága** (HIGH-1): A `streamDiagnosisGenerator` csak 3 callback-et ad tovább a lehetséges 8-ból, ezért a haladás és közbülső lépések sosem jelennek meg generátor alapú fogyasztóknál.

---

## Code Quality Audit
**Auditor:** Code-Quality Specialist
**Date:** 2026-03-29
**Files Reviewed:**
- `backend/app/core/security.py`
- `backend/app/services/email_service.py`
- `backend/app/db/postgres/repositories.py`
- `backend/app/core/exceptions.py`
- `backend/app/core/error_handlers.py`

---

### Summary

13 issues found: 1 HIGH, 5 MEDIUM, 7 LOW.

---

### Issue CQ-1 — HIGH: Duplicate `send_password_reset` methods — unescaped path is active; XSS-safe method is dead code

**File:** `backend/app/services/email_service.py`, lines 251–365

**Description:**
`EmailService` has two separate public methods for sending password reset emails:

1. `send_password_reset(to_email, name, reset_link)` (line 251) — uses `PASSWORD_RESET_TEMPLATE_HU/HTML` with `.format()`. `name` and `reset_link` are inserted **without HTML-escaping** into the HTML body. A malicious username (`<script>alert(1)</script>`) would be embedded raw into the email HTML.

2. `send_password_reset_email(to_email, reset_token, username, expires_minutes)` (line 287) — correctly calls `html.escape()` on `safe_username` and `safe_reset_url` before embedding. This is the secure version.

The module-level convenience function `send_password_reset_email` (line 638) routes to `service.send_password_reset()` — the **first (unescaped)** method, not the second. The `auth.py` endpoint imports and calls this module-level function at line 961, so production traffic uses the unescaped path.

The instance method `EmailService.send_password_reset_email` (the safe version) has **zero callers** in production — it is dead code.

**Checklist items:** #9 (email service duplication), #3 (dead code)

---

### Issue CQ-2 — MEDIUM: `check_password_strength` and `PASSWORD_PATTERN` are dead code

**File:** `backend/app/core/security.py`, lines 277–319

**Description:**
`check_password_strength` (line 277) has no callers anywhere in `app/`. The functionally equivalent `validate_password_strength` (line 322) is the one used by `schemas/auth.py` as a Pydantic validator.

`PASSWORD_PATTERN` (line 319) is a compiled regex that is also never referenced. `validate_password_strength` uses individual `re.search()` calls, not this pattern.

Both are dead exported symbols inflating the module's public API surface.

**Checklist item:** #3 (dead code)

---

### Issue CQ-3 — MEDIUM: `verify_csrf_token` is never called — CSRF validation not enforced server-side

**File:** `backend/app/core/security.py`, lines 396–409

**Description:**
`verify_csrf_token(token)` is defined and exported, but a full codebase search finds **zero callers**. `generate_csrf_token()` is called in `auth.py` (lines 549, 664) to mint tokens returned to the frontend, but the verification step is never invoked on incoming state-changing requests. CSRF tokens are generated, returned, and then silently ignored on the server side. This renders the CSRF mechanism incomplete.

**Checklist item:** #3 (dead code / missing enforcement)

---

### Issue CQ-4 — MEDIUM: `decode_token` uses undocumented `leeway=30` — magic number with security implications

**File:** `backend/app/core/security.py`, line 176

**Description:**
`jwt.decode(..., leeway=30)` silently accepts tokens that expired up to 30 seconds ago. No comment explains the rationale (clock-skew tolerance? distributed deployment?). The value is a magic number with no named constant. A future developer reducing token lifetime to a short window (e.g. 60-second one-time-use tokens) may not notice the leeway extends validity by 50%. Should be extracted to a named constant with a docstring.

**Checklist item:** #4 (magic numbers)

---

### Issue CQ-5 — MEDIUM: `_smtp_send` parameter typed as `object` but calls `.as_string()` — type annotation error

**File:** `backend/app/services/email_service.py`, lines 421–430

**Description:**
```python
def _smtp_send(self, msg: object, to: str) -> None:
    ...
    smtp.sendmail(self._from_email, [to], msg.as_string())
```
`object` has no `as_string()` method. The actual runtime type is `email.mime.multipart.MIMEMultipart`. The mypy error is masked by `# type: ignore[arg-type]` on the call site in `_send()` (line 425), not in `_smtp_send` itself. Direct or future calls to `_smtp_send` with a non-MIME argument would raise `AttributeError` at runtime with no type-check warning.

**Checklist item:** #6 (type annotations)

---

### Issue CQ-6 — MEDIUM: `get_dtc_frequency` builds a `conditions` list that is never composed into any query

**File:** `backend/app/db/postgres/repositories.py`, lines 686–713

**Description:**
```python
conditions: List[ColumnElement[bool]] = [DiagnosisSession.is_deleted.is_(False)]
if user_id:
    conditions.append(DiagnosisSession.user_id == user_id)
```
This ORM-style conditions list is assembled then immediately abandoned — the method switches to two separate raw `text()` SQL strings that hard-code the same filters. The `conditions` variable has no effect on any executed query. Any future developer adding a condition to `conditions` will find it silently ignored.

(This finding also aligns with DB-4 from the Database Specialist.)

**Checklist item:** #3 (dead code), #5 (confusing duplicate logic)

---

### Issue CQ-7 — LOW: `VehicleMakeRepository` and `VehicleModelRepository` `__init__` missing `-> None` return annotation and docstring

**File:** `backend/app/db/postgres/repositories.py`, lines 363–364, 377–378

**Description:**
Both `__init__` methods omit the `-> None` return type annotation and have no docstring, inconsistent with `UserRepository` and `DTCCodeRepository` which are fully annotated. Minor inconsistency flagged by strict mypy.

**Checklist items:** #6 (type annotations), #7 (docstrings)

---

### Issue CQ-8 — LOW: `check_password_strength` feedback strings are ASCII-only; `validate_password_strength` uses proper Hungarian accents — inconsistency

**File:** `backend/app/core/security.py`, lines 301–308 vs 344–364

**Description:**
`feedback_map` entries (e.g. `"Nagyon gyenge jelszo"`) use ASCII-only transliterations. `validate_password_strength` in the same file uses accented Hungarian (`"A jelszónak legalább 8 karakter hosszúnak kell lennie."`). Since `check_password_strength` is already dead code (CQ-2) this is low priority, but the inconsistency should be resolved if the function is ever activated.

**Checklist item:** #10 (Hungarian consistency)

---

### Issue CQ-9 — LOW: `_send_email` detects email type/language via fragile subject-line substring matching

**File:** `backend/app/services/email_service.py`, lines 553–566

**Description:**
```python
if "Welcome" in subject or "Üdvözöljük" in subject or "Üdvözlünk" in subject:
    email_type = "welcome"
if any(eng in subject for eng in ["Welcome", "Confirm your"]):
    language = "en"
```
Email type and language for the n8n webhook payload are inferred from subject substrings — a fragile heuristic. A subject string change silently changes the webhook payload with no compile-time or test-time signal. `send_via_n8n` already accepts explicit `email_type` and `language` parameters; callers should pass them directly rather than routing through `_send_email`'s guessing logic.

**Checklist item:** #4 (magic strings)

---

### Issue CQ-10 — LOW: `RequestContextMiddleware.dispatch` missing return type annotation

**File:** `backend/app/core/error_handlers.py`, line 50

**Description:**
```python
async def dispatch(self, request: Request, call_next: Callable):
```
Return type is unannotated. The correct type is `Response` (from `starlette.responses`). All other async handler functions in this file have explicit `-> JSONResponse` return types. Flagged by strict mypy.

**Checklist item:** #6 (type annotations)

---

### Issue CQ-11 — LOW: `RAGException` default message has typo — `"Tudazbazis"` instead of `"Tudasbazis"`

**File:** `backend/app/core/exceptions.py`, line 615

**Description:**
```python
message: str = "Tudazbazis keresesi hiba.",
```
The correct word is `"Tudasbazis"` (knowledge base). The `ERROR_MESSAGES_HU` dict (line 121) correctly uses `"Tudasbazis keresesi hiba."` — the exception default message and the messages dict are inconsistent.

**Checklist item:** #10 (Hungarian consistency / typo)

---

### Issue CQ-12 — LOW: Multiple exception subclasses override `self.code` post-`__init__` — fragile post-init mutation pattern

**File:** `backend/app/core/exceptions.py`, lines 226, 245, 285, 377, 409, 506, 541, 558

**Description:**
Several exception classes call `super().__init__()` with one error code, then immediately mutate `self.code`:
```python
super().__init__(message=message, original_error=original_error)  # sets code=NEO4J_ERROR
self.code = ErrorCode.NEO4J_CONNECTION                            # then mutates it
```
The correct approach is to pass the intended `code` value directly in `super().__init__()`. Post-init mutation is confusing and makes `__init__` signatures unreliable as documentation. Affected classes: `DTCValidationException`, `VINValidationException`, `VehicleNotFoundException`, `Neo4jConnectionException`, `QdrantConnectionException`, `NHTSARateLimitException`, `LLMRateLimitException`, `LLMUnavailableException`.

**Checklist item:** #5 (inconsistent pattern)

---

### Issue CQ-13 — LOW: `send_via_n8n` logs `response.text[:200]` without `_sanitize_log` — potential log injection from external response body

**File:** `backend/app/services/email_service.py`, line 524

**Description:**
```python
logger.warning(
    f"n8n webhook returned {response.status_code}: {response.text[:200]}"
)
```
`response.text` is third-party-controlled content. A response body containing embedded newlines (common in JSON error messages) can inject fake log lines. All other log statements in this file use `_sanitize_log(...)`. This one inconsistently does not.

**Checklist item:** #8 (log injection)

---

### Checklist Results

| # | Check | Result | Issues |
|---|-------|--------|--------|
| 1 | None safety — `Optional` returns checked before use | PASS | All `Optional` repository returns are guard-checked at call sites |
| 2 | Exception handling — `bare except` / stacktrace swallowing | PASS | All `except Exception as e` blocks log `e`; no bare `except:` found |
| 3 | Dead code — uncalled functions / imports | FAIL | CQ-2 (`check_password_strength`, `PASSWORD_PATTERN`), CQ-3 (`verify_csrf_token`), CQ-6 (`conditions` list) |
| 4 | Magic numbers/strings — hardcoded values without constants | MEDIUM | CQ-4 (`leeway=30` undocumented), CQ-9 (subject-substring language detection) |
| 5 | Duplicate code — repeated logic | HIGH | CQ-1 (two password reset methods; active path is unescaped; safe method unreachable) |
| 6 | Type annotations — missing return/param types | MEDIUM | CQ-5 (`_smtp_send` typed `object`), CQ-7 (repo `__init__`), CQ-10 (`dispatch`) |
| 7 | Docstrings — public API methods | LOW | CQ-7 (`VehicleMakeRepository`, `VehicleModelRepository`) |
| 8 | Log levels / log injection | LOW | CQ-13 (`response.text` not sanitized in `send_via_n8n`) |
| 9 | Email service — duplicate `send_password_reset` methods | HIGH | CQ-1 (module-level function routes to unescaped method; secure method is dead code) |
| 10 | Hungarian string consistency | LOW | CQ-8 (`check_password_strength` ASCII-only), CQ-11 (`RAGException` typo) |
