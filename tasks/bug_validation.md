# Frontend Validation Audit

## A) Auth forms

### LoginPage (frontend/src/pages/LoginPage.tsx)
- **severity: MEDIUM** — `LoginPage.tsx:29-37` — Only checks `email.trim()` truthy + `password` non-empty. **No regex/email format validation** client-side. Relies on HTML5 `type="email"` (`:93`) which is bypassed if user disables it. No minimum password length check (login).
- **severity: LOW** — `LoginPage.tsx:42-44` — `catch {}` swallows the error silently; relies entirely on `AuthContext.error` state to surface it. If AuthContext fails to populate `error`, the user sees nothing.
- **severity: GOOD** — `LoginPage.tsx:75-79, 47` — Backend error IS displayed via `displayError = localError || error` — and `ApiError.fromAxiosError` (`api.ts:88-93`) correctly parses backend `detail.error.message_hu` / `message` / string `detail` field. So FastAPI 422/401 JSON errors render.
- **severity: GOOD** — `LoginPage.tsx:153, 271 (Register)` — `disabled={isLoading}` prevents double-submit while in flight.

### RegisterPage (frontend/src/pages/RegisterPage.tsx)
- **severity: GOOD** — `RegisterPage.tsx:12-19, 52-56` — Strong password policy: 8+ chars, upper, lower, digit, special. Failed reqs show generic message + `PasswordStrengthMeter` (`:220-222`).
- **severity: MEDIUM** — `RegisterPage.tsx:53-54` — Error message only says "A jelszo nem felel meg a kovetelmenyeknek" — **does NOT list which specific requirement(s) failed** in the error banner (the meter shows it, but only when focused). User loses field focus → meter hides → unclear what's wrong.
- **severity: MEDIUM** — `RegisterPage.tsx:41-44` — Email validation = `email.trim()` truthy only. No format regex. Backend rejection (e.g. duplicate email, malformed) surfaces only via AuthContext.error.
- **severity: LOW** — `RegisterPage.tsx:286-294` — Terms checkbox is a static `<p>` with `href="#"` — **no explicit consent checkbox** required, violates typical GDPR practice (CLAUDE.md mentions GDPR compliance).

## B) VIN/DTC validation

### VIN (frontend/src/services/vehicleService.ts + components/VehicleSelector.tsx)
- **severity: GOOD** — `vehicleService.ts:58-78` — `validateVIN()` enforces exactly 17 chars, rejects I/O/Q, regex `^[A-HJ-NPR-Z0-9]{17}$`. Called both in `decodeVIN()` (`:41-44`) and in `VehicleSelector.tsx:330` before mutation.
- **severity: GOOD** — `VehicleSelector.tsx:382, 388` — Live `vinInput.length/17` counter + decode button `disabled={... || vinInput.length !== 17 || vinDecode.isPending}`.
- **severity: LOW** — `VehicleSelector.tsx:112` — `vinInput` state has no max-length input attribute visible at line 367 area; user can paste >17 chars (button stays disabled, no truncation). Minor UX.

### DTC (frontend/src/components/features/diagnosis/DiagnosisForm.tsx)
- **severity: GOOD** — `DiagnosisForm.tsx:111-116` — Strict client regex `^[PBCU][0-9A-F]{4}$` applied to each tokenized code. Invalid codes listed in error message: `Érvénytelen hibakód(ok): ...`.
- **severity: GOOD** — `DiagnosisForm.tsx:185` — Auto-uppercases input: `e.target.value.toUpperCase()` — avoids common case mismatch.
- **severity: MEDIUM** — `DiagnosisForm.tsx:99-125` — `validateForm()` only checks DTC + `vehicleMake`. `vehicleYear`, `vehicleModel`, `ownerComplaints` are NOT required, **but the backend `DiagnosisRequest` (api.ts:206-215) treats `vehicle_year: number` and `symptoms: string` as required**. If user leaves year empty, `NewDiagnosisPage.tsx:132` falls back to `new Date().getFullYear()` (silently wrong data) and `symptoms` may be empty string passed through.
- **severity: MEDIUM** — `DiagnosisForm.tsx:103-106` — Splitter `/[,\s]+/` won't catch DTC with a hyphen prefix or quoted variant — minor edge case but no normalization beyond trim+upper.

## C) CSRF/session/double-submit

### CSRF token handling (frontend/src/services/api.ts)
- **severity: GOOD** — `api.ts:103-129` — CSRF token stored in-memory (not localStorage → XSS safe), attached as `X-CSRF-Token` header for POST/PUT/PATCH/DELETE. `withCredentials: true` (`:15`) sends httpOnly cookies.
- **severity: HIGH** — `api.ts:103-111` — **CSRF token is in-memory only**: on page refresh (F5) or new tab, `csrfToken` is `null` until next successful `/auth/refresh` (`:171-177`). State-changing requests issued BEFORE refresh completes will go out **without** CSRF header → backend rejects them. No bootstrap/init path seen that fetches CSRF on app mount.
- **severity: MEDIUM** — `api.ts:135-195` — 401 auto-refresh logic queues failed requests. If `/auth/refresh` fails, `window.dispatchEvent('auth:unauthorized')` is fired (`:191`) — but AuthContext must listen for it. If not wired, user stays on a broken page with no auto-logout/redirect.
- **severity: MEDIUM** — `api.ts:14` — `timeout: 30000` (30s) for ALL requests including diagnosis analysis. LLM RAG pipeline (CLAUDE.md mentions PartsPriceService + LLM) can easily exceed 30s → silent timeout → user sees network error instead of "still processing".
- **severity: LOW** — `LoginPage.tsx:153, RegisterPage.tsx:271, DiagnosisForm.tsx:268` — `isLoading`/`isSubmitting` disables submit button (good). But **no idempotency key** on diagnosis POST: if user hits Enter twice fast or network blip causes axios retry, two diagnosis records can be created.

### Top 3 problems
1. **CSRF token bootstrap gap** — fresh page load has no token until refresh; first state-change can fail.
2. **30s axios timeout** clashes with LLM-backed diagnosis endpoint that may take longer.
3. **No idempotency key** on diagnosis POST — double-submit risk on slow networks despite `isSubmitting` guard.

## Olvasott fájlok
- /home/user/AutoCognitix/frontend/src/pages/LoginPage.tsx
- /home/user/AutoCognitix/frontend/src/pages/RegisterPage.tsx
- /home/user/AutoCognitix/frontend/src/pages/NewDiagnosisPage.tsx
- /home/user/AutoCognitix/frontend/src/components/features/diagnosis/DiagnosisForm.tsx
- /home/user/AutoCognitix/frontend/src/components/VehicleSelector.tsx
- /home/user/AutoCognitix/frontend/src/services/vehicleService.ts
- /home/user/AutoCognitix/frontend/src/services/api.ts
