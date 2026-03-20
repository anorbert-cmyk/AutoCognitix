# Sprint 11: Frontend Testing & Production Hardening

## Dátum: 2026-03-20
## Státusz: IN PROGRESS

## Cél
- Frontend tesztek: 5/93 → 40+ komponens tesztelve (core pages + components)
- MEDIUM audit hibák javítása (backend + frontend)
- Audit tesztek a javításokhoz

## Csapat (12 fő)

| # | Szerep | Fókusz | Fájlok |
|---|--------|--------|--------|
| 1 | **Lead Koordinátor** | Delegate, verify, commit | - |
| 2 | FE Test Alpha | DiagnosisPage tesztek | `pages/__tests__/DiagnosisPage.test.tsx` |
| 3 | FE Test Beta | ResultPage tesztek | `pages/__tests__/ResultPage.test.tsx` |
| 4 | FE Test Gamma | DemoResultPage + PartStoreCard | `pages/__tests__/DemoResultPage.test.tsx`, `components/__tests__/PartStoreCard.test.tsx` |
| 5 | FE Test Delta | Auth pages (Login, Register) | `pages/__tests__/LoginPage.test.tsx`, `pages/__tests__/RegisterPage.test.tsx` |
| 6 | FE Test Epsilon | HistoryPage + NotFoundPage | `pages/__tests__/HistoryPage.test.tsx`, `pages/__tests__/NotFoundPage.test.tsx` |
| 7 | FE Test Zeta | ErrorBoundary + UI components | `components/__tests__/ErrorBoundary.test.tsx`, `components/ui/__tests__/ErrorState.test.tsx` |
| 8 | Backend Fix Alpha | diagnosis_service: commit, fallback, hasattr | `services/diagnosis_service.py` |
| 9 | Backend Fix Beta | rag_service: SHA256, shadowed k, redundant global | `services/rag_service.py` |
| 10 | Frontend Fix | Unsplash deprecated, keyboard a11y, hardcoded fallbacks | `pages/ResultPage.tsx`, `pages/DiagnosisPage.tsx` |
| 11 | Audit Test Writer | test_sprint11_audit.py backend fix tesztek | `tests/test_sprint11_audit.py` |
| 12 | Lint + Verify | Ruff, TypeScript build, run all tests | - |

## Deliverables

### Frontend Tests (Agent 2-7)
- [ ] DiagnosisPage: form validation, submit, speech recognition, DTC format
- [ ] ResultPage: data display, optional chaining, PDF/print, empty states
- [ ] DemoResultPage: pre-filled data, store cards, pricing display
- [ ] PartStoreCard: rendering, stock status, price formatting
- [ ] LoginPage: form validation, submit, error states
- [ ] RegisterPage: form validation, password strength, submit
- [ ] HistoryPage: list rendering, empty state, navigation
- [ ] NotFoundPage: rendering, navigation link
- [ ] ErrorBoundary: error catching, recovery, retry
- [ ] ErrorState: error type detection, display modes

### Backend Fixes (Agent 8-9)
- [ ] diagnosis_service: add db.commit() in _save_diagnosis_session
- [ ] diagnosis_service: fix _fallback_diagnosis missing used_fallback=True
- [ ] diagnosis_service: replace hasattr with getattr
- [ ] rag_service: MD5 → SHA256 for cache keys
- [ ] rag_service: fix shadowed lambda parameter k → item_key
- [ ] rag_service: remove redundant _rag_service global (keep singleton __new__)

### Frontend Fixes (Agent 10)
- [ ] Replace deprecated Unsplash Source URL with placeholder
- [ ] Add keyboard accessibility to clickable divs (role, tabIndex, onKeyDown)
- [ ] Remove hardcoded fallback data for empty history cards

### Audit Tests (Agent 11)
- [ ] test_diagnosis_service_commits_session
- [ ] test_fallback_diagnosis_used_fallback_true
- [ ] test_rag_service_uses_sha256
- [ ] test_no_shadowed_lambda_k
- [ ] test_no_redundant_global_rag_service
