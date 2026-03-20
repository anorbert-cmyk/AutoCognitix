# Sprint 12: SSE Streaming Integration & Auth Flow Completion

## Dátum: 2026-03-20
## Státusz: IN PROGRESS

## Cél
- SSE streaming bekötése frontenden (AnalysisProgress valós adatokkal)
- Email service bekötése az auth endpointokba
- Jelszó erősség validáció (FE + BE)
- Integration tesztek a kritikus flow-kra

## Csapat (12 fő)

| # | Szerep | Fókusz | Fájlok |
|---|--------|--------|--------|
| 1 | **Lead Koordinátor** | Delegate, verify, commit | - |
| 2 | SSE Service | diagnosisService.ts SSE consumer | `services/diagnosisService.ts` |
| 3 | SSE Progress | AnalysisProgress.tsx valós SSE | `components/features/diagnosis/AnalysisProgress.tsx` |
| 4 | SSE DiagnosisPage | DiagnosisPage streaming route | `pages/DiagnosisPage.tsx` |
| 5 | Email Auth Wire | auth.py email integration | `api/v1/endpoints/auth.py` |
| 6 | Password Validator BE | Password strength endpoint | `api/v1/endpoints/auth.py`, `core/security.py` |
| 7 | Password Validator FE | Password strength component | `components/ui/PasswordStrength.tsx` |
| 8 | RegisterPage Wire | RegisterPage password meter | `pages/RegisterPage.tsx` |
| 9 | SSE Types | Streaming type definitions | `types/streaming.ts` |
| 10 | Backend SSE Test | Streaming endpoint audit | `tests/test_sprint12_streaming.py` |
| 11 | Frontend SSE Test | SSE integration test | `services/__tests__/diagnosisService.test.ts` |
| 12 | Ruff + Lint + Verify | Final verification | - |

## Deliverables

### P0 - SSE Streaming (Agent 2-4, 9)
- [ ] StreamingDiagnosisEvent TypeScript types
- [ ] streamDiagnosis() function in diagnosisService using EventSource
- [ ] AnalysisProgress.tsx: consume real SSE events, map to progress steps
- [ ] DiagnosisPage: route through streaming when available, fallback to POST
- [ ] Timeout handling for streaming (separate from regular API)

### P0 - Auth Email Integration (Agent 5)
- [ ] Wire email_service.send_password_reset_email() in forgot-password endpoint
- [ ] Wire email_service.send_welcome_email() in register endpoint
- [ ] Error handling: email failure should not block registration

### P1 - Password Strength (Agent 6-8)
- [ ] Backend: password_strength_check() in security.py
- [ ] Frontend: PasswordStrengthMeter component
- [ ] RegisterPage: integrate strength meter
- [ ] Rules: min 8 chars, uppercase, lowercase, digit, special char

### P1 - Tests (Agent 10-11)
- [ ] test_sprint12_streaming.py: SSE event format, progress tracking
- [ ] diagnosisService.test.ts: streamDiagnosis mock tests
