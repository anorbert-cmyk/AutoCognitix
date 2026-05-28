# API Contract Mismatch Audit

## A) Diagnosis response

| mező | backend | frontend | mismatch | severity |
|------|---------|----------|----------|----------|
| `id` | `Optional[UUID]` nullable (diagnosis.py:193) | `id: string` required (api.ts:292) | Frontend assumes always present; `result.id?.slice(-4)` in ResultPage.tsx:110 actually uses optional chain, but type is required string → TS believes never-undefined | HIGH |
| `created_at` | `datetime` default_factory (diagnosis.py:257), required | `created_at: string` required (api.ts:302) | OK shape, but type-wise backend serializes `datetime` → ISO string, frontend treats as string. No bug, just convention. | LOW |
| `RelatedComplaint` | structured: `odi_number`, `components`, `summary`, `crash`, `fire`, `similarity_score` (diagnosis.py:164-178) | `{complaint_id?, summary?, incident_date?, severity?}` (api.ts:284-289) **plus** union `RelatedComplaint[] \| string[]` (api.ts:307) | Field names ENTIRELY different (`odi_number` vs `complaint_id`, `crash`/`fire` missing, `severity`/`incident_date` not in backend). Frontend will display nothing. | CRITICAL |
| `urgency_level` | required with default `"medium"` (diagnosis.py:214) | `urgency_level?: string` optional (api.ts:309) | Minor — backend always sends it. | LOW |
| `safety_warnings`, `diagnostic_steps` | required, default `[]` (diagnosis.py:217,222) | optional `?` (api.ts:310-311) | Minor laxity; OK. | LOW |
| `ai_disclaimer` | required with long default (diagnosis.py:247) | optional `?` (api.ts:316) | Backend always sends; type laxity only. | LOW |

ResultPage.tsx reads: `vehicle_make`, `dtc_codes`, `probable_causes[0].title/description`, `symptoms`, `root_cause_analysis`, `confidence_score`, `recommended_repairs`, `parts_with_prices`, `total_cost_estimate.*`, `related_recalls` — all present in backend. **Does NOT read `similar_complaints`** → CRITICAL mismatch above is silent/unused on this page but would break any consumer.

## B) Garage response

- Backend `UserVehicleListResponse` (garage.py:124-126): `{vehicles: List[...], total: int}` — NO `page`/`size`/`limit`/`offset`.
- Frontend `UserVehicleListResponse` (garageService.ts:93-96): `{vehicles: UserVehicle[], total: number}` — **MATCHES**.
- `getVehicles()` (garageService.ts:174) calls `GET /garage/vehicles` with NO query params → no pagination at all is wired client-side; backend likely supports skip/limit but client never sends it. severity: MEDIUM — works only because user vehicle counts are tiny.
- `MaintenanceReminderListResponse` (garage.py:173 vs garageService.ts:134): `{reminders, total, overdue_count, urgent_count}` — MATCHES exactly.
- Field-level UserVehicle: backend `fuel_type: Optional[str]` (garage.py:108) vs frontend `fuel_type?: FuelType | null` (garageService.ts:57). Backend is unconstrained string; frontend narrows to enum. If DB has any value outside `FuelType` enum, TS lies. severity: MEDIUM.

## C) Recalls contract

- `RelatedRecall` backend (diagnosis.py:153-161): `campaign_number, component, summary, consequence?, remedy?, recall_date?` — all snake_case.
- `RelatedRecall` frontend (api.ts:275-282): identical fields, snake_case. **MATCHES**.
- `populate_by_name=True` / `ConfigDict` / `alias`: **NOT PRESENT** anywhere in `diagnosis.py` or `garage.py` (grep returned empty). Backend emits raw snake_case; frontend consumes snake_case → no aliasing layer needed.
- `VehicleRecall` (garageService.ts:279-) for `GET /garage/vehicles/{id}/recalls`: `campaign_number` snake_case again, same convention. Consistent.
- **No mismatch**, but also no defensive aliasing — if anyone later sets `alias_generator=to_camel` without `populate_by_name`, frontend breaks silently. severity: LOW (latent risk).

## Olvasott fájlok

- /home/user/AutoCognitix/backend/app/api/v1/schemas/diagnosis.py (1-299)
- /home/user/AutoCognitix/backend/app/api/v1/schemas/garage.py (98-220)
- /home/user/AutoCognitix/frontend/src/services/api.ts (200-340, 420)
- /home/user/AutoCognitix/frontend/src/services/garageService.ts (45-200, 279-296)
- /home/user/AutoCognitix/frontend/src/pages/ResultPage.tsx (28-495, grep only)
