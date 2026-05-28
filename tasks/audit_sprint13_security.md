# Sprint 13 Security Audit

## A) AuthZ / IDOR

- severity: LOW (no IDOR found — all endpoints properly scoped)
- findings:
  - `backend/app/api/v1/endpoints/garage.py:116-127` — `GET /vehicles` listázás: `service.get_vehicles(db, str(current_user.id), ...)` — user_id filter kötelező, OK.
  - `backend/app/api/v1/endpoints/garage.py:213-219` — `GET /vehicles/{id}` egyetlen jármű: `_get_vehicle_or_404(vehicle_id, str(current_user.id), db)` → `service.get_vehicle` `WHERE id=? AND user_id=?` (vehicle_garage_service.py:121-125). OK.
  - `backend/app/api/v1/endpoints/garage.py:239-251` — `PUT /vehicles/{id}` frissítés: kétlépcsős check (`_get_vehicle_or_404` + `service.update_vehicle(...user_id)`). OK.
  - `backend/app/api/v1/endpoints/garage.py:283-293` — `DELETE /vehicles/{id}` törlés: `_get_vehicle_or_404` + `service.delete_vehicle(..., str(current_user.id))`. OK.
  - `backend/app/api/v1/endpoints/garage.py:526-534` — `POST /reminders/{id}/complete`: `service.complete_reminder(db, reminder_id, str(current_user.id))` → query `WHERE id=? AND user_id=?` (service line 355-360). OK.
  - `backend/app/api/v1/endpoints/garage.py:573-581` — `DELETE /reminders/{id}`: `service.delete_reminder(db, reminder_id, str(current_user.id))` — `WHERE id=? AND user_id=?` (service line 380-385). OK.
  - `backend/app/api/v1/endpoints/garage.py:708-714` — `GET /vehicles/{id}/recalls`: `_get_vehicle_or_404` ellenőrzés van. OK.
  - **Általános minta**: minden endpoint `current_user: User = Depends(get_current_user_from_token)` paraméterrel + minden service hívás `user_id` paraméterrel, és minden service-query `WHERE user_id=?` szűrővel — IDOR nincs.

## B) JWT + rate limit

- severity: MEDIUM
- findings:
  - `backend/app/core/config.py:38` — `JWT_ALGORITHM = "HS256"` (symmetric). Egy adatbázis-szivárgás esetén ugyanaz a kulcs aláírásra és verifikációra használt. Multi-service architektúrában (frontend SSR / mobile API) **RS256** preferált. severity: MEDIUM.
  - `backend/app/core/security.py:172-194` — `jwt.decode()` `leeway=10` OK, type-confusion check (line 181) OK, JTI blacklist check (line 189) OK. NINCS `iss`/`aud` validáció, és nincs `options={"require": [...]}` — minimális kötelező claim lista. severity: LOW.
  - `backend/app/core/rate_limit.py:280-328` — `check_rate_limit_with_redis_fallback` **NEM tisztán fail-closed**: ha Redis kiesik, in-memory fallback fut (line 328). A docstring (line 293-296) "fail-closed policy"-t állít, de valójában a fallback engedi a kérést a limit alatt — multi-worker setup-ban (Railway gunicorn workers) ez per-worker limitet jelent, tehát N× a tényleges limit. severity: MEDIUM (production-degraded mód).
  - `backend/app/core/security.py:75-105` — `create_refresh_token` egyedi `jti`-t ad → blacklist alapú rotáció lehetséges. Verifikálandó az `/auth/refresh` endpoint-on, hogy a régi refresh JTI-t valóban blacklistelik-e (out of scope ezen audit-ban, lásd korábbi audit).

## C) Secrets exposure

- severity: LOW
- findings:
  - Frontend grep (`sk-|pk_|AIza|ghp_|eyJ|API_KEY|SECRET_KEY|JWT_SECRET`) `frontend/src/` alatt: **0 találat**. Nincs hardcoded literal secret. OK.
  - `backend/app/core/config.py:36-37` — `SECRET_KEY: str = ""` és `JWT_SECRET_KEY: str = ""` üres default + `field_validator` (line 47-65) startup-ra elutasít minden 32 karakternél rövidebb értéket. **Nincs placeholder ("changeme") default**. OK.
  - `backend/app/core/config.py:94,96` — `POSTGRES_PASSWORD: str = ""` és `DATABASE_URL: str = ""` üres defaultok, nincs hardcoded credential. OK.
  - `backend/app/core/config.py:26-27` — `DEBUG: bool = False` és `ENVIRONMENT: str = "development"` defaultok. `ENVIRONMENT="development"` default — Railway-en kötelezően `ENVIRONMENT=production` beállítandó; nincs startup-check ami `DEBUG=True && ENVIRONMENT=production` esetén figyelmeztet. severity: LOW.

## Olvasott fájlok

- `/home/user/AutoCognitix/backend/app/api/v1/endpoints/garage.py` (1-200 + grep teljes)
- `/home/user/AutoCognitix/backend/app/services/vehicle_garage_service.py` (grep teljes, user_id check verifikáció)
- `/home/user/AutoCognitix/backend/app/core/security.py` (155-204 + grep teljes)
- `/home/user/AutoCognitix/backend/app/core/rate_limit.py` (1-80, 280-328)
- `/home/user/AutoCognitix/backend/app/core/config.py` (1-100)
- Grep: `frontend/src/` hardcoded literal secret (3 grep)
