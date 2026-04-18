# Sprint 13 Security Audit

## A) JWT & auth

- severity: HIGH
- findings:
  - `backend/app/core/security.py:173-178` — `jwt.decode()` nem ellenőriz `iss`/`aud` claimeket (csak HS256 + `exp` via leeway=10). Nincs `options={"require": ["exp","iat","sub","type","jti"]}` sem. Token más service-ből való újrahasznosítás (token-reuse across services) ellen nincs védelem.
  - `backend/app/api/v1/endpoints/auth.py:463-561` — **A login endpoint-on NINCS rate limit dekorátor** (se `slowapi`, se `@limiter.limit`). A védelem csak az account lockout-ra (`record_failed_login`) támaszkodik felhasználónként, így elosztott email-enumerációt / credential stuffing-et nem fékez. severity: **HIGH**.
  - `backend/app/api/v1/endpoints/auth.py:583-599` — Refresh endpoint rotálja a tokeneket (új access+refresh kiadása), de a `refresh_tokens()` függvényben **nem látszik a régi refresh token explicit `blacklist_token()` hívása** — token rotation incomplete, potenciális replay ablak. severity: HIGH. (Verifikálandó a 300+ sorral később.)
  - `backend/app/core/security.py:26` — Bcrypt default work factor (passlib alapértelmezés: rounds=12). Elfogadható, de nincs explicit konfigurálva (`bcrypt__rounds`), így passlib verzióváltás csendben csökkentheti. severity: LOW.
  - `backend/app/core/security.py:159-194` — `decode_token` `expected_type` ellenőrzése jó (type confusion elleni védelem), `jti` blacklist ellenőrzés fail-closed megfelelő (line 248-275). severity: OK.

## B) Injection

- severity: LOW (jó hygiene)
- findings:
  - `backend/app/api/v1/endpoints/health.py:211` — f-string Cypher (`MATCH (n:{label})`), DE a `label` változó **whitelist-ből** származik (`ALLOWED_LABELS = {"DTCCode","Symptom","Component","Repair"}` line 207) → nem user input, biztonságos. severity: OK.
  - `backend/app/db/postgres/repositories.py:369,521,526` — `ilike(f"%{escape_ilike(query)}%")` — helyesen használt `escape_ilike` a wildcard injection ellen (11 helyen, lásd `app/services/vehicle_service.py`, `rag_service.py`, `repositories.py`). severity: OK.
  - `backend/app/services/vehicle_garage_service.py` (13+ `db.execute` hely) — mind SQLAlchemy `select()`/paraméterezett statement, nincs raw SQL user input-tal. severity: OK.

## C) Secret hygiene

- severity: MEDIUM
- findings:
  - `.env.example:48,62` — placeholder secret-ek (`your_secret_key_here_generate_with_openssl_rand_hex_32`, `your_jwt_secret_key_here_minimum_32_characters`). OK, hogy placeholder, de **`DEBUG=true` + `ENVIRONMENT=development`** defaultok (line 49-50) veszélyesek, ha valaki véletlenül ezt másolja .env-be prodra. Ajánlás: `ENVIRONMENT=production` default + startup check, ami elutasítja az ismert placeholder értékeket. severity: MEDIUM.
  - `.gitleaks.toml:11-26` — konfiguráció **túl megengedő**: a `'''.*test.*\.py$'''` path allowlist bármilyen fájlra illeszkedik, ami tartalmazza a "test" szót (pl. `latest_config.py`). A `'''example.*token'''` és `'''test.*token'''` regex allowlist-ek szintén túl szélesek — valós secret-ek, amik a környezetükben "test" vagy "example" szót tartalmaznak, átcsúszhatnak. severity: MEDIUM.
  - Grep `os.getenv.*API_KEY|os.environ\[.*SECRET` a `backend/app/` alatt: **no matches** — minden API kulcs és secret a `settings` objektumon (Pydantic BaseSettings) keresztül jön, nincs szóródó `os.getenv` használat. severity: OK.

## Olvasott fájlok

- `/home/user/AutoCognitix/backend/app/core/security.py` (300 sor)
- `/home/user/AutoCognitix/backend/app/api/v1/endpoints/auth.py` (600 sor, 2 részben)
- `/home/user/AutoCognitix/backend/app/api/v1/endpoints/health.py` (kontextus, 195-224)
- `/home/user/AutoCognitix/.env.example` (80 sor)
- `/home/user/AutoCognitix/.gitleaks.toml` (30 sor)
- Grep: `db.execute|session.run|tx.run`, `_escape_ilike`, `os.getenv.*API_KEY`, `rate_limit|slowapi`, `iss|aud|issuer` (5 grep)
