# Wave2 Refix — Logic/Correctness Lead

Commit auditált: `9ad6470` ("fix: 5 lead audit kritikus + magas találatok javítva")
Branch: `claude/sprint13-bugfixes-wave2`

---

## A) `_normalize_make` Title-case fallback — all-caps acronym regresszió

- **severity:** HIGH
- **file:line:** `backend/app/services/nhtsa_service.py:326` (`return cleaned.title()`)
- **finding:** A fallback `cleaned.title()` Python-szemantikailag minden ASCII-betűs token első karakterét nagybetűsíti, a többit kisbetűsre **kényszeríti**. Ez a "VOLKSWAGEN" → "Volkswagen" eset miatt készült (és arra jó), de **regressziót okoz az all-caps acronym brand-eknél, amik NEM szerepelnek a `BRAND_ALIASES`-ben**:
  - `"BMW".title()` → `"Bmw"` — az NHTSA recall API case-sensitive, a `"Bmw"` 0 találatot ad. A `BRAND_ALIASES` csak `"bmw ag"`-t mappeli, a puszta `"BMW"` átesik a fallback-en.
  - `"GMC".title()` → `"Gmc"` — `BRAND_ALIASES`-ben csak `"gmc truck"` van.
  - `"MINI"` esete: a dict-ben `"mini"` → `"MINI"` szerepel, tehát ezt menti a lookup, mielőtt a fallback elérné. Ez kivétel, nem szabály.
  - `"AUDI".title()` → `"Audi"` — itt szerencsére az `"Audi"` az NHTSA canonical, tehát véletlenül helyes.
  - Hibrid eset: `"Mercedes-Benz".title()` → `"Mercedes-Benz"` (kötőjeles tokenek külön kapitálizáltak, az NHTSA így várja). Ez OK, de szerencse.
  - Apostroph: `"O'Reilly".title()` → `"O'Reilly"` (oké, márka-irreleváns).
- **bizonyíték:** Python builtin `str.title()` minden alfanumerikus run elejére kapitalizál, a többit kisbetűsíti. CPython doc: "Return a titlecased version of the string where words start with an uppercase character and the remaining characters are lowercase." Az "ALL-CAPS acronym brand" → mixed-case konverzió matematikai biztossággal hiba az olyan brandeknél, ahol az NHTSA canonical maga is all-caps (BMW, GMC, MG, SRT).
- **javasolt fix (csak terv):**
  1) Bővíteni a `BRAND_ALIASES`-t puszta acronym kulcsokkal: `"bmw": "BMW", "gmc": "GMC", "mg": "MG", "srt": "SRT"` — ez O(1), backward compatible.
  2) Vagy: a fallback-ben SHORT (≤4 char) tokeneknél tartsa meg az `upper()` formát, hosszabbaknál `.title()`. De ez ad-hoc, az alias whitelist tisztább.
- **teszt-rés:** `tests/test_sprint_review_audit.py` jelenleg NEM teszteli az `_normalize_make` outputot — ajánlott parametrizált teszt: `[("BMW", "BMW"), ("bmw", "BMW"), ("VOLKSWAGEN", "Volkswagen"), ("GMC", "GMC"), ("land rover", "Land Rover")]`.

---

## B) Sentry `_capture_to_sentry` — `route.path` típus-feltételezés és WebSocket eset

- **severity:** LOW (defensive)
- **file:line:** `backend/app/core/error_handlers.py:329-330`
  ```python
  route = request.scope.get("route")
  route_template = getattr(route, "path", None) or "<no-route>"
  ```
- **finding:**
  1) **404 / nem-matchelt request:** ekkor a `request.scope`-ban a `"route"` kulcs hiányzik vagy `None`. A `getattr(None, "path", None)` → `None`, az `or "<no-route>"` fallback megfogja. **OK.**
  2) **Startup error / middleware-előtti exception:** ha az exception a routing előtt történik (pl. CORS middleware), a scope nem tartalmaz `"route"`-ot. **OK ugyanúgy.**
  3) **Mount-olt sub-app vagy `Mount` route:** a Starlette `Mount` típusnak van `.path` attribute-ja string formában (a prefix), tehát működik, de a tag value az **app-prefix** lesz, nem a sub-route — ez nem hiba, csak felbontás-veszteség. Dokumentálandó.
  4) **WebSocket route (`APIWebSocketRoute`):** ennek is van `.path` string attribute-ja, így nem hibázik. DE: a `_capture_to_sentry`-t request handler-ek hívják, WebSocket exception-ök NEM erre az útra futnak. Tehát ez gyakorlatban nem érintett.
  5) **`route.path` típus-garancia:** Starlette `BaseRoute` subclass-okban (`Route`, `WebSocketRoute`, `Mount`, `Host`) a `.path` ctor-argument str, és a class string-ként tárolja. Custom `BaseRoute` leszármazott elvileg felülírhatja, de a `or "<no-route>"` egy falsy stringet (üres string) is fallback-re visz, ami helyes viselkedés.
  6) **Tényleges edge case:** `Host` route-nál `path` lehet üres string (a Host csak `host` matchel). Az `or` operator üres stringet falsy-nak tekint → `<no-route>` lesz. **OK.**
- **konklúzió:** A `getattr(..., None) or "<no-route>"` minta robusztus a vizsgált esetekre. **Nincs tényleges bug**, csak két javaslat:
  - **teszt:** `test_capture_to_sentry_handles_missing_route` (scope nélkül) és `test_capture_to_sentry_handles_mount_route` — mindkettő ne raise-eljen.
  - **doc-komment:** `# route.path is str by Starlette contract; or-fallback handles None/empty.`

---

## C) Migration 019 `if_not_exists` Alembic kompatibilitás

- **severity:** RESOLVED (nem bug, megerősítve)
- **file:line:** `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py:32, 38` (`if_not_exists=True`), `:87-89` (`if_exists=True`)
- **vizsgált pin:** `backend/requirements.txt:15` → `alembic==1.13.1`; `backend/requirements.prod.txt:17` → `alembic==1.13.1`. **Egységes pin.**
- **finding:**
  - Az `op.create_index(..., if_not_exists=True)` kwarg-ot az Alembic **1.12.0** (2023-08-31, [changelog #1320](https://alembic.sqlalchemy.org/en/latest/changelog.html#change-1.12.0)) vezette be. Az `op.drop_index(..., if_exists=True)` ugyanebben a release-ben jelent meg.
  - **1.13.1 (prod pin) ezt támogatja** — a `BatchOperations` és `Operations` create_index signature is tartalmazza az `if_not_exists: bool = False` paramétert.
  - Mivel a projekt mindkét requirements fájlban (`requirements.txt`, `requirements.prod.txt`) ugyanazt a `1.13.1` pin-t használja, **nincs verzió-divergencia kockázat lokális vs Railway között**.
  - DB-szinten: a `CREATE INDEX IF NOT EXISTS` Postgres 9.5+ óta működik (Railway PG16 → OK), SQLite-ban is támogatott (teszt-runner OK).
- **figyelmeztetés (nem blokkoló):**
  - Ha valaki később downgrade-eli az Alembic-et `<1.12`-re, **TypeError: unexpected keyword argument 'if_not_exists'** fog repülni — nem silent ignore. Védelem: a `requirements*.txt`-ben `alembic>=1.12,<2` constraint biztonságosabb lenne a szigorú `==1.13.1`-nél, de a strict pin elfogadható reproducible-build policy.
  - **Suggestion:** `pyproject.toml`-ban a `dependencies` blokkba is felvenni az `alembic>=1.12` constraint-et, ha valaha pyproject-based install-ra váltunk.
- **konklúzió:** A `if_not_exists`/`if_exists` használata helyes a 1.13.1 pin mellett. **Nincs élő hiba**, csak dokumentáljuk a minimum verzió-igényt.

---

## Olvasott fájlok

1. `backend/app/services/nhtsa_service.py` (270-329)
2. `backend/app/core/error_handlers.py` (309-360)
3. `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py` (teljes, git diff via)
4. `backend/requirements.txt` (alembic pin)
5. `backend/requirements.prod.txt` (alembic pin)
6. `backend/pyproject.toml` (constraint check)
