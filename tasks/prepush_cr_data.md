# Pre-Push Audit — Data/Migration Lead

**Branch:** `claude/code-review-fixes` vs `main` | **Dátum:** 2026-06-10
**Scope:** migration 019, schemas/diagnosis.py normalize_make, redis_cache.py dead code

## Összegzés: NINCS CRITICAL/HIGH — push nem blokkolt. 1 MEDIUM, 4 LOW.

---

## A) Migration 019 végállapot (`backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py`)

### A1. SET LOCAL érvényesség — OK, de szivárgás a batch-en belül — LOW
- `env.py:57` `context.begin_transaction()` → a migrációk tranzakcióban futnak, a `SET LOCAL lock_timeout` (019:54) érvényes SQL, nem warning-ol el.
- **DE:** `env.py` NEM használ `transaction_per_migration=True`-t → az összes pending migráció EGY tranzakcióban fut. A `SET LOCAL` így a 019 UTÁN futó migrációkra (020+) is érvényes marad ugyanabban a deploy batch-ben — egy jövőbeli, lassú lock-ot igénylő migráció váratlanul 30s timeout-tal bukhat. (A 38–53. sori komment csak a session-szintű szivárgást zárja ki, a tranzakción belülit nem.)
- Javaslat (nem blokkoló): `transaction_per_migration=True` az env.py `context.configure`-ban, vagy komment pontosítás.

### A2. Downgrade — JAVÍTVA, OK
- `downgrade()` (019:129-134) már CSAK a `diagnosis_archive_user_id_fkey`-t dobja `DROP CONSTRAINT IF EXISTS`-szel; a 013-hoz tartozó `ix_diagnosis_archive_*` indexeket nem érinti. Helyes.

### A3. Upgrade idempotencia — LOW
- DELETE: idempotens. `create_index(..., if_not_exists=True)`: idempotens.
- `op.create_foreign_key` (019:108) NEM guarded — ha a constraint már létezik (out-of-band létrehozás, pl. `create_all` futtatott dev DB), az upgrade elhasal. Alembic version-tracking + Postgres tranzakcionális DDL miatt normál úton nem fordulhat elő részleges állapot → csak LOW. Opció: `DROP CONSTRAINT IF EXISTS` közvetlenül a create előtt.

### A4. NOT VALID/VALIDATE — MOOT
- A végállapot teljesen elhagyta a NOT VALID/VALIDATE párt: SHARE lock + sima `ADD CONSTRAINT` (ami ADD közben validál). A "mit csinál a VALIDATE újrafuttatáskor" kérdés így tárgytalan. A docstring (23-24. sor) korrekten dokumentálja, hogy nagy táblánál más minta kellene.

---

## B) normalize_make hatása a MENTETT history adatokra

### B1. Split-brain history adat + nem normalizált filter — MEDIUM
- `schemas/diagnosis.py:34-44` és `:484-488`: `DiagnosisRequest` és `DiagnosisStreamRequest` kap validatort → ÚJ rekordok kanonikus make-kel ("Volkswagen") mentődnek.
- RÉGI rekordok nyers make-kel ("vw") maradnak a `diagnosis_sessions`-ben — **nincs backfill migráció** a diff-ben.
- A history endpoint (`endpoints/diagnosis.py:392`) a `vehicle_make`-et nyers `Query` paramként veszi át (NEM a `DiagnosisHistoryFilter` schemán keresztül!) és normalizálás NÉLKÜL adja a `repositories.py:521` `ilike(f"%{escape_ilike(vehicle_make)}%")` szűrőnek.
- Következmény: "vw" szűrés a régi nyers rekordokat találja, az újakat NEM ("%vw%" nem illeszkedik "Volkswagen"-re); "Volkswagen" szűrés fordítva. Egyik irány sem konzisztens.
- **Javaslat:** (1) backfill data migráció: `UPDATE diagnosis_sessions SET vehicle_make = <canonical>` a normalize_make táblával, ÉS (2) a filter param normalizálása az endpointban. Önmagában csak a filter normalizálása nem elég (régi nyers sorok láthatatlanok maradnak).

### B2. DiagnosisHistoryFilter dead schema — LOW
- `schemas/diagnosis.py:300-312` `DiagnosisHistoryFilter`-t semmi nem használja app kódban (az endpoint sima Query paramokkal dolgozik) → nem kapott validatort, de nem is él. Törlendő vagy bekötendő — jelenleg félrevezető kontraktus.

---

## C) redis_cache.py dead code törlés

### C1. Törölt NHTSA_* hivatkozások — OK
- Grep a teljes repón: `NHTSA_RECALLS|NHTSA_COMPLAINTS|NHTSA_VIN|get/set_nhtsa_*|get/set_vin_decode` → kódban NINCS maradék hivatkozás (csak a `tasks/code_review_fixes_2c28099.md` tasklog említi). Tiszta törlés.

### C2. CacheTTL.NHTSA_DATA most már szintén halott — LOW
- `redis_cache.py:71` `NHTSA_DATA = 21600` egyetlen használója a törölt metóduscsalád volt; az `nhtsa_service.py:276-277` saját TTL-eket használ (`VIN_CACHE_TTL=86400`, `RECALLS_CACHE_TTL=3600`).
- Egyetlen maradék hivatkozás: `tests/unit/test_redis_cache.py:63` assert a konstans értékére — egy halott konstans tesztelése. Javaslat: konstans + assert törlése (vagy hagyni, nem blokkoló).

---

## Besorolás összesítő

| # | Súly | Találat | Hely |
|---|------|---------|------|
| B1 | MEDIUM | Régi nyers / új kanonikus make split-brain, filter nem normalizált, nincs backfill | repositories.py:521, endpoints/diagnosis.py:392 |
| A1 | LOW | SET LOCAL átszivárog a batch további migrációira (nincs transaction_per_migration) | 019:54, env.py:57 |
| A3 | LOW | create_foreign_key nem guarded újrafuttatásra | 019:108 |
| B2 | LOW | DiagnosisHistoryFilter dead schema, validator nélkül | schemas/diagnosis.py:300 |
| C2 | LOW | CacheTTL.NHTSA_DATA halott konstans + teszt | redis_cache.py:71, test_redis_cache.py:63 |

**Verdikt: PUSH ENGEDÉLYEZETT** — MEDIUM (B1) backlog issue-ként felveendő.
