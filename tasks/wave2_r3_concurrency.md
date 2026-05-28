# Wave2 R3 — Concurrency

Scope: utolsó commit (`4258ddc` — "fix: 5 HIGH a wave2 refix auditból").
Branch: `claude/sprint13-bugfixes-wave2`.

---

## A) Migration 019 LOCK időtartam

- **severity:** MEDIUM
- **finding:** A `LOCK TABLE diagnosis_archive IN SHARE MODE` Alembic
  transakcióban van (Alembic alapból `transaction_per_migration=True`),
  így a SHARE lock NEM csak a DELETE-ig, hanem a teljes `upgrade()`
  futás végéig (COMMIT) tart. A SHARE mód `ROW EXCLUSIVE`-ot
  (INSERT/UPDATE/DELETE) blokkol → minden writer vár.

  A lock alatt a migráció a következőket teszi:
  1. `DELETE ... RETURNING id` (gyors, kis taken).
  2. `op.create_index(...)` × 2 — **plain CREATE INDEX** (nem
     CONCURRENTLY). Ez maga `SHARE` lockot vesz, kompatibilis a már
     tartott SHARE-ral, de a teljes táblát végigolvassa.
  3. `op.create_foreign_key(...)` (plain ADD CONSTRAINT, nem NOT VALID).
     Ez **ACCESS EXCLUSIVE** lockot szerez (catalog update) és **full
     table scan**-t csinál a validációhoz. Ez a domináns idő.

  **Production-ben mennyi időig blokkol INSERT-eket?**
  - A `diagnosis_archive` jelenleg kicsi (a docstring szerint
    "small in practice"), tehát másodperc nagyságrend → elfogadható.
  - **Skálázódási kockázat:** ha az archív tábla több millió sorra nő
    (pl. 90 napos retention × magas forgalom), a teljes upgrade ideje
    percekre nyúlhat, és ez alatt az `archive cleanup` háttér worker
    INSERT-jei **mind blokkolódnak** — nem ütköznek (deadlock nincs,
    mert a worker SHARE-rel sem versenyez), csak várnak a worker
    pool-ban → connection pool exhaustion lehetséges, ha az
    application szerver lock_timeout nélkül vár.
  - A ROLLBACK semantikája viszont biztosítva van: ha bárhol elhasal,
    a SHARE lock felszabadul, a purge visszagörgetődik (audit trail
    print viszont már megtörtént stdout-ra → félrevezető lehet, mert
    a log szerint töröltünk, de a tábla változatlan).

  **Mitigáció (jelen kódban):** explicit megjegyzés a docstringben,
  hogy nagy táblához külön online pattern kellene. Nincs `lock_timeout`
  beállítva — egy elhúzódó migráció bizonytalan ideig blokkolna.

  **Worker race a docstring szerint zárva:** a SHARE lock pont azt
  garantálja, hogy a purge és a FK validation között új orphan nem
  szülessen — ez a TOCTOU védelem helyesen működik.

- **javaslat (LOW priority, nem blocker):** add `op.execute("SET
  lock_timeout = '30s'")` a LOCK TABLE előtt, hogy production-ben
  fail-fast legyen ha valami patológiásan elhúzódik, ahelyett hogy
  az egész deploy connection pool-t kiéheztetné.

---

## B) cd.yml job sorrend race

- **severity:** CLEAN (nincs race)
- **finding:** A GitHub Actions `needs:` szigorú dependency —
  egy job CSAK akkor indul, ha ÖSSZES `needs:` listájában szereplő
  job `success` állapotban végzett. Nincs időalapú race.

  Konkrétan:
  - `run-migrations`: `needs: [prepare, build-backend]` →
    elindulhat amint a backend image kész (frontent buildet nem várja,
    helyesen — a migráció DB-only).
  - `deploy-railway`: `needs: [prepare, build-backend, build-frontend,
    run-migrations]` → akkor és csak akkor indul, ha **mind a 4**
    sikeres. Ha `build-frontend` később végez mint `run-migrations`,
    a `deploy-railway` egyszerűen vár a `build-frontend`-re. Ha
    fordítva, vár a migration-ra. Sorrend irreleváns, pure DAG.

  Az új sorrend (migration ELŐTT deploy) helyesen garantálja, hogy
  az app sosem indul el régi DB schema-val. A frontend build
  párhuzamosan futhat a migration-nel (egyik sem ír DB-be), ami
  ideális — minimális kritikus út.

  **Egyéb job-ok ellenőrizve:**
  - `smoke-tests: needs: [prepare, deploy-railway, run-migrations]`
    → mindkettő success kell, helyes.
  - `notify: needs: [..., deploy-railway, smoke-tests]` + `if: always()`
    → mindig fut, helyes.
  - `rollback: needs: [..., smoke-tests, build-backend, build-frontend]`
    → `if: failure() && (build-backend/frontend failure)` → migration
    failure NEM triggereli rollback issue-t, ami **kissé hiányos**
    (ha a migration elhasal, deploy nem fut, de rollback issue sem
    nyílik) — de ez nem race, csak gap. **Out of scope** ehhez a R3 körhöz.

---

## Összegzés

- A) MEDIUM (skálázódási watch-out, jelen méretnél elfogadható).
- B) CLEAN.
