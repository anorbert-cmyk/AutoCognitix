# Wave2 Push — Security Lead Review

Branch: `claude/sprint13-bugfixes-wave2` (commit 4f84293, 1 commit vs main)

## A) Migration 019 biztonság

- **severity: HIGH**
- **finding:** `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py:38-45` —
  `op.create_foreign_key(... ondelete="CASCADE")` futtatása ELŐTT NINCS orphan-cleanup
  lépés. A 013 migration óta a `diagnosis_archive.user_id` plain UUID volt FK nélkül,
  így bármely olyan archive sor, amely már törölt `users.id`-ra mutat (vagy hibás
  GDPR purge után árva maradt), PRODUCTION-BEN `ForeignKeyViolation`-nel megöli a
  `alembic upgrade head` lépést → release rollback / downtime.
- **finding 2 (LOW):** `validate=True` (default) az `add_foreign_key`-nél a teljes
  táblát végigfuttatja — nagy `diagnosis_archive` mellett hosszú lock + table scan
  a release ablakban. PostgreSQL-en ezért szokás `NOT VALID` + `VALIDATE CONSTRAINT`
  két lépésben (lock-light).
- **javaslat:**
  1. A FK létrehozás ELŐTT explicit orphan-purge a migration-be (az ORM modell
     `ondelete="CASCADE"` szemantikájával konzisztens, mert ezek a sorok amúgy is
     törlődtek volna):
     ```python
     op.execute("""
         DELETE FROM diagnosis_archive
         WHERE user_id NOT IN (SELECT id FROM users)
     """)
     ```
     vagy ha a GDPR audit nem engedi a néma törlést → migration előtti
     adat-konzisztencia ellenőrző script + manual review.
  2. Két-fázisú FK: `op.create_foreign_key(..., postgresql_not_valid=True)` majd
     `op.execute("ALTER TABLE diagnosis_archive VALIDATE CONSTRAINT diagnosis_archive_user_id_fkey")`.
  3. A `CONCURRENTLY` flag nincs jelen az index létrehozásánál (sor 26-35) — nagy
     táblán szintén lockolja az írásokat. `op.create_index(..., postgresql_concurrently=True)`
     + `with op.get_context().autocommit_block():` ajánlott prod-hoz.

## B) Sentry capture PII

- **severity: MEDIUM**
- **finding:** `backend/app/core/error_handlers.py:336-338` — `scope.set_tag("path", request.url.path)`.
  A `request.url.path` UUID-eket / user_id-t / VIN-t tartalmazhat (pl.
  `/api/v1/garage/vehicles/<uuid>/recalls`, `/api/v1/vehicles/decode-vin/<VIN>`).
  Sentry a **tag**-eket searchable módon indexeli és **alacsony cardinality** mezőnek
  tekinti — UUID/VIN tag-ek (1) cardinality blow-up-ot okoznak (Sentry quota),
  (2) PII-ket searchable formában tárolnak (GDPR Article 25 — data minimization sérül),
  (3) tag value max 200 char, query string-et bele NEM teszünk, de path-ban a path-param
  PII marad.
- **finding 2 (LOW):** `sanitize_log` NINCS alkalmazva itt (CRLF-injection a Sentry
  payload-ba nem életszerű, mert a Sentry SDK serializál, de a projekt belső
  szabálya szerint minden user-eredetű string felé kötelező — ld. CLAUDE.md
  "CodeQL log injection" tanulság).
- **javaslat:**
  1. `set_tag` helyett `set_extra` PII-érzékeny mezőkhöz — extra nem index, nem
     searchable, és nem számít cardinality-ba: `scope.set_extra("path", request.url.path)`.
     Tag-be csak az alacsony-cardinality route TEMPLATE menjen (pl.
     `request.scope.get("route").path` → `/api/v1/garage/vehicles/{vehicle_id}/recalls`).
  2. Globális Sentry init-ben `send_default_pii=False` (alapértelmezett, de explicit
     setting ajánlott) + `before_send` hook ami scrubol UUID/VIN regex-szel.
  3. `request_id` tag OK (random UUID, nem PII). `method` tag OK (max 8 érték).

## C) Embedding cache key

- **severity: LOW**
- **finding:** `backend/app/db/redis_cache.py:565-577` — `_embedding_cache_key` az
  SHA256 hash-be belekeveri `settings.HUBERT_MODEL` + `settings.HUBERT_REVISION`
  salt-ot. `config.py:152,156`: a két érték `Pydantic Settings`-en keresztül
  env-overridable (`HUBERT_MODEL`, `HUBERT_REVISION` env var). User input NEM
  fér hozzá direktben (csak deploy/admin az env-en keresztül).
- **risk értékelés:**
  - **Cache poisoning user-felől: nincs.** A user csak a `text` paramétert
    befolyásolja, a salt mindig az aktív server config.
  - **SHA256 collision: elhanyagolható** (2^-128 effektív).
  - **Verzió bump → stale eviction:** model upgrade esetén az új salt új kulcsot
    generál, a régi 1 órán belül TTL-en kiesik — helyes viselkedés.
  - **Operator risk (LOW):** ha az ENV véletlenül üres string-re kerül
    (`HUBERT_REVISION=""`), a salt `model@|` lesz, és minden környezet ami szintén
    üres-stringgel fut, közös cache namespace-be ír → nem biztonsági, hanem
    helyességi probléma.
- **javaslat:**
  1. Pydantic validátor a `HUBERT_REVISION` mezőre `min_length=1` vagy default
     `"main"` enforcement — már most `"main"` a default, de explicit
     `field_validator` védelmet adna ENV override esetén üres értéknél.
  2. A salt elválasztóként a `|` karakter OK, de a model név is tartalmazhat
     `@`-et theoretikusan — biztonsabb a `|` mindkét oldalon:
     `f"{HUBERT_MODEL}|{HUBERT_REVISION}|"`. Nem CVE, csak tisztább szeparáció.

## Összegzés

| Severity | Db |
|---|---|
| CRITICAL | 0 |
| HIGH | 1 (Migration 019 orphan FK) |
| MEDIUM | 1 (Sentry tag PII/cardinality) |
| LOW | 3 (FK validate lock, sanitize_log hiány, ENV validátor) |

A wave2 push diff produkció-blokkoló biztonsági CVE-t nem hoz, de a Migration 019
orphan-FK kockázata release-time crash-t okozhat — `alembic upgrade head` előtt
orphan-cleanup szükséges.

## Olvasott fájlok
- `/home/user/AutoCognitix/backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py`
- `/home/user/AutoCognitix/backend/app/core/error_handlers.py`
- `/home/user/AutoCognitix/backend/app/db/redis_cache.py`
- `/home/user/AutoCognitix/backend/app/core/config.py` (HUBERT_MODEL/REVISION defaults)
- `/home/user/AutoCognitix/backend/app/db/postgres/models.py` (DiagnosisArchive ORM)
- `/home/user/AutoCognitix/backend/app/core/log_sanitizer.py` (sanitize_log helper)
