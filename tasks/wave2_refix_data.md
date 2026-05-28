# Wave2 Refix — Data/Migration Lead

Branch: `claude/sprint13-bugfixes-wave2` @ HEAD `9ad6470`
Scope: migration 019 final correctness, `_embedding_cache_key` test determinism, Redis TTL orphan dokumentáció.

---

## A) Migration 019 audit trail + downgrade

### A1 — Orphan DELETE without row count / audit log
- **severity:** HIGH (GDPR-relevant data loss without trace)
- **finding:** `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py:41-47` — `op.execute(sa.text("DELETE FROM diagnosis_archive WHERE user_id IS NOT NULL AND user_id NOT IN (SELECT id FROM users)"))` returns no row count. In prod, ha 1k+ archive row vész el (historikus `user_id` deletions miatt) az SILENTLY történik — sem stdout-ra, sem audit table-ba nem kerül.
- **fix:** Cseréld `DELETE ... RETURNING id`-re és `op.get_bind().execute(...)` + `rowcount`/`result.fetchall()`, majd `print(f"[019] purged {n} orphan diagnosis_archive rows: {ids}")` — Alembic stdout-ja CI/CD log-ba megy, így van nyom. Opcionális: INSERT-eld előbb egy `migration_audit_purge` táblába (id, table_name, deleted_ids JSONB, migration_id, ts).

### A2 — Downgrade nem szimmetrikus, ciklus inkonzisztens
- **severity:** LOW (downgrade ritka, és csak rollback scenarióban probléma)
- **finding:** `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py:81-90` — downgrade `drop_constraint` → most már az old code újra létrehozhat orphan `user_id` row-okat (FK nélkül). Egy `downgrade → forgalom → upgrade` ciklus újra orphan-eket termel, amit a következő upgrade ismét NÉMÁN töröl (lásd A1). 
- **fix:** Nem cél a downgrade-et javítani (semantically lehetetlen az orphan-eket "visszahozni"), de docstring-be írd be: `"NOTE: downgrade leaves the table without FK enforcement; a subsequent re-upgrade WILL silently purge any new orphans created in the interim."` — az operator tudja, hogy downgrade után gyors re-upgrade kockázatos.

### A3 — `if_not_exists=True` Alembic >= 1.13 dependency
- **severity:** OK (verified)
- **finding:** `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py:57,63,87,89` — `if_not_exists=True` / `if_exists=True` az `op.create_index` / `op.drop_index` paramétere.
- **verify:** `backend/requirements.txt:` `alembic==1.13.1` és `requirements.prod.txt:` szintén `alembic==1.13.1`. A flag 1.13.0-ben került be → **OK, megfelelő**.

### A4 — NOT VALID + VALIDATE pattern — async asyncpg gotcha
- **severity:** MEDIUM (potential prod failure)
- **finding:** `019:68-78` — `ADD CONSTRAINT ... NOT VALID` és `VALIDATE CONSTRAINT` két külön `op.execute()`-ban. Alembic alapból DDL-tranzakcióban fut (psycopg2/asyncpg) → mindkét statement egy tranzakcióban van, így a `NOT VALID` előny (rövid lock) elveszik, mert a `VALIDATE` ugyanabban a tranzakcióban full table scan-t csinál + tartja a lock-ot. 
- **fix:** Vagy `op.execute("COMMIT")` a két statement között (autocommit block, lásd a 49-52. sor kommentárt — de ez NEM lett megcsinálva!), vagy explicit `transaction_per_migration = False` + manuális commit. A jelenlegi kód a komment ígéretét NEM tartja be.

---

## B) Test settings determinism

### B1 — `_expected_key` az élő `settings`-ből olvas → CI nem-determinisztikus
- **severity:** MEDIUM (flaky test risk)
- **finding:** `backend/tests/unit/test_redis_cache.py:554-560` — `_expected_key` importálja `app.core.config.settings`-et. `Settings(BaseSettings)` env_file=".env" + os.environ-ból olvas (`config.py:17-19`). Ha a CI runner env-ben `HUBERT_REVISION=foo` ki van állítva (pl. Railway preview deploy spillover), a test `_expected_key` és a service `_embedding_cache_key` szinkron értéket adnak, de a hash más, mint amit a fejlesztő lokálisan lát.
- **status:** `conftest.py:13-24` NEM állítja be `HUBERT_REVISION`-t (csak DATABASE_URL, NEO4J, stb.). `.env.example:136` `HUBERT_REVISION=main` — ha CI nem dump-olja ezt env-be, default `"main"` érvényes. Jelenlegi CI matrix (lásd `.github/workflows/ci.yml`) tiszta env-tel fut → **most determinisztikus, de fragile.**
- **fix:** `conftest.py:os.environ.setdefault("HUBERT_REVISION", "main")` + `os.environ.setdefault("HUBERT_MODEL", "SZTAKI-HLT/hubert-base-cc")` — vagy `@pytest.fixture(autouse=True)` monkeypatch a `TestEmbeddingCache` class-on. Védi a tesztet attól, hogy env spillover megváltoztassa a cache key-t.

### B2 — `test_embedding_key_changes_with_revision` mutál globális singleton-t
- **severity:** MEDIUM (parallel test breakage, pytest-xdist)
- **finding:** `test_redis_cache.py:584-593` — `settings.HUBERT_REVISION = "v1-abc"` egy module-level singleton mutáció. `try/finally` restore van, de ha pytest-xdist `-n auto` fut és párhuzamos worker olvassa `settings.HUBERT_REVISION`-t a `_expected_key`-ben (lásd B1), race condition: az `_expected_key` egy másik teszttől a mutated értéket kapja.
- **fix:** `monkeypatch.setattr(settings, "HUBERT_REVISION", "v1-abc")` pytest fixture-ön át — process-isolated, és pytest automatikusan restore-ol. NE direct attribute assignment singleton-on.

---

## C) Redis TTL orphan — backward compat

### C1 — Régi kulcs formátum orphan-ek a Redis-ben (TTL=3600s)
- **severity:** LOW (cosmetic / docs only)
- **finding:** A `_embedding_cache_key` régi formája `embed:{sha256(text)}` volt, új `embed:{sha256(MODEL@REVISION|text)}`. A deploy után minden régi kulcs unreachable lesz a kódból. `CacheTTL.EMBEDDINGS = 3600` (1 óra, lásd `redis_cache.py`) → 1 órán belül auto-expire.
- **embedding count megjegyzés:** A 35K HuBERT embedding NEM Redis-ben van, hanem Qdrant-ban (CLAUDE.md adatbázis tábla megerősíti). Redis-ben csak short-lived query cache → orphan size negligible (max néhány MB).
- **akció:** Csak dokumentáció. Javasolt sor a `tasks/lessons.md`-be:
  > "2026-05-28: `_embedding_cache_key` schema változás (MODEL@REVISION prefix) — régi `embed:*` kulcsok TTL=3600s alatt önmaguktól lejárnak, nincs kód-elérés. Qdrant 35K vector ÉRINTETLEN. No-op a deployhez, csak warm cache 1 órára elveszik."

---

## Olvasott fájlok
- `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py` (teljes)
- `backend/tests/unit/test_redis_cache.py:551-595`
- `backend/app/db/redis_cache.py:566-595`
- `backend/app/core/config.py:11-24,148-160`
- `backend/tests/conftest.py:13-34`
- `backend/requirements.txt`, `backend/requirements.prod.txt` (alembic==1.13.1)
- `.env.example:134-136`, `.env.railway.example:85`
- `git show HEAD --stat`, `git diff HEAD~1 HEAD -- backend/alembic backend/app/db backend/tests/unit/test_redis_cache.py`
