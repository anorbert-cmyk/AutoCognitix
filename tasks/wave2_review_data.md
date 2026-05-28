# Wave2 Push — Data/Migration Lead

Branch: `claude/sprint13-bugfixes-wave2` vs `main`. Diff scope:
- `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py` (new)
- `backend/app/db/redis_cache.py` (embedding cache key salting)
- `backend/app/core/config.py` (`HUBERT_REVISION` setting)

## A) Alembic 019 korrektség

### A1. Orphan `user_id` → FK creation crash (production)
- severity: **CRITICAL**
- finding: `backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py:38-45` —
  `op.create_foreign_key("diagnosis_archive_user_id_fkey", ..., ondelete="CASCADE")`
  jelenleg azonnal validál minden meglévő `user_id`-t.
  A `013` migration óta a `diagnosis_archive` írásra nyitva volt FK nélkül,
  user CASCADE delete pedig nem futott át ide → reálisan vannak/lehetnek
  olyan `diagnosis_archive.user_id` értékek, amik már nincsenek a `users.id`-ban
  (törölt felhasználók GDPR purge után). A migration ezeknél leáll
  (`ForeignKeyViolation`) → production deploy blokk.
- javaslat: kétlépcsős "NOT VALID" minta + cleanup:
  ```python
  # 1. orphan rows takarítása (vagy NULL-ozása, ha policy engedi)
  op.execute("""
      DELETE FROM diagnosis_archive
      WHERE user_id NOT IN (SELECT id FROM users)
  """)
  # 2. FK NOT VALID-ként, hogy ne table-lock-oljon
  op.execute("""
      ALTER TABLE diagnosis_archive
      ADD CONSTRAINT diagnosis_archive_user_id_fkey
      FOREIGN KEY (user_id) REFERENCES users(id)
      ON DELETE CASCADE NOT VALID
  """)
  # 3. háttér-validáció
  op.execute("ALTER TABLE diagnosis_archive VALIDATE CONSTRAINT diagnosis_archive_user_id_fkey")
  ```
  Ha a GDPR policy szerint az orphan record megtartandó (archive ≠ live),
  akkor `user_id` legyen `nullable=True` és először `UPDATE … SET user_id = NULL`.

### A2. Index létrehozás táblazár nélkül (Postgres ACCESS EXCLUSIVE)
- severity: **HIGH**
- finding: `019_fix_diagnosis_archive_indexes_and_fk.py:26-35` — `op.create_index`
  alapból blokkoló (`CREATE INDEX`), production `diagnosis_archive` nőhet
  → írás-blokk a migration idejére. A `013/018` is így csinálta (precedens),
  de archive táblánál az insert latency felhasználói flow-t blokkolhat.
- javaslat: `op.create_index(..., postgresql_concurrently=True)` + `with op.get_context().autocommit_block():`
  wrapper (CONCURRENTLY nem futhat tranzakcióban). Vagy explicit deploy-time
  maintenance window dokumentálása.

### A3. Konvenció és névütközés — OK
- severity: **LOW**
- finding: `ix_diagnosis_archive_user_id` / `ix_diagnosis_archive_original_id` /
  `diagnosis_archive_user_id_fkey` — végignéztem `backend/alembic/versions/*.py`,
  ezek a nevek sehol máshol nincsenek definiálva. A `013` csak
  `ix_diagnosis_archive_archived_at`-et hoz létre (`013…py:42`). Nincs ütközés.
  Naming konvenció (`ix_<tábla>_<col>`, `<tábla>_<col>_fkey`) egyezik a `018`-cal.
- javaslat: nincs.

### A4. Downgrade rend — OK, de törékeny
- severity: **LOW**
- finding: `019…py:48-55` — drop sorrend (FK → user_id idx → original_id idx)
  helyes (FK-t kell előbb dobni, mert szülő-indexre támaszkodhat).
  Hiányzik `if_exists` — ha downgrade fél-állapotból fut (pl. upgrade
  megakadt az FK-nál, indexek már megvannak), a `drop_constraint` `ProgrammingError`-t
  dob, downgrade nem fut át.
- javaslat: `op.execute("ALTER TABLE diagnosis_archive DROP CONSTRAINT IF EXISTS diagnosis_archive_user_id_fkey")`
  és `op.execute("DROP INDEX IF EXISTS ix_diagnosis_archive_user_id")`.

## B) Embedding cache backward compat

### B1. Orphan kulcsok — TTL miatt nem kritikus
- severity: **LOW**
- finding: `backend/app/db/redis_cache.py:565-577` — új kulcs salt
  `HUBERT_MODEL@HUBERT_REVISION|` + sha256. A régi formátum `EMBEDDING:<sha(text)>`.
  `CacheTTL.EMBEDDINGS = 3600` (`redis_cache.py:74`) → orphan kulcsok **1 óra**
  alatt maguktól kifutnak. Production hatás: <1 óra magasabb cache miss arány
  (HuBERT újraszámol, ~50-100ms/text MPS-en) → enyhe latency-spike, nem outage.
  35K vector "tényleg cache-elt" száma << 35K, mert TTL=1h. Nincs Redis OOM-kockázat.
- javaslat: opcionálisan deploy után `redis-cli --scan --pattern 'EMBEDDING:*' | xargs redis-cli del`
  a miss-spike rövidítésére, de nem kötelező. Runbook bejegyzés
  `docs/DEPLOYMENT.md`-be: "embedding cache config-változás → 1 óra meleg-cache rebuild".

### B2. Lazy import minden hívásnál
- severity: **LOW**
- finding: `redis_cache.py:573` — `from app.core.config import settings` minden
  `_embedding_cache_key` hívásnál fut. `embedding_service` batch-elt
  hívásnál (`get_embeddings_batch`) listcomp-ban N-szer importálja. Python
  cache-eli a modult, ez ~µs nagyságrendű, de stilisztikailag inkonzisztens
  a fájl többi részével (top-level import a normál minta).
- javaslat: import a fájl tetejére, vagy `functools.lru_cache(None)` egy
  `_active_model_salt()` helper-en.

## C) HUBERT_REVISION default

### C1. `"main"` default + nincs env override példa
- severity: **HIGH**
- finding: `backend/app/core/config.py:156` — `HUBERT_REVISION: str = "main"`.
  HuggingFace `main` branch tip bármikor változhat (modell frissítés,
  tokenizer-csere) → ugyanaz a `text` más vektort generál, Qdrant-ban
  meglévő 35K vektor szemantikus távolsága szétcsúszik → diagnosztikai
  találati arány csendesen romlik. **Nincs HUBERT_REVISION sor** sem
  `.env.example`-ben, sem `.env.railway.example`-ben (csak `HUBERT_MODEL`),
  ezért operátor nem tudja hogy override-olni kell.
- javaslat:
  1. `.env.example:131` és `.env.railway.example:82` után:
     ```
     # Pin to a verified HuggingFace commit hash in production.
     # "main" follows the branch tip and may change embeddings silently.
     HUBERT_REVISION=<sha-from-huggingface>
     ```
  2. `docs/HUNGARIAN_NLP.md` "huBERT" szekcióhoz egy "Model pinning" alszekció:
     verify-commit URL minta + a Qdrant collection version-bump procedúra
     (revision változás → újraindexelés).
  3. `Settings.__post_init__`-ben warning, ha `HUBERT_REVISION == "main"`
     és `ENVIRONMENT == "production"`.

### C2. Cache + Qdrant verzió-drift
- severity: **MEDIUM**
- finding: a B1 fix (salt a cache kulcsban) védi a Redis-t a model-drift
  ellen, **de a Qdrant 35K vektort nem**. Ha valaki revision-t vált
  (vagy `main` magától csúszik), Redis új vektort cache-el, Qdrant a régi
  modell-szerinti vektorokkal hasonlít → torz cosine similarity.
- javaslat: `EMBEDDING_DIMENSION` mellé `EMBEDDING_VERSION_TAG` (model@revision)
  metadata mező a Qdrant collection payload-ban; mismatch esetén reindex job.
  Vagy collection-naming: `embeddings_<hash(model@rev)>`.

## Olvasott fájlok
- `/home/user/AutoCognitix/backend/alembic/versions/019_fix_diagnosis_archive_indexes_and_fk.py`
- `/home/user/AutoCognitix/backend/alembic/versions/018_fix_diagnosis_session_fk_and_expires_index.py`
- `/home/user/AutoCognitix/backend/alembic/versions/013_add_diagnosis_archive_table.py`
- `/home/user/AutoCognitix/backend/app/db/redis_cache.py` (52-77, 550-595 fókusz)
- `/home/user/AutoCognitix/backend/app/core/config.py:140-180`
- `/home/user/AutoCognitix/.env.example` (HUBERT* keresés, csak `HUBERT_MODEL`)
- `/home/user/AutoCognitix/.env.railway.example` (HUBERT* keresés, csak `HUBERT_MODEL`)
- `/home/user/AutoCognitix/docs/HUNGARIAN_NLP.md` (revision-pinning nincs)
- `git diff main...HEAD -- backend/alembic backend/app/db`
- `backend/alembic/versions/` listing (FK + index névütközés ellenőrzés)
