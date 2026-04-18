# Sprint 13 Data Integrity Audit

## A) Seed idempotencia
- **severity: MEDIUM** (idempotent by check-then-insert; race-prone under concurrency, but seed is single-run)
- **findings:**
  - `scripts/seed_database.py:292-296` — DTC: `session.query(DTCCode).filter_by(code=...).first()` + `continue` if exists. Idempotent sequentially, NO `ON CONFLICT DO NOTHING` — race condition risk if two seed jobs run concurrently (rare).
  - `scripts/seed_database.py:323-327` — `VehicleMake` same pattern (check-then-insert).
  - `scripts/seed_database.py:345-349` — `VehicleModel` same pattern.
  - `scripts/seed_database.py:446-449` — Neo4j DTC node: "Check if node already exists" skip pattern (Cypher MERGE NOT used -> comment says check-then-create, should migrate to `MERGE`).
  - Double-run behavior: would NOT duplicate (primary-key on `code`/`id` uniqueness + explicit skip). Safe for normal operation.
  - Gap: no DB-level `ON CONFLICT DO NOTHING` / `MERGE` makes it fragile under race conditions and increases round-trips (N queries instead of bulk upsert).

## B) Embedding modell pinning
- **severity: HIGH** (model hard-coded without revision pin; cache key lacks model version -> silent poisoning if model upgraded)
- **findings:**
  - `backend/app/core/config.py:152` — `HUBERT_MODEL: str = "SZTAKI-HLT/hubert-base-cc"` hard-coded default, NO `revision=` hash pin passed to `from_pretrained`.
  - `backend/app/services/embedding_service.py:212-221` — `AutoTokenizer.from_pretrained(settings.HUBERT_MODEL, ...)` + `AutoModel.from_pretrained(settings.HUBERT_MODEL, ...)`. No `revision="<commit_sha>"` arg -> HuggingFace pulls latest `main`. If upstream model is updated, embeddings drift silently against 35k Qdrant vectors.
  - `backend/requirements.txt:45-47` — `sentence-transformers==2.3.1`, `transformers==4.37.2`, `torch==2.2.0` pinned (good). But `requirements.prod.txt:48-50` comments them out ("embeddings are pre-computed and stored in Qdrant Cloud") — prod cannot regenerate.
  - `backend/app/db/redis_cache.py:567-574` — **cache key = `sha256(text)` only**. Does NOT include `HUBERT_MODEL` name or revision. Consequence: if model changes, stale embeddings returned from Redis indefinitely (TTL-bound). Should be `f"{model}:{revision}:{sha256(text)}"`.
  - No `pyproject.toml` in `backend/` (dependency sole source: `requirements.txt`).

## C) Backup lefedettség
- **severity: HIGH** (all 3 DBs backed up, restore scripted, but no automated restore test; RTO/RPO undocumented)
- **findings:**
  - `scripts/backup_data.py:11-15` — Targets all: PostgreSQL (pg_dump + Python fallback), Neo4j (Cypher export), Qdrant snapshot, JSON data files. Coverage complete.
  - `scripts/backup_data.py:29-38` — CLI supports `--restore`, `--verify`, `--list`, `--cleanup --keep N`, `--incremental`. Restore IS scripted.
  - `scripts/backup_data.py:79-95` — `BackupState` tracks `last_full_backup`, `last_incremental_backup`, `backup_history` -> incremental strategy works.
  - `docs/BACKUP.md:19-27` — Overview lists all three DBs; emphasizes "All three databases should be backed up together" (consistency).
  - `docs/BACKUP.md:480-490` — Partial recovery documented via `data_sync.py --postgres-neo4j --postgres-qdrant --force`.
  - `docs/BACKUP.md:492-499` — Best practices mention "Test restore procedures monthly" but NO concrete automated restore test in `tests/`.
  - **GAP**: No `RTO` / `RPO` keywords found in `docs/BACKUP.md` (grep: only "Disaster Recovery" heading + "recovery" appears twice). No target recovery objectives documented.
  - **GAP**: No scheduled automated backup verification — `--verify` exists as tool but no cron/CI hook.

## Olvasott fájlok
- /home/user/AutoCognitix/scripts/seed_database.py (1-150, 280-370)
- /home/user/AutoCognitix/backend/app/services/embedding_service.py (1-100, 200-240, 660-695)
- /home/user/AutoCognitix/scripts/backup_data.py (1-100)
- /home/user/AutoCognitix/docs/BACKUP.md (1-50, 470-520)
- /home/user/AutoCognitix/backend/requirements.txt (grep)
- /home/user/AutoCognitix/backend/requirements.prod.txt (grep)
- /home/user/AutoCognitix/backend/app/core/config.py (grep hit l.152)
- /home/user/AutoCognitix/backend/app/db/redis_cache.py (545-595)
