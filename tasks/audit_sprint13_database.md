# Sprint 13 Database Audit

## A) Alembic drift
- **Severity**: MEDIUM
- **Találatok**:
  - `backend/app/db/postgres/models.py:281` vs `backend/alembic/versions/013_add_diagnosis_archive_table.py:29` — `DiagnosisArchive.user_id` a modellben `ForeignKey("users.id", ondelete="CASCADE")`, a migrációban viszont sima `UUID` oszlop, FK constraint NÉLKÜL. Az ORM integritási feltételt vár, a DB nem kényszeríti.
  - `backend/app/db/postgres/models.py:279` vs `backend/alembic/versions/013_add_diagnosis_archive_table.py:28` — `DiagnosisArchive.original_id` a modellben `index=True`, de a migráció csak `sa.Column(..., index=True)`-t ír és Alembic-ben ez NEM hoz létre indexet (explicit `op.create_index` kellene, mint az `archived_at`-nél). Index hiányzik a DB-ben.
  - `backend/app/db/postgres/models.py:281` vs `backend/alembic/versions/013_add_diagnosis_archive_table.py:29` — `DiagnosisArchive.user_id` szintén `index=True` a modellben, de a migráció ugyanezt a csapdát tartalmazza, index nem jön létre.
  - `backend/app/db/postgres/models.py:288` (`dtc_codes: Optional[List[str]] = JSONB`) — szemantikailag is ellentmondásos: a modell Python `List[str]`-t typehintel JSONB-n keresztül, miközben a projekt többi helyén (`DiagnosisSession.dtc_codes`, line 249) `ARRAY(String)`. A 013-as migráció JSONB-t használ, így DB-szinten konzisztens, de inter-táblás query (JOIN/összehasonlítás) típuseltérést fog okozni.
  - `backend/alembic/versions/016_add_garage_tables.py:38` — `is_active` oszlop `nullable=False`, de a default `default=True` csak ORM-szintű; a migrációból hiányzik a `server_default="true"`, ezért nyers SQL INSERT (pl. psql-ből vagy data_sync scriptből) hibára fut. Ugyanez igaz `is_completed`-re (line 68) és `is_deleted`/stb. mintára a többi táblában is.

## B) Cross-DB source of truth
PostgreSQL az egyértelmű source of truth. A `scripts/data_sync.py` kizárólag PostgreSQL -> Neo4j és PostgreSQL -> Qdrant irányban hidratál (`sync_postgres_to_neo4j` line 429, `sync_postgres_to_qdrant` line 521): PG-ből olvassa a DTC kódokat (`get_postgres_dtc_codes`, line 446) és abból hoz létre/frissít Neo4j `DTCNode` node-okat, illetve Qdrant vektor pointokat. Létezik visszairányú `sync_neo4j_to_postgres` (line 653), de a default `--all` pipeline-ban csak a PG->Neo4j és PG->Qdrant fut; a Neo4j->PG külön `--neo4j-postgres` flag mögött van, elsősorban recovery / backfill célra. A Neo4j és Qdrant tehát derived/projected store, PG a kanonikus.

## C) Tranzakció határok
- `backend/app/services/diagnosis_service.py:1194` — `_save_diagnosis_session` hívja `await self.diagnosis_repository.create(session_data)` + `await self.db.flush()`, DE **nem commit-ol**. A commit a `get_db` dependency-ben történik (`backend/app/db/postgres/session.py:130` `await session.commit()` a yield után). Az endpoint (`diagnosis.py:242`) sem commit-ol, tehát a caller/dependency felel. Tipikus Unit-of-Work minta, de ha bármely endpoint megkerüli a `get_db`-t (pl. background task saját sessionnel), a diagnózis soha nem perzisztál.
- `backend/app/services/diagnosis_service.py` — a szolgáltatás más DB-mutáló metódust (delete/update/soft-delete) NEM definiál; a `get_diagnosis_by_id` (1204) és `get_user_history` (1274) csak read-only. Tehát csak EGY service-szintű mutáció van (save), ami flush-only. Nincs `rollback()` a service-ben — hiba esetén (line 1199 except) némán False-t ad vissza, és a request-session a FastAPI dependency-ben gurul vissza.

## Olvasott fájlok
- `/home/user/AutoCognitix/backend/alembic/versions/018_fix_diagnosis_session_fk_and_expires_index.py`
- `/home/user/AutoCognitix/backend/alembic/versions/017_add_password_reset_tokens.py`
- `/home/user/AutoCognitix/backend/alembic/versions/016_add_garage_tables.py`
- `/home/user/AutoCognitix/backend/alembic/versions/013_add_diagnosis_archive_table.py` (user_id FK és index-hiány gyanúra)
- `/home/user/AutoCognitix/backend/alembic/versions/011_add_user_security_columns.py` (cross-check)
- `/home/user/AutoCognitix/backend/app/db/postgres/models.py`
- `/home/user/AutoCognitix/scripts/data_sync.py`
- `/home/user/AutoCognitix/backend/app/services/diagnosis_service.py`
- `/home/user/AutoCognitix/backend/app/api/v1/endpoints/diagnosis.py` (Section 230-300 to confirm no commit)
- `/home/user/AutoCognitix/backend/app/db/postgres/session.py` (get_db commit location)
