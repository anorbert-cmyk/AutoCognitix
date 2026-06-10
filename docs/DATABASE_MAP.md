# AutoCognitix - Database Map

Melyik adat melyik adatbázisban van - pontos táblák, node-ok, collection-ok, cache kulcsok. Minden hivatkozás konkrét forrásfájlra.

---

## 1. Összefoglaló

| Adat típus | PostgreSQL | Neo4j | Qdrant | Redis |
|------------|:---:|:---:|:---:|:---:|
| Felhasználók, auth | X | | | |
| Diagnózis session history | X | | | |
| DTC kódok (kanonikus) | X | X | X (embedding) | X (cache) |
| Jármű gyártó/modell/motor | X | X | | X (cache) |
| DTC -> Symptom -> Component -> Repair gráf | | X | X (embeddings) | |
| NHTSA visszahívások és panaszok | X | | | X (cache 6h) |
| Felhasználói garázs, emlékeztetők | X | | | |
| HuBERT embeddingek (768-dim) | | | X | X (1h cache per text) |
| Session / rate limit counter | | | | X |
| LLM response cache | | | | X |

---

## 2. PostgreSQL - strukturált adatok

**Kapcsolódás:** `backend/app/db/postgres/session.py` (async SQLAlchemy 2.0 + asyncpg).
**Modellek:** `backend/app/db/postgres/models.py`.
**Migrációk:** `backend/alembic/versions/` (001-018).

### Auth + felhasználók
| Tábla | Forrás |
|-------|--------|
| `users` | `models.py::User` (38. sor); migration `001_initial_schema.py` + `011_add_user_security_columns.py`. |
| `password_reset_tokens` | `models.py::PasswordResetToken` (78. sor); migration `017_add_password_reset_tokens.py`. |
| `newsletter_subscribers` | `models.py::NewsletterSubscriber` (99. sor); migration `013_newsletter_subscribers.py`. |

### Jármű katalógus
| Tábla | Forrás |
|-------|--------|
| `vehicle_makes` | `models.py::VehicleMake` (119. sor); migration `005_vehicle_schema.py` + seed `009`, `010`. |
| `vehicle_models` | `models.py::VehicleModel` (140. sor); migration `005`. |
| `vehicle_engines` | `models.py::VehicleEngine` (297. sor); migration `005`. |
| `vehicle_platforms` | `models.py::VehiclePlatform` (357. sor); migration `005`. |
| `vehicle_model_engines` (M:N) | `models.py::VehicleModelEngine` (395. sor); migration `005`. |
| `epa_vehicles` | `models.py::EPAVehicle` (729. sor); migration `012_epa_vehicles.py`. |

### DTC kódok + ismert problémák
| Tábla | Forrás |
|-------|--------|
| `dtc_codes` | `models.py::DTCCode` (168. sor); migration `001` + `002_add_dtc_sources_column.py`. |
| `known_issues` | `models.py::KnownIssue` (205. sor); migration `001`. |
| `vehicle_dtc_frequency` | `models.py::VehicleDTCFrequency` (424. sor); migration `005`. |
| `vehicle_tsb` | `models.py::VehicleTSB` (483. sor); migration `005`. |

### Diagnózis + archive
| Tábla | Forrás |
|-------|--------|
| `diagnosis_sessions` | `models.py::DiagnosisSession` (237. sor); migration `001` + `007_soft_delete.py` + `014_add_diagnosis_dedup_index.py` + `018_fix_diagnosis_session_fk_and_expires_index.py`. |
| `diagnosis_archive` | `models.py::DiagnosisArchive` (278. sor); migration `013_add_diagnosis_archive_table.py`. |

### NHTSA adatok
| Tábla | Forrás |
|-------|--------|
| `vehicle_recalls` | `models.py::VehicleRecall` (527. sor); migration `003_vehicle_recalls.py`. |
| `vehicle_complaints` | `models.py::VehicleComplaint` (565. sor); migration `003`. |
| `dtc_recall_correlations` | `models.py::DTCRecallCorrelation` (608. sor); migration `003`. |
| `dtc_complaint_correlations` | `models.py::DTCComplaintCorrelation` (631. sor); migration `003`. |
| `nhtsa_sync_log` | `models.py::NHTSASyncLog` (654. sor); migration `003`. |
| `nhtsa_vehicle_sync_tracking` | `models.py::NHTSAVehicleSyncTracking` (677. sor); migration `006_nhtsa_sync.py`. |

### Garázs (Sprint 9)
| Tábla | Forrás |
|-------|--------|
| `user_vehicles` | `models.py::UserVehicle` (764. sor); migration `016_add_garage_tables.py`. |
| `maintenance_reminders` | `models.py::MaintenanceReminder` (800. sor); migration `016`. |
| `maintenance_costs` | `models.py::MaintenanceCost` (830. sor); migration `016`. |

### Indexek + FK
Teljesítmény indexek: migration `004_perf_indexes.py`. FK constraintek: `008_add_fk_constraints.py`. Head merge: `015_merge_heads.py`.

---

## 3. Neo4j - diagnosztikai gráf

**Kapcsolódás:** `backend/app/db/neo4j_models.py` (Neomodel ORM). Config: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (`backend/app/core/config.py`).
**Seed script:** `scripts/seed_neo4j_aura.py`, `scripts/load_all_to_neo4j.py`, `scripts/expand_neo4j_graph.py`.
**Indexek:** `scripts/neo4j_indexes.cypher`, `scripts/neo4j_schema.cypher`, `scripts/setup_neo4j_indexes.py`.

### Node típusok (StructuredNode)
| Node | Forrás | Kulcs mezők |
|------|--------|-------------|
| `DTCNode` | `neo4j_models.py:133` | `code` (unique), `description_en`, `description_hu`, `category`, `severity`, `system`. |
| `SymptomNode` | `neo4j_models.py:152` | `symptom_id`, `name` (HU), `name_en`, `category`, `severity`, `keywords[]`, `possible_causes[]`. |
| `ComponentNode` | `neo4j_models.py:173` | `name`, `name_hu`, `system`, `part_number`. |
| `RepairNode` | `neo4j_models.py:191` | `name`, `description_hu`, `difficulty`, `estimated_time_minutes`, `estimated_cost_min/max`. |
| `PartNode` | `neo4j_models.py:211` | `name_hu`, `part_number`, `oem_part_number`, `price_min/max`, `currency`. |
| `TestPointNode` | `neo4j_models.py:227` | `name`, `test_type`, `expected_value`, `expected_range_min/max`, `unit`. |
| `VehicleNode` | `neo4j_models.py:245` | `make`, `model`, `year_start/end`, `platform`, `engine_codes[]`, `body_types[]`, `segment`. |
| `EngineNode` | `neo4j_models.py:267` | `code` (unique), `family` (EA888/B58/stb.), `displacement_l`, `fuel_type`, `aspiration`, `power_hp`. |
| `PlatformNode` | `neo4j_models.py:299` | `code` (unique), `name`, `manufacturer`, `segment`, `drivetrain_options[]`. |

### Relációk (StructuredRel)
| Reláció | Forrás | Irány | Tulajdonságok |
|---------|--------|-------|---------------|
| `CAUSES` | `CausesRel` (70. sor) | `DTCNode -> SymptomNode` | `confidence`, `data_source`. |
| `INDICATES_FAILURE_OF` | `IndicatesFailureRel` (77. sor) | `DTCNode -> ComponentNode` | `confidence`, `failure_mode`. |
| `REPAIRED_BY` | `RepairedByRel` (84. sor) | `ComponentNode -> RepairNode` | `difficulty`, `estimated_time_minutes`. |
| `USES_PART` | `UsesPartRel` (91. sor) | `RepairNode -> PartNode` | `quantity`, `optional`. |
| `HAS_COMMON_ISSUE` | `HasCommonIssueRel` (98. sor) | `VehicleNode/EngineNode -> DTCNode` | `frequency`, `year_start/end`, `occurrence_count`, `data_source` (nhtsa/tsb/forum). |
| `COMMON_REPAIR` (`RequiresRepairRel`) | `RequiresRepairRel` (108. sor) | `VehicleNode/EngineNode -> RepairNode` | `confidence`, `is_primary_fix`, `estimated_labor_hours`. |
| `USES_ENGINE` | `UsesEngineRel` (116. sor) | `VehicleNode -> EngineNode` | `year_start/end`, `is_base_engine`, `variant_name`. |
| `SHARES_PLATFORM` | `SharesPlatformRel` (125. sor) | `VehicleNode -> VehicleNode` | `platform_code`. |
| `RELATED_TO` | (anonymous) | `DTCNode -> DTCNode`, `SymptomNode -> SymptomNode` | - |
| `REQUIRES_CHECK` | (anonymous) | `SymptomNode -> TestPointNode` | - |
| `LEADS_TO` | (anonymous) | `TestPointNode -> RepairNode` | - |
| `CONTAINS` | (anonymous) | `ComponentNode -> ComponentNode` (hierarchia) | - |
| `USES_COMPONENT` | (anonymous) | `VehicleNode -> ComponentNode` | - |

Jelenlegi méret (CLAUDE.md szerint): **~26,816 node**.

---

## 4. Qdrant - vektor adatbázis

**Kapcsolódás:** `backend/app/db/qdrant_client.py::QdrantService` (AsyncQdrantClient, grpc/REST). Config: `QDRANT_URL`, `QDRANT_API_KEY` (cloud) vagy `QDRANT_HOST`, `QDRANT_PORT` (local).
**Index scriptek:** `scripts/index_qdrant_hubert.py` (HuBERT, 768-dim), `scripts/index_qdrant.py`, `scripts/index_qdrant_full.py`, `scripts/index_qdrant_robust.py`, `scripts/init_qdrant.py`.

### Collection-ok (mind 768-dim, COSINE distance)
| Collection név | Tartalom | Forrás |
|----------------|---------|--------|
| `dtc_embeddings_hu` | DTC kódok szemantikus embeddingjei (HU leírás + symptoms). | `qdrant_client.py:34` (`DTC_COLLECTION`). |
| `symptom_embeddings_hu` | Panasz/tünet szövegek embeddingjei. | `qdrant_client.py:35` (`SYMPTOM_COLLECTION`). |
| `component_embeddings_hu` | Jármű alkatrész/komponens nevek + leírások. | `qdrant_client.py:36` (`COMPONENT_COLLECTION`). |
| `repair_embeddings_hu` | Javítási eljárások leírásai. | `qdrant_client.py:37` (`REPAIR_COLLECTION`). |
| `known_issue_embeddings_hu` | Ismert problémák (TSB, forum, adatbázis) szövegei. | `qdrant_client.py:38` (`ISSUE_COLLECTION`). |

### Konfiguráció
- **Dimension:** `768` (`QdrantService.EXPECTED_DIMENSION`, `qdrant_client.py:31`).
- **Distance metric:** `COSINE` (`qdrant_models.Distance.COSINE`, `qdrant_client.py:96`).
- **Embedding model:** `hubert-base-cc-v1` (`EMBEDDING_MODEL_VERSION`, `qdrant_client.py:28`). Minden vektor payloadjában ott van `_embedding_model_version` mező - így biztosítható, hogy csak azonos modellel készült vektorokat hasonlítunk össze.
- **Storage alert threshold:** 50,000 vector / collection (`STORAGE_WARN_THRESHOLD`, `qdrant_client.py:41`).
- **Legacy (angol) collections:** `dtc_embeddings`, `symptom_embeddings`, `known_issue_embeddings` - backwards compatibility miatt megmaradtak (`qdrant_client.py:44-46`).
- **Létrehozás:** `_create_collection_if_not_exists()` automatikusan futtatódik a `initialize_collections()` során (`qdrant_client.py:68`).

Jelenlegi méret (CLAUDE.md szerint): **35,000+ vector**.

### Qdrant helper metódusok
- `search_dtc()` - DTC keresés category + severity szűrővel.
- `search_similar_symptoms()` - panasz keresés vehicle_make szűrővel.
- `search_components()` - komponens keresés system szűrővel (engine/transmission/brakes).
- `search_repairs()` - javítás keresés difficulty szűrővel.
- `delete_by_user(user_id)` - GDPR Article 17 cleanup, mind az 5 collection-ból törli a usert.

---

## 5. Redis - cache és rate limiting

**Kapcsolódás:** `backend/app/db/redis_cache.py::RedisCacheService` (singleton, connection pool). Config: `REDIS_URL` (`backend/app/core/config.py`).

### Cache kulcs prefixumok (`CachePrefix`, `redis_cache.py:86`)
| Prefix | Tartalom | TTL | Forrás / callsite |
|--------|---------|-----|--------------------|
| `dtc:code:{CODE}` | DTC kód részletek (JSON). | 1h (`CacheTTL.DTC_CODE`) | `redis_cache.py::get_dtc_code()` / `set_dtc_code()` (453-461. sor). |
| `dtc:search:{sha256}` | DTC keresési eredmények (query+category+limit hash). | 15m (`CacheTTL.DTC_SEARCH`) | `redis_cache.py::get_dtc_search_results()` (463-482). |
| `dtc:related:{CODE}` | Kapcsolódó DTC kódok listája. | 1h | `redis_cache.py:495-503`. |
| `issues:*` | Ismert problémák cache. | 30m (`CacheTTL.KNOWN_ISSUES`) | prefix, `CacheTTL.KNOWN_ISSUES`. |
| `vehicle:make:*`, `vehicle:model:*` | Jármű gyártó/modell adatok. | 24h (`CacheTTL.VEHICLE_DATA`) | prefix, `CacheTTL.VEHICLE_DATA`. |
| `nhtsa:{md5(prefix:args)}` | NHTSA recalls / complaints / VIN decode. Az NHTSA service saját cache backend-et használ (`RedisCache` vagy `InMemoryCache` fallback), kulcs: `_generate_cache_key()` md5 hash a `recalls` / `complaints` / `vin` prefixből + argumentumokból. | VIN: 24h (`VIN_CACHE_TTL`), recalls/complaints: 1h (`RECALLS_CACHE_TTL` / `COMPLAINTS_CACHE_TTL`) | `backend/app/services/nhtsa_service.py::_generate_cache_key()`; használat: `decode_vin()`, `get_recalls()`, `get_complaints()`. |
| `embed:{sha256(text)}` | huBERT embedding vektor (768 float). | 1h (`CacheTTL.EMBEDDINGS`) | `get_embedding()` / `set_embedding()` (563-574). Használat: `backend/app/services/embedding_service.py::embed_text()` + `embed_text_async()`. |
| `ratelimit:{identifier}` | Rate limit counter (atomic Lua INCR+EXPIRE). | ablaktól függ | `check_rate_limit()` (588-628). Callsite-ok: `backend/app/core/rate_limit.py`, `backend/app/core/rate_limiter.py`. Fail-closed policy Sprint 9 óta. |
| `api:diagnosis:{session_id}*`, `api:user:{user_id}:history*` | Diagnózis válasz + user history cache. | 5m (`CacheTTL.API_RESPONSE`) | `redis_cache.py::invalidate_diagnosis_cache()` (224-234). |

### Egyéb Redis funkciók
- **Circuit breaker:** 5 hiba után 30s cooldown (`CIRCUIT_BREAKER_COOLDOWN`, `redis_cache.py:124`). Nyitott circuit esetén minden cache művelet None/False.
- **Connection pool:** max 20 connection, 5s socket timeout (`redis_cache.py:166-173`).
- **Lua script:** atomic INCR + EXPIRE a rate limit race condition elkerülésére (`LUA_INCR_EXPIRE`, `redis_cache.py:39-45`).
- **Statisztika:** `get_stats()` - hit_rate, used_memory, connected_clients (`redis_cache.py:633-652`).

### `@cached` dekorátor
**Forrás:** `redis_cache.py:689-744`.
Service szintű függvényeken használható: automatikus kulcs generálás (SHA256 args hash) + TTL beállítás. Cache miss esetén a függvény fut és az eredmény eltárolódik. Használati példa: lásd `backend/app/services/embedding_service.py`, `llm_provider.py`, `parts_price_service.py`, `rag_service.py`.

---

## 6. Cross-DB konzisztencia

A három domain-DB (Postgres + Neo4j + Qdrant) **szinkronban tartása** sprint 9-ben lett jobban lefedve:
- `scripts/sync_postgres_sprint9.py`, `scripts/sync_neo4j_sprint9.py`, `scripts/sync_qdrant_sprint9.py` - egyszeri sync.
- `backend/app/services/consistency_service.py` - runtime konzisztencia check.
- `rag_service.py::verify_cross_db_consistency()` (1274. sor) - health endpoint számára.

Amennyi `Postgres dtc_codes` rekord van, annyi `DTCNode` kell legyen Neo4j-ben és annyi pontnak a `dtc_embeddings_hu` collection-ban. Eltérés warning-ot generál.
