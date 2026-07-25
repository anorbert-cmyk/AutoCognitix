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

**A vektorok EGYETLEN, `type`-diszkriminált collectionben vannak. A per-típus `*_hu` collectionök léteznek, de üresek.**

#### A tényleges vektortároló

| Collection név | Tartalom | Forrás |
|----------------|---------|--------|
| **`autocognitix`** | **Minden HuBERT vektor.** A payload `type` mezője diszkriminál: `"dtc"`, `"complaint"`, `"recall"`. Egyéb payload-kulcsok típusonként: DTC-nél `code`/`category`/`severity`, complaint/recall-nál `make`/`model`/`year`/`component` (nyers, all-caps NHTSA értékek). | `settings.QDRANT_UNIFIED_COLLECTION` (`core/config.py`), env-overridable. Indexelő: `scripts/index_qdrant_hubert.py`. |

Runtime hozzáférés: `QdrantService.search_unified(query_vector, type_=...)`, illetve az arra épülő `search_dtc()`. A RAG-út: `rag_service.py::retrieve_from_qdrant(type_="dtc" | "complaint")`.

> **`type = "symptom"` NEM létezik** a collectionben. A RAG "symptom" retrieval-lába ezért `type="complaint"`-re képez - az indexelt NHTSA panasz-narratívák maguk a tünetleírások.

#### Legacy per-típus collectionök (LÉTEZNEK, de ÜRESEK)

| Collection név | Eredeti szándék | Forrás | Valós állapot |
|----------------|-----------------|--------|---------------|
| `dtc_embeddings_hu` | DTC kódok szemantikus embeddingjei. | `QdrantService.DTC_COLLECTION` | üres |
| `symptom_embeddings_hu` | Panasz/tünet szövegek embeddingjei. | `QdrantService.SYMPTOM_COLLECTION` | üres |
| `component_embeddings_hu` | Jármű alkatrész/komponens nevek. | `QdrantService.COMPONENT_COLLECTION` | üres |
| `repair_embeddings_hu` | Javítási eljárások leírásai. | `QdrantService.REPAIR_COLLECTION` | üres |
| `known_issue_embeddings_hu` | Ismert problémák (TSB, forum) szövegei. | `QdrantService.ISSUE_COLLECTION` | üres |

Ezeket a `QdrantService.initialize_collections()` **minden app-induláskor létrehozza**, ha hiányoznak (hívó: `app/main.py` lifespan). Ezért látszanak a Qdrant Cloudon.

> **Történeti megjegyzés:** a `/diagnosis/analyze` RAG-ja hónapokig ezeket az ÜRES collectionöket kérdezte, miközben a vektorok az `autocognitix`-ban voltak - és mivel a hiba üres találatlistaként jelentkezett, semmi nem jelezte. Javítva: `c64dcf1`. A teljes történet: `docs/EMBEDDING_ARCHITECTURE_DECISION.md`.

### Konfiguráció
- **Dimension:** `768` (`QdrantService.EXPECTED_DIMENSION`, `qdrant_client.py:31`).
- **Distance metric:** `COSINE` (`qdrant_models.Distance.COSINE`, `qdrant_client.py:96`).
- **Embedding modell:** `SZTAKI-HLT/hubert-base-cc`, **SHA-ra pinelve** (`HUBERT_REVISION`, `core/config.py`). Production inference: **ONNX Runtime fp32** (`EMBEDDING_BACKEND=onnx`), dev/indexelés: torch.
- **`_embedding_model_version` payload:** az `upsert_vectors()` minden általa írt payloadba beleteszi (`EMBEDDING_MODEL_VERSION = "hubert-base-cc-v1"`). **DE az `autocognitix` collection pontjai NEM hordozzák** - más indexelő úton készültek. Ezért a unified keresésnél a `model_version` szűrőt **tilos** átadni, különben mindent kizárna.
- **Storage alert threshold:** 50,000 vector / collection (`STORAGE_WARN_THRESHOLD`). Figyelem: a `get_storage_stats()` / `check_storage_alerts()` **csak az öt legacy collectiont nézi**, tehát a valódi vektortároló (`autocognitix`) méretét ma nem monitorozza.
- **Degenerált query-vektor guard:** a `search()` `ValueError`-t dob üres vagy `norm < 1e-6` query vektorra (`_validate_query_vector`, `MIN_QUERY_VECTOR_NORM`). Cosine távolságnál a nullvektor nem rossz, hanem **értelmetlen** találatokat ad - ez a hiba rejtette hónapokig a törött embedding utat.
- **Legacy (angol) collections:** `dtc_embeddings`, `symptom_embeddings`, `known_issue_embeddings` - konstansként megmaradtak, de nem jönnek létre és nem kérdezi őket semmi.
- **Létrehozás:** `_create_collection_if_not_exists()` automatikusan futtatódik az `initialize_collections()` során. **Az `autocognitix` collectiont NEM ez hozza létre**, hanem az indexelő script.

Jelenlegi méret: lásd a `CLAUDE.md` "Aktuális Adatbázis Állapot" tábláját - **a repó forrásai ellentmondanak egymásnak**, és ez ott dokumentálva van.

### Qdrant helper metódusok

**Aktívan használt:**
- `search_unified(query_vector, type_, ...)` - a unified `autocognitix` collection keresése `type` diszkriminátorral. A `type` szűrő **utoljára** kerül be, így hívói `extra_filters` nem tudja felülírni.
- `search_dtc()` - DTC keresés; delegál a `search_unified(type_="dtc")`-re, opcionális category + severity szűrővel.
- `search()` - alacsony szintű keresés tetszőleges collectionre (guarddal).

**Halott kód - a legacy ÜRES collectionökre mutat, nincs hívója az `app/`-ban és a `scripts/`-ben:**
- `search_similar_symptoms()` - `symptom_embeddings_hu`, `vehicle_make` szűrővel.
- `search_components()` - `component_embeddings_hu`.
- `search_repairs()` - `repair_embeddings_hu`.

**GDPR (Article 17):**
- `delete_by_user(user_id)` - **elsőként a unified `autocognitix` collectiont** törli, majd azokat a legacy collectionöket, amik **ténylegesen léteznek** (`_legacy_collections_present()` előzetes probe). Korábban csak a legacy (üres) collectionöket söpörte, és minden hibát elnyelt - így a törlés **semmit nem törölt, miközben sikert jelentett**. Ma **minden hiba propagál** (`QdrantException`), hogy a `DELETE /api/v1/auth/me` részleges-törlés hibát tudjon jelenteni a felhasználónak.

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

Amennyi `Postgres dtc_codes` rekord van, annyi `DTCNode` kell legyen Neo4j-ben, és annyi `type="dtc"` pontnak a unified **`autocognitix`** collectionben. Eltérés warning-ot generál.

> **Történeti csapda:** a `consistency_service` korábban az ÜRES `dtc_embeddings_hu` collectiont számolta, ezért permanensen 0 Qdrant vektort jelentett és hamis inkonzisztenciát kiáltott — **a drift-detektor maga is a drift áldozata volt**. Javítva (`62158b3`): a unified collectiont számolja `type` szűrővel, és egy hiányzó collection valódi hibaként jelenik meg, nem csendes nullaként.
