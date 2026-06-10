# Code Review Fixes — `2c28099` felülvizsgálat után (claude/code-review-fixes branch)

## Javított findings (10/10 + 4 levágott)

| # | Finding | Fix | Agent |
|---|---------|-----|-------|
| 1 | Streaming út halott (snake_case/camelCase mismatch) | AnalysisProgress konverzió + direkt import + timeout onError | F1 |
| 2 | `.title()` make-korrupció (McLaren→Mclaren, RAM→Ram) | `vehicle_makes.py`: canonical lookup, ismeretlen pass-through | F2 |
| 3 | cd.yml: migration-success + deploy-failure → nincs rollback | rollback `always()` + 4 esetre bontott issue + schema-drift label | F5 |
| 4 | Make-normalizáció csak NHTSA-ágon | Pydantic field_validator a DiagnosisRequest/StreamRequest-en | F2 |
| 5 | scripts/ indexerek unpinned revision | `HUBERT_REVISION` env + revision= mind a 10 from_pretrained-ben | F4 |
| 6 | Dupla/tripla Sentry event + PII-rés a nem-kézi útvonalakon | `_capture_to_sentry` törölve; egyetlen út: exc_info=True + before_send redakció | F3 |
| 7 | `_VIN_RE` csak uppercase (GDPR leak) | `pii.py` case-insensitive, metrics-mintákkal egyeztetve | F3 |
| 8 | Neo4j/Qdrant/httpx 503 handler-ek Sentry nélkül | exc_info=True minden 5xx handler-en | F3 |
| 9 | `SET lock_timeout` session-leak + 019 hamis premissza + downgrade 013-indexdrop | SET LOCAL + docstring + downgrade fix | F4 |
| 10 | Thread pool head-of-line (preprocess inference mögött) | külön `_nlp_pool` (2 worker) + publikus async wrapper + közös shutdown | F5 |
| 11 | `find() or len()` teszt-bug + vacuous assertions | `_section` helper + behavior-alapú tesztek | F6 |
| 12 | Halott `get_nhtsa_recalls` cache-séma család | törölve (redis_cache.py + tesztek + docs hivatkozás) | F6 |
| 13 | `_expected_key` teszt-helper duplikálta a prod logikát | prod `_embedding_cache_key` hívása a tesztből | F6 |

## Pre-Push audit utáni javítások (2. kör)

| Severity | Finding | Státusz |
|----------|---------|---------|
| HIGH | `before_send` nem redaktálta az exception.values + breadcrumbs mezőket | ✅ javítva (logging.py) |
| MEDIUM | Whitespace-only make átment a validátoron (üres stringgé normalizálva) | ✅ javítva (ValueError) |
| MEDIUM | Kifutott US márkák hiányoztak (Pontiac, Saturn, HUMMER, Mercury, Oldsmobile, Plymouth, Geo, Scion) | ✅ felvéve |
| MEDIUM | rollback job `issues: write` permission hiánya | ✅ javítva (cd.yml) |
| LOW | `shutdown_thread_pools` try/finally | ✅ javítva |
| LOW | `CacheTTL.NHTSA_DATA` halott konstans | ✅ törölve |

## NYITOTT — Pre-Push auditból elhalasztott MEDIUM-ok (Sprint 14 jelölt)

- **History make split-brain (MEDIUM)**: a régi PG history rekordok nyers make-kel ("vw"), az újak kanonikussal ("Volkswagen") mentődnek; a make-szűrés egyik halmazt sem fedi teljesen. Teendő: egyszeri backfill migráció (`UPDATE diagnosis_sessions SET vehicle_make = <canonical>`) a `normalize_make` táblájával + a history filter query param normalizálása (endpoints/diagnosis.py).
- **Retry dupla-diagnózis (MEDIUM)**: AnalysisProgress timeout-fallback alatt a Retry gomb párhuzamos második streamet indíthat, a Cancel-t a fallback navigate felülírhatja. Teendő: fallback-in-flight flag a DiagnosisPage-en.
- **Stream EOF complete/error nélkül (MEDIUM)**: ha az SSE kapcsolat event nélkül zárul, 120s-ig nincs fallback (diagnosisService.ts:496). Teendő: EOF-detektálás → onError hívás.
- **Issue body markdown-injekció tag-névből (LOW)**: cd.yml rollback issue body interpolált tag — sanitize javasolt.
- **`DiagnosisHistoryFilter` dead schema (LOW)**: senki nem használja — törlés vagy bekötés.

## NYITOTT — külön adat-migrációs feladat (Sprint 14 jelölt)

### Qdrant make-filteres symptom keresés MINDIG 0 találat (HIGH)

Az F2 agent vizsgálata szerint:
- `rag_service.py:786` a `symptom_embeddings_hu` kollekcióra `vehicle_make` kulcsú exact `MatchValue` must-filtert tesz
- DE az `index_qdrant_full.py` symptom payload-jai **nem tartalmaznak** `vehicle_make` mezőt (csak symptom_text/related_dtc_codes/source)
- A `sync_qdrant_sprint9.py` `make` kulcsot ír (nem `vehicle_make`), nyers NHTSA all-caps casinggel ("VOLKSWAGEN")

Következmény: a márka-szűrt szemantikus keresés jelenleg üres eredményt ad; a RAG vagy filter nélkül fut, vagy 0 kontextust kap.

Teendő (reindex igényel):
1. Egységes `vehicle_make` payload-kulcs MINDEN kollekcióban, `normalize_make()`-kel kanonizálva
2. A query-oldali filter ugyanazt a kanonikus értéket használja (a schema-validator után már azt kapja)
3. Reindex a 35K vektorra (checkpoint-os batch script)
4. `HUBERT_REVISION` bump-pal együtt ütemezni (egy reindex, két cél)
