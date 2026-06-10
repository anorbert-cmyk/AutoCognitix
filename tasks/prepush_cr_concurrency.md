# Pre-Push Audit — Concurrency/Async Lead

Branch: `claude/code-review-fixes` vs `main` | Dátum: 2026-06-10
Scope: `git diff main` — embedding_service pool lifecycle, AnalysisProgress stream race, Sentry before_send hot path.

## Verdikt: NINCS CRITICAL/HIGH — push NEM blokkolt (1 MEDIUM, 4 LOW)

---

## A) `_nlp_pool` lifecycle — embedding_service.py + main.py

**Vizsgálat:**
- `shutdown_thread_pools()` (`backend/app/services/embedding_service.py:856-865`): `_thread_pool.shutdown(wait=True)` majd `_nlp_pool.shutdown(wait=True)`.
- `main.py:164-170` lifespan: try/except köré csomagolva, warning logolással.

**Megállapítások:**

1. **LOW — Hiányzó try/finally a két pool között** (`embedding_service.py:863-864`)
   Ha `_thread_pool.shutdown(wait=True)` kivételt dobna, a `_nlp_pool` nem áll le.
   Gyakorlatban a `ThreadPoolExecutor.shutdown()` nem dob kivételt (a worker
   kivételek a Future-ökben maradnak), és a folyamat amúgy is kilép — ezért nem
   HIGH. Javasolt mégis:
   ```python
   try:
       _thread_pool.shutdown(wait=True)
   finally:
       _nlp_pool.shutdown(wait=True)
   ```
2. **OK — Idempotencia:** `Executor.shutdown()` többszöri hívása biztonságos
   (stdlib garancia), dupla-shutdown nem probléma.
3. **OK — Sorrend:** inference pool (lassú, multi-sec feladatok) előbb, NLP pool
   utána; `wait=True` mindkettőnél — graceful. A lifespan-ben a Neo4j close UTÁN
   fut, de a poolok nem függnek Neo4j-től — sorrend rendben.
4. **OK — `preprocess_hungarian_async` (`:891-908`):** shutdown utáni submit
   RuntimeError-t dob, dokumentálva; a lifespan shutdown a request-fogadás
   leállítása után fut, így élő request nem fut bele normál esetben.

---

## B) AnalysisProgress abort/cleanup race — AnalysisProgress.tsx + DiagnosisPage.tsx

**Vizsgálat:** 2 perces timeout (`AnalysisProgress.tsx:340-353`) most már
`onErrorRef.current?.(...)`-t is hív → `DiagnosisPage.handleStreamingError`
(`DiagnosisPage.tsx:143-176`) fallback `mutateAsync` (120s axios timeout).

**Megállapítások:**

1. **NINCS dupla onError (timeout vs stream):** A timeout handler ELŐBB hívja
   `abortRef.current.abort()`-ot, és a `streamDiagnosis` mindkét await-pontján
   (`diagnosisService.ts:393-396` fetch, `:497-500` read loop) az AbortError
   csendben return-öl, `onError` hívás nélkül. JS single-thread: a timeout
   callback atomikusan lefut, a stream callback nem interleave-elhet. Fordított
   irányban a stream `onError` törli a `streamTimeoutRef`-et (`:311-314`) —
   szintén nincs dupla hívás. OK.
2. **OK — mountedRef védelem:** Minden SSE callback `if (!mountedRef.current)
   return`-nel kezdődik; unmount cleanup (`:372-388`) abort + mindhárom timeout
   törlése. Abort után érkező event nincs, mert a read loop AbortError-ral kilép.
3. **MEDIUM — Timeout-fallback alatt a hiba-UI él marad, Retry/Cancel ütközik a
   fallback hívással** (`AnalysisProgress.tsx:351` + `DiagnosisPage.tsx:143-176`)
   A timeout `onError`-je elindítja a 120s-os blocking `mutateAsync`-ot, de a
   `currentStep` 'analysis' marad → az AnalysisProgress mountolva marad
   hiba-állapotban, Retry és Mégse gombokkal:
   - **Retry:** új stream indul, MIKÖZBEN a fallback fut → két párhuzamos
     diagnózis a backenden; amelyik előbb végez, navigál (a fallback `navigate`
     unmountolja és abortálja a retry streamet — a user folyamatjelzője alól
     "kirántja" az oldalt).
   - **Cancel:** `setCurrentStep('input')` visszalép, de a fallback promise
     tovább él, és sikernél `navigate(/diagnosis/{id})` — felülírja a user
     cancel döntését.
   State-korrupció (setState ütközés) NINCS — a callbackek ref-eken át stabilak,
   és a fallback nem nyúl az AnalysisProgress state-jéhez. UX/duplikált-munka
   race, nem crash → MEDIUM. Javaslat: `handleStreamingError`-ben guard
   (`fallbackInFlightRef`) + a fallback indításakor loading állapot jelzése /
   az AnalysisProgress error-UI Retry letiltása, és cancel-kor a fallback
   eredmény eldobása (cancelled flag).
4. **LOW — A timeout handler nem nullázza `streamTimeoutRef.current`-et**
   (`:340-353`): kisütött timer ID marad benne; a későbbi `clearTimeout`
   no-op, ártalmatlan, de a "cleared = null" invariáns sérül.
5. **LOW — Effect cleanup `mountedRef.current = false` dep-változásra is lefut**
   (`:372`, deps: `[streamingEnabled, diagnosisRequest, startStreaming]`;
   `startStreaming` dep: `diagnosisId`). Ha a `diagnosisId` prop unmount nélkül
   változna, a cleanup false-ra állítja a mountedRef-et, az effect korai
   `streamStartedRef` return miatt nem állítja vissza → minden callback némán
   no-op (UI befagy). Jelenleg `setDiagnosisId` csak közvetlenül `navigate`
   előtt fut (unmount követi), és `diagnosisRequest` stabil ref — nem
   reprodukálható, de törékeny minta. `handleRetry` (`:492`) helyesen
   visszaállítja `mountedRef.current = true`-t.

---

## C) `_sentry_before_send` hot path — logging.py:583-619

**Megállapítások:**

1. **OK — Bounded bejárás:** csak top-level kulcsok: `request.url`,
   `request.query_string`, `extra` top-level str értékei, `message`,
   `logentry.message`. NINCS rekurzió, nested dict/list nem kerül bejárásra —
   a munka mérete korlátos.
2. **OK — Regex költség:** `redact_pii` (`backend/app/core/pii.py:23-32`) két
   előre fordított, lineáris mintát futtat (UUID fix hosszú karakterosztályok,
   VIN `\b[...]{17}\b`) — nincs katasztrofális backtracking, O(n) a string
   hosszában. Tipikus event-méretnél mikroszekundumos.
3. **LOW — Sync futás a hívó (event loop) szálán:** a Sentry `before_send` a
   `capture_event` hívó szálán fut, ami async appban az event loop szál lehet.
   Extrém nagy (több MB-os) extra-stringeknél ez mérhető blokkolást adhat, de a
   Sentry amúgy is vágja az event méretet (~1MB), és a regexek lineárisak —
   gyakorlati kockázat minimális. Nem teendő, csak dokumentált.
4. **Megjegyzés (nem concurrency):** `request.data` (POST body) és nested
   `extra` értékek nem redaktáltak — PII-coverage kérdés, a Security Lead
   hatásköre.

---

## Összegzés

| # | Súlyosság | Hely | Leírás |
|---|-----------|------|--------|
| 1 | MEDIUM | DiagnosisPage.tsx:143-176 + AnalysisProgress.tsx:351 | Timeout-fallback alatt Retry→párhuzamos dupla diagnózis; Cancel-t a fallback navigate felülírja |
| 2 | LOW | embedding_service.py:863-864 | shutdown_thread_pools: nincs try/finally a két pool között |
| 3 | LOW | AnalysisProgress.tsx:340-353 | Timeout handler nem nullázza streamTimeoutRef-et |
| 4 | LOW | AnalysisProgress.tsx:372-389 | mountedRef=false dep-változás cleanup-on is — törékeny minta |
| 5 | LOW | logging.py:583-619 | before_send sync az event loop szálon — lineáris, bounded, elfogadható |

CRITICAL: 0 | HIGH: 0 | MEDIUM: 1 | LOW: 4 → **Push engedélyezett**, a MEDIUM-hoz issue/TODO felveendő.
