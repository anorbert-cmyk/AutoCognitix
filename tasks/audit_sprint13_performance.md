# Sprint 13 Performance Audit

## A) Async/sync mismatch
- **severity: LOW-MEDIUM** (already mostly mitigated via `run_in_executor` + async wrappers)
- **findings:**
  - `backend/app/services/diagnosis_service.py:427` — `await loop.run_in_executor(None, preprocess_hungarian, symptoms)` uses the **default** executor. Hungarian spaCy preprocessing is CPU-heavy (tokenize + lemmatize); on a busy event loop the default `ThreadPoolExecutor` can starve. The dedicated pool `_thread_pool` in `embedding_service.py:57` exists but is not used here. FIX: pass `_thread_pool` explicitly.
  - `backend/app/services/embedding_service.py:330` — sync `embed_text()` (torch forward pass on CPU/GPU) is still called from `embed_text_async()` (line 648) via `run_in_executor`, which is correct, BUT the sync `embed_text` and `embed_batch` remain publicly exported (`embedding_service.py:806, 820`). Risk: callers may invoke the sync variant inside `async def` → event loop blocks 200-800ms per call. RECOMMEND: mark sync variants as `_embed_text_sync` or raise warning when called from running loop.
  - `backend/app/services/diagnosis_service.py:1194-1195` — `await self.diagnosis_repository.create(...)` then `await self.db.flush()` inside the main pipeline. Acceptable, but consider moving save to a background task (`asyncio.create_task`) to shave ~50-150ms off p95 response time since persistence failure is already non-fatal (line 261).

## B) N+1 garage
- **severity: LOW** (no actual N+1 detected in the list path)
- **findings:**
  - `backend/app/services/vehicle_garage_service.py:91-113` — `get_vehicles()` runs **2 separate round trips** (count + paged select) on the same filter. No `selectinload/joinedload` used, but `UserVehicleResponse` (`schemas/garage.py:98-121`) references NO relationships, so no lazy loads fire → **no N+1 in list**. Optimization: combine into `SELECT ..., COUNT(*) OVER() FROM ...` to save 1 round-trip (~5-10 ms per request).
  - `backend/app/api/v1/endpoints/garage.py:169` — `create_vehicle` calls `service.get_vehicles(..., limit=1)` only to read `existing_total`, forcing BOTH a count AND a row fetch. Should call a dedicated `count_vehicles()` method, or reuse a cached count. Extra ~5 ms per create.
  - `backend/app/services/vehicle_garage_service.py:99-102` — Query filters on `(user_id, is_active)` but `UserVehicle` model (`models.py:766, 780`) only has `index` on `user_id` alone, no composite index `(user_id, is_active)`. At low volume fine; at scale a partial/composite index helps.

## C) Frontend bundle
- **severity: OK** (already well optimized)
- **findings:**
  - `frontend/src/App.tsx:12-33` — All 19 page components use `React.lazy(() => import(...))` → proper route-based code splitting. Auth pages, protected pages, demo all split.
  - `frontend/vite.config.ts:61-76` — `manualChunks` groups vendor libs into 7 stable chunks (react, router, query, forms, ui, http, **map**). `vendor-map` (leaflet + react-leaflet, ~150 kB) is separate → only loaded when `/services` is visited. Good.
  - MINOR: `vite.config.ts:49 minify: 'terser'` — terser is slower than the default `esbuild` minifier with near-identical output. Consider switching to `minify: 'esbuild'` to cut build time ~30-50%.
  - MINOR: `App.tsx:3` — `import * as Sentry from '@sentry/react'` is eagerly imported in the entry chunk. Sentry SDK is ~60 kB gzipped. Consider `import('@sentry/react')` lazily or guard with `if (import.meta.env.PROD)`.

## Olvasott fájlok
- /home/user/AutoCognitix/backend/app/services/embedding_service.py (első 150 sor + grep)
- /home/user/AutoCognitix/backend/app/services/diagnosis_service.py (teljes, fókusz analyze_vehicle)
- /home/user/AutoCognitix/backend/app/api/v1/endpoints/garage.py (első 200 sor)
- /home/user/AutoCognitix/backend/app/services/vehicle_garage_service.py (első 200 sor)
- /home/user/AutoCognitix/backend/app/api/v1/schemas/garage.py (részlet)
- /home/user/AutoCognitix/backend/app/db/postgres/models.py (UserVehicle szelet)
- /home/user/AutoCognitix/frontend/vite.config.ts (teljes)
- /home/user/AutoCognitix/frontend/src/App.tsx (teljes)
