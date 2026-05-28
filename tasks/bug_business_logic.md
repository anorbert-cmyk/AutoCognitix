# Business Logic & Edge Cases

Audit of `analyze_vehicle()` pipeline – PartsPriceService, DiagnosisService, NHTSAService recall matching.

## A) PartsPriceService

- **severity: HIGH — Static-only data, no live shop integration.**
  - gap: `parts_price_service.py:457-469` — every returned record has `"from_static": True` and `"sources": []`. There is NO Bárdi/Uni/AUTODOC HTTP call anywhere in the file; the "cache stale 24h" question is moot because the system never fetches live data. Frontend `PartStoreCard` (per `CLAUDE.md`) implies bolt-specific pricing exists, but service returns one generic range.
- **severity: HIGH — Discontinued part (0 ár) handling missing.**
  - gap: `parts_price_service.py:441-444` — `STATIC_PARTS_PRICES.get(part_key)` returns None for unknown keys, BUT no check guards `price_min == 0` / `price_max == 0` for "kifutott" parts. `int(0 * multiplier) = 0` flows downstream and `parts_cost_min` aggregates to 0 silently (`parts_price_service.py:566-567`).
- **severity: MEDIUM — Hidden zero-result, caller cannot tell "no parts" from "no DTC mapping".**
  - gap: `parts_price_service.py:506-509` — when `DTC_PARTS_MAPPING` has no entry, returns `[]`. Caller (`diagnosis_service.py:894-896`) treats this identically to a successful lookup: `"No parts found … skipping price enrichment"`, then `total_cost_estimate=None`. User sees blank table with no warning that the DTC is simply outside the mapping table.
- **severity: MEDIUM — `dtc_code` None branch silently degrades.**
  - gap: `parts_price_service.py:554-563` — if both `dtc_code` and `parts` are None, sets `parts=[]`, then computes 0 HUF estimate. Returned cost estimate has total 0 with no error flag.
- **severity: MEDIUM — Vehicle make NOT validated against any DB.**
  - gap: `parts_price_service.py:448-455` — string `make.upper()` whitelisted into 3 multiplier brackets. Typo ("vw" vs "VOLKSWAGEN") → falls through to default 1.0 silently. There is no "vehicle/part not in Bárdi DB" check because there is no Bárdi DB lookup at all.
- **severity: LOW — Cache TTL never invalidated on price drift.**
  - gap: `parts_price_service.py:283-284` — `DEFAULT_TTL = 86400`. Even if live integration is added, the static dict cannot become "stale" — but the cache key in `get_part_price` (`parts_price_service.py:433`) lacks a version/date suffix, so static-data updates require flushing Redis.
- **severity: LOW — Cache-key collision risk.**
  - gap: `parts_price_service.py:433` — `f"part:{part_key}:{None}:{None}:{None}"` is a literal "None" string when vehicle missing. Differing call sites that pass empty string vs None produce different keys for the same logical query.

## B) DiagnosisService pipeline

- **severity: HIGH — 10+ DTC codes: no upper bound, parallel `gather()` could saturate Redis/HTTP.**
  - gap: `diagnosis_service.py:871-880` — `tasks = [service.get_parts_for_dtc(code, …) for code in dtc_codes]` fans out unboundedly. If user pastes 30 codes, 30 cache lookups + 30 STATIC_PARTS_PRICES traversals + 30 dedup runs occur. No `asyncio.Semaphore` / batch cap.
- **severity: HIGH — RAG pipeline failures fully silent fallback.**
  - gap: `diagnosis_service.py:650-656` — bare `except Exception` swallows ANY RAG error (LLM timeout, Qdrant outage, Neo4j down, prompt-template bug). Logs `error` then silently returns `_fallback_diagnosis()` with `used_fallback=True`. UI receives apparently-valid response — only the `used_fallback` flag distinguishes degraded mode. There is no timeout configured on the RAG call itself; it relies on the LLM/Qdrant client's defaults.
- **severity: HIGH — `_enrich_with_parts_prices` swallows all errors.**
  - gap: `diagnosis_service.py:915-917` — `except Exception` → `return {"parts": [], "cost_estimate": None}` with warning log. Customer-facing total cost silently disappears; no flag like `parts_enrichment_failed` on the response.
- **severity: MEDIUM — 0 DTC + symptoms-only path drops into fallback with cosmetic confidence.**
  - gap: `diagnosis_service.py:344-346` + `diagnosis_service.py:820-827` — empty `dtc_codes` → empty `dtc_details` → fallback `confidence = 0.3` baseline + 0.0 added (no DTC, no recall, no complaint). Fallback returns confidence 0.3 with `probable_causes=[]`. Frontend may render a "diagnosis" with zero causes but non-zero confidence number.
- **severity: MEDIUM — NHTSA timeout/503 → silent empty list.**
  - gap: `diagnosis_service.py:490-495` — `asyncio.TimeoutError` returns `([], [])`. `_fetch_nhtsa_data` swallows generic `Exception` at `:493-495` and returns `([], [])`. Pipeline continues, `recalls=[]` so urgency calculation in `_determine_urgency` (`diagnosis_service.py:1129-1136`) never escalates to "high". A car with active critical recall but NHTSA returning 503 may receive "low" urgency.
- **severity: MEDIUM — DB save failure non-blocking.**
  - gap: `diagnosis_service.py:261-267` — `save_ok=False` only flips a `save_error` flag via `model_copy`. User gets a `diagnosis_id` that does NOT exist in DB. Next history fetch shows ghost id; client-side polling breaks.
- **severity: MEDIUM — Save error path: response.id remains the un-persisted UUID.**
  - gap: `diagnosis_service.py:253-267` — Caller has no way to retrieve this diagnosis later. `get_diagnosis_by_id` returns None → user-facing 404.
- **severity: LOW — Critical-recall keyword match in English only for HU output.**
  - gap: `diagnosis_service.py:1131-1135` — keyword set `["crash","fire","injury","death","baleset","tuz"]` mixes EN+HU; NHTSA always English so HU keywords are dead code. Misses "fatal", "collision", "burn".
- **severity: LOW — Symptoms preprocessing failure surfaces raw text.**
  - gap: `diagnosis_service.py:425-430` — broad `except Exception` returns raw text; embedding service may produce poor results, lowering RAG retrieval quality silently.
- **severity: LOW — Duplicate detection skipped when no `dtc_codes`.**
  - gap: `diagnosis_service.py:177` — `if user_id and request.dtc_codes` — symptoms-only resubmissions always re-run RAG/parts/NHTSA, no dedup.

## C) Recall matching

- **severity: HIGH — Matching is exact-string make/model/year, NO VIN, NO engine, NO trim.**
  - gap: `nhtsa_service.py:506-526` — `get_recalls(make, model, year)` builds query `{"make": make, "model": model, "modelYear": year}` against `recallsByVehicle`. VIN is NEVER used for recall lookup (only for `decode_vin`).
  - gap: `diagnosis_service.py:217-221` — `_fetch_nhtsa_data(make=request.vehicle_make, model=request.vehicle_model, year=request.vehicle_year)` uses the request's free-text make/model, not the VIN-decoded canonical name. A user who types "VW" instead of "Volkswagen" gets 0 recalls (NHTSA expects "VOLKSWAGEN"). No normalization layer (`.strip()` only at `nhtsa_service.py:506-507`).
  - **False-positive vector:** every 2018 VW Golf recall returned applies to ALL 1.0/1.4/1.6/2.0 TSI/TDI trims indiscriminately. The 1.4 TSI-specific user sees DSG-7 recalls relevant to the 2.0 TDI variant.
  - **False-negative vector:** sub-trim ("Golf R", "Golf GTI") often a separate NHTSA model record → user typing "Golf" misses them.
- **severity: HIGH — `recalls[:5]` truncation order is API-defined, not relevance-ranked.**
  - gap: `diagnosis_service.py:1005` — `for recall in recalls[:5]` — first 5 from NHTSA response, not first 5 most relevant to the symptom/DTC. A user reporting engine misfire may see 5 airbag recalls instead.
- **severity: MEDIUM — Component string never matched against DTC category.**
  - gap: `diagnosis_service.py:996-1006` — `RelatedRecall` is built blind; `component="Airbag"` recall is attached to a P0300 misfire diagnosis with no filter. The `_fallback_diagnosis` at `:790-799` even auto-promotes recall to `confidence=0.9` probable_cause regardless of relevance to DTC.
- **severity: MEDIUM — Cached recall data keyed without normalization.**
  - gap: `nhtsa_service.py:510` — `_generate_cache_key("recalls", make, model, year)` lowercases via `:322` but does not canonicalize. "VW" / "Volkswagen" / "VOLKSWAGEN" produce 3 distinct cache entries with 3 different result sets.
- **severity: LOW — `recall_date=None` always.**
  - gap: `diagnosis_service.py:1003` — comment says "Not available from NHTSA service" but `nhtsa_service.py:540` already maps `ReportReceivedDate` to `recall.recall_date`. Field is dropped at response build.

Snippet of the exact-string match logic:
```python
# nhtsa_service.py:521-526
url = f"{self.RECALLS_BASE_URL}/recallsByVehicle"
params = {
    "make": make,      # raw user-typed string, only .strip()'d at :506
    "model": model,    # ditto - "Golf" misses "Golf GTI" records
    "modelYear": year, # int; correct
}
```

## Olvasott fájlok
- /home/user/AutoCognitix/backend/app/services/parts_price_service.py (lines 1-200, 200-600)
- /home/user/AutoCognitix/backend/app/services/diagnosis_service.py (full)
- /home/user/AutoCognitix/backend/app/services/nhtsa_service.py (full)
