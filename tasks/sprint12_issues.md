## SSE Streaming Logic Audit
**Auditor:** SSE-Logic Specialist
**Date:** 2026-03-29
**Files Reviewed:**
- `backend/app/services/streaming_service.py`
- `backend/app/api/v1/endpoints/diagnosis.py` (streaming route)
- `frontend/src/hooks/useStreamingDiagnosis.ts`
- `frontend/src/services/diagnosisService.ts`

---

### Summary

7 issues found: 2 HIGH, 4 MEDIUM, 1 LOW.

---

### Issue #1 — HIGH: Timeout check only between events, not mid-pipeline

**File:** `backend/app/api/v1/endpoints/diagnosis.py`, lines 1127–1152

**Description:**
The `_timeout_wrapper()` enforces the 300-second deadline by checking `loop.time() > deadline` **between yielded events**. However, the RAG pipeline step (`_run_rag_pipeline`) is a single `await` call with no internal yields. If the LLM takes longer than `STREAM_TIMEOUT_SECONDS`, the timeout check never fires because the pipeline never yields an event during that await. The deadline is only evaluated after `yield event`, meaning a slow LLM call can block indefinitely regardless of the configured timeout.

**Reproducer:** LLM takes > 300 seconds to respond → generator is stuck inside `_run_rag_pipeline`, `_timeout_wrapper` cannot interrupt it since no event is yielded.

**Checklist item:** #5 (Timeout)

---

### Issue #2 — HIGH: `_stream_semaphore` is a module-level global, not event-loop-aware

**File:** `backend/app/api/v1/endpoints/diagnosis.py`, line 61

```python
_stream_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STREAMS)
```

**Description:**
`asyncio.Semaphore` is created at **module import time**, before the event loop exists (or on the wrong event loop in test environments). Under ASGI servers that create new event loops (e.g. Hypercorn, uvicorn with `--workers > 1` + `--loop`), the semaphore is bound to a different loop and raises `RuntimeError: Task attached to different loop`. In uvicorn single-worker mode this works, but it is fragile. The semaphore should be created lazily (e.g. via a dependency or `startup` event).

**Checklist item:** #6 (Concurrent streams)

---

### Issue #3 — MEDIUM: `analysis` event sends `stage`/`message` fields, hook reads `text` field

**File (backend):** `backend/app/api/v1/endpoints/diagnosis.py`, lines 954–999
**File (frontend):** `frontend/src/hooks/useStreamingDiagnosis.ts`, line 73

**Description:**
The backend emits `analysis` events with `data: {"stage": "rag_start", "message": "..."}` and `data: {"stage": "rag_complete", "message": "..."}`. The frontend `onAnalysis` handler reads `eventData.text`:

```typescript
const chunk = typeof eventData.text === 'string' ? eventData.text : ''
```

Since `text` is never present in any `analysis` event payload, `chunk` is always `''`. The conditional `if (chunk)` then prevents any state update. The result: `fullText` is **never populated** from the analysis events, and `chunks` remains empty throughout streaming. Any UI that renders `fullText` to show LLM output will display nothing.

This is also confirmed by `streaming_service.py` (`stream_result_as_chunks`) which uses `chunk` as the field name, not `text` — though that service is not used in the main streaming route.

**Checklist item:** #9 (fullText accumulation)

---

### Issue #4 — MEDIUM: No reconnection logic; frontend has no automatic retry

**File:** `frontend/src/services/diagnosisService.ts`, `frontend/src/hooks/useStreamingDiagnosis.ts`

**Description:**
If the SSE connection drops mid-stream (network flap, server restart, proxy timeout), the frontend `fetch`-based client **silently stops**. The `reader.read()` loop resolves with `done: true` on connection close — no error is raised and no `onError` callback fires. The hook ends in state `{ isStreaming: false, isDone: false, error: null }` — a "stuck" state with no feedback to the user and no reconnect attempt. Standard `EventSource` would automatically reconnect using `Last-Event-ID`; this custom fetch-based implementation has no equivalent mechanism.

**Note:** There is no retry loop risk since there is no reconnection at all, but the user experience consequence is an invisible failure.

**Checklist item:** #4 (Reconnection)

---

### Issue #5 — MEDIUM: `streamDiagnosisGenerator` has a memory/cleanup issue when caller breaks early

**File:** `frontend/src/hooks/useStreamingDiagnosis.ts`, lines 154–223

**Description:**
The `streamDiagnosisGenerator` uses a closure-based queue and a hanging `Promise<void>` to bridge callbacks into an async generator. If the caller breaks out of `for await` early (e.g. component unmounts), the `finally` block calls `controller.abort()` — correct. However, the `streamDiagnosis` callbacks (`onAnalysis`, `onComplete`, `onError`) still hold references to `queue`, `resolve`, and `finished` via closure. After abort, the fetch may still be in-flight for a brief window. When `onAnalysis` fires on the aborted-but-not-yet-cancelled network response, it pushes to `queue` and calls `notify()`, which resolves the orphaned `Promise<void>` — but since the generator has already returned, nobody consumes the queue. The closure objects remain in memory until the next GC cycle. This is not a true leak (GC will collect), but in high-frequency usage the orphaned callbacks accumulate between abort and actual network cancellation.

**Checklist item:** #2 (Memory leak)

---

### Issue #6 — MEDIUM: `complete` event payload is a summary dict, not the full diagnosis result

**File (backend):** `backend/app/api/v1/endpoints/diagnosis.py`, lines 1021–1037
**File (frontend):** `frontend/src/hooks/useStreamingDiagnosis.ts`, lines 83–90

**Description:**
The `onComplete` callback receives `eventData` (the `data` field of the `complete` SSE event) and stores it as `fullResult`:

```typescript
onComplete: (eventData) => {
  setState((prev) => ({ ...prev, fullResult: eventData }))
}
```

The backend `complete` event only sends a summary:
```python
data={
    "diagnosis_id": str(diagnosis_id),
    "confidence_score": ...,
    "urgency_level": ...,
    "probable_causes_count": ...,
    "repairs_count": ...,
    ...
}
```

Any consumer of `fullResult` expecting the full `DiagnosisResponse` structure (with `probable_causes`, `recommended_repairs`, arrays etc.) will find it absent. The structured result is assembled and saved to DB (`_save_diagnosis_session`) but **never sent over the SSE stream**. To get the full result, the consumer must make a separate `GET /diagnosis/{id}` call. This is undocumented and mismatched with the `StreamChunk.full_result` interface exported from the hook.

**Checklist item:** #9 (fullText accumulation / complete event contract)

---

### Issue #7 — LOW: `event:` line in SSE format is ignored by `parseSSEEvents`

**File:** `frontend/src/services/diagnosisService.ts`, lines 279–309

**Description:**
The backend formats events with a named event line:
```
event: analysis
data: {"event_type": "analysis", ...}

```

The `parseSSEEvents` function only extracts `data:` lines and discards the `event:` line entirely. Routing is done via `event.event_type` inside the JSON body — which works correctly in practice. However, if a future event omits `event_type` from the JSON body (relying on the SSE `event:` field), it would be silently dropped. This is a robustness concern rather than a current bug, but it creates inconsistency: the `event:` field in the SSE envelope is written but never read.

**Checklist item:** #7 (SSE format)

---

### Checklist Results

| # | Check | Result | Issue |
|---|-------|--------|-------|
| 1 | Connection cleanup / `CancelledError` | PASS | `asyncio.CancelledError` caught in `generate_events` (line 1101), semaphore released in `finally` |
| 2 | Memory leak on stream abort | MEDIUM | Issue #5 — orphaned callbacks between abort and network cancel |
| 3 | Error propagation backend→SSE→frontend | PASS | Error events formatted and dispatched; `onError` callback triggered |
| 4 | Reconnection logic | MEDIUM | Issue #4 — no reconnect; silent failure on connection drop |
| 5 | Timeout | HIGH | Issue #1 — timeout only fires between events; LLM await not interruptible |
| 6 | Concurrent streams | HIGH | Issue #2 — module-level `asyncio.Semaphore` created before event loop |
| 7 | SSE format | LOW | Issue #7 — `event:` line written but not parsed |
| 8 | AbortController | PASS | `controller.abort()` called in `stopStreaming()` and generator `finally` block |
| 9 | fullText accumulation | MEDIUM | Issue #3 — `analysis` event uses `stage`/`message`; hook reads `text` → always empty |
| 10 | Type safety | MEDIUM | Issue #6 — `complete` event sends summary only; `fullResult` contract broken vs `StreamChunk` |
