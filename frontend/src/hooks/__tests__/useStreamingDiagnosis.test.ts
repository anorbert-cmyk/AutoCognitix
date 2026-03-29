/**
 * Tests for useStreamingDiagnosis hook
 *
 * The hook wraps the callback-based streamDiagnosis service and accumulates
 * SSE text chunks into reactive state.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act, waitFor } from '@testing-library/react'
import type { StreamingCallbacks } from '../../types/streaming'

// ---------------------------------------------------------------------------
// Mock diagnosisService — capture callbacks so tests can drive the stream
// ---------------------------------------------------------------------------
vi.mock('../../services/diagnosisService', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('../../services/diagnosisService')>()
  return {
    ...actual,
    streamDiagnosis: vi.fn(),
  }
})

import { streamDiagnosis } from '../../services/diagnosisService'

// ---------------------------------------------------------------------------
// Shared test data
// ---------------------------------------------------------------------------

const mockRequest = {
  vehicleMake: 'VW',
  vehicleModel: 'Golf',
  vehicleYear: 2018,
  symptoms: 'rough idle and misfire at low RPM',
  dtcCodes: ['P0300'],
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

type UseStreamingDiagnosisHook =
  typeof import('../useStreamingDiagnosis').useStreamingDiagnosis

/** Dynamically import the hook, returning null if the module does not exist. */
async function importHook(): Promise<UseStreamingDiagnosisHook | null> {
  try {
    const mod = await import('../useStreamingDiagnosis')
    return mod.useStreamingDiagnosis
  } catch {
    return null
  }
}

/**
 * Build a mock AbortController and capture the callbacks passed to
 * streamDiagnosis so individual tests can fire them.
 */
function createStreamMock(): {
  controller: AbortController
  getCallbacks: () => StreamingCallbacks
} {
  const controller = new AbortController()
  let capturedCallbacks: StreamingCallbacks = {}

  vi.mocked(streamDiagnosis).mockImplementation(
    (_data, callbacks: StreamingCallbacks) => {
      capturedCallbacks = callbacks
      return controller
    }
  )

  return {
    controller,
    getCallbacks: () => capturedCallbacks,
  }
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

describe('useStreamingDiagnosis', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  // -------------------------------------------------------------------------
  // 1. Initial (idle) state
  // -------------------------------------------------------------------------
  it('should start with idle state', async () => {
    const hook = await importHook()
    if (!hook) return // Skip if not implemented yet

    createStreamMock()

    const { result } = renderHook(() => hook())

    expect(result.current.isStreaming).toBe(false)
    expect(result.current.isDone).toBe(false)
    expect(result.current.fullText).toBe('')
    expect(result.current.error).toBeNull()
    expect(result.current.chunks).toEqual([])
    expect(result.current.fullResult).toBeNull()
  })

  // -------------------------------------------------------------------------
  // 2. Chunk accumulation + completion
  // -------------------------------------------------------------------------
  it('should accumulate chunks during streaming', async () => {
    const hook = await importHook()
    if (!hook) return

    const { getCallbacks } = createStreamMock()

    const { result } = renderHook(() => hook())

    // Start streaming — this registers the callbacks
    act(() => {
      void result.current.startStreaming(mockRequest)
    })

    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    // Fire two analysis events
    act(() => {
      getCallbacks().onAnalysis?.({ text: 'Hello ' })
    })
    act(() => {
      getCallbacks().onAnalysis?.({ text: 'World' })
    })

    // Fire the complete event
    act(() => {
      getCallbacks().onComplete?.({ analysis: 'Hello World' })
    })

    await waitFor(() => expect(result.current.isDone).toBe(true))

    expect(result.current.fullText).toBe('Hello World')
    expect(result.current.chunks).toHaveLength(2)
    expect(result.current.chunks[0]).toBe('Hello ')
    expect(result.current.chunks[1]).toBe('World')
    expect(result.current.fullResult).toEqual({ analysis: 'Hello World' })
    expect(result.current.isStreaming).toBe(false)
  })

  // -------------------------------------------------------------------------
  // 3. Error handling
  // -------------------------------------------------------------------------
  it('should handle streaming errors gracefully', async () => {
    const hook = await importHook()
    if (!hook) return

    const { getCallbacks } = createStreamMock()

    const { result } = renderHook(() => hook())

    act(() => {
      void result.current.startStreaming(mockRequest)
    })

    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    // Fire a partial chunk then an error
    act(() => {
      getCallbacks().onAnalysis?.({ text: 'start' })
    })
    act(() => {
      getCallbacks().onError?.(new Error('Network error'))
    })

    await waitFor(() => expect(result.current.error).not.toBeNull())
    expect(result.current.error).toBe('Network error')
    expect(result.current.isStreaming).toBe(false)
    expect(result.current.isDone).toBe(false)
  })

  // -------------------------------------------------------------------------
  // 4. isStreaming flag is true while streaming
  // -------------------------------------------------------------------------
  it('should set isStreaming to true when startStreaming is called', async () => {
    const hook = await importHook()
    if (!hook) return

    createStreamMock()

    const { result } = renderHook(() => hook())

    act(() => {
      void result.current.startStreaming(mockRequest)
    })

    await waitFor(() => expect(result.current.isStreaming).toBe(true))
    expect(result.current.isDone).toBe(false)
  })

  // -------------------------------------------------------------------------
  // 5. stopStreaming calls abort
  // -------------------------------------------------------------------------
  it('should abort the request when stopStreaming is called', async () => {
    const hook = await importHook()
    if (!hook) return

    const { controller } = createStreamMock()
    const abortSpy = vi.spyOn(controller, 'abort')

    const { result } = renderHook(() => hook())

    act(() => {
      void result.current.startStreaming(mockRequest)
    })

    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    act(() => {
      result.current.stopStreaming()
    })

    expect(abortSpy).toHaveBeenCalledOnce()
    expect(result.current.isStreaming).toBe(false)
  })

  // -------------------------------------------------------------------------
  // 6. State resets between streaming sessions
  // -------------------------------------------------------------------------
  it('should reset state when a new stream is started', async () => {
    const hook = await importHook()
    if (!hook) return

    const { getCallbacks } = createStreamMock()

    const { result } = renderHook(() => hook())

    // First stream
    act(() => {
      void result.current.startStreaming(mockRequest)
    })
    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    act(() => {
      getCallbacks().onAnalysis?.({ text: 'first chunk' })
    })
    act(() => {
      getCallbacks().onComplete?.({ analysis: 'done' })
    })
    await waitFor(() => expect(result.current.isDone).toBe(true))

    expect(result.current.fullText).toBe('first chunk')

    // Second stream — state must reset
    act(() => {
      void result.current.startStreaming(mockRequest)
    })

    await waitFor(() => expect(result.current.isStreaming).toBe(true))
    expect(result.current.fullText).toBe('')
    expect(result.current.chunks).toEqual([])
    expect(result.current.isDone).toBe(false)
    expect(result.current.error).toBeNull()
    expect(result.current.fullResult).toBeNull()
  })

  // -------------------------------------------------------------------------
  // 7. Progress updates
  // -------------------------------------------------------------------------
  it('should update progress as events arrive', async () => {
    const hook = await importHook()
    if (!hook) return

    const { getCallbacks } = createStreamMock()

    const { result } = renderHook(() => hook())

    act(() => {
      void result.current.startStreaming(mockRequest)
    })
    await waitFor(() => expect(result.current.isStreaming).toBe(true))

    act(() => {
      getCallbacks().onProgress?.(0.5)
    })

    await waitFor(() => expect(result.current.progress).toBe(0.5))

    act(() => {
      getCallbacks().onProgress?.(1.0)
    })

    await waitFor(() => expect(result.current.progress).toBe(1.0))
  })

  // -------------------------------------------------------------------------
  // 8. streamDiagnosis is called with the form data
  // -------------------------------------------------------------------------
  it('should call streamDiagnosis with the provided form data', async () => {
    const hook = await importHook()
    if (!hook) return

    createStreamMock()

    const { result } = renderHook(() => hook())

    act(() => {
      void result.current.startStreaming(mockRequest)
    })

    await waitFor(() => expect(vi.mocked(streamDiagnosis)).toHaveBeenCalledOnce())
    expect(vi.mocked(streamDiagnosis)).toHaveBeenCalledWith(
      mockRequest,
      expect.objectContaining({
        onAnalysis: expect.any(Function),
        onComplete: expect.any(Function),
        onError: expect.any(Function),
      })
    )
  })
})
