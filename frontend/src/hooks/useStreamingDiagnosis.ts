/**
 * useStreamingDiagnosis
 *
 * React hook that wraps the SSE streaming diagnosis service.
 * Accumulates incremental text chunks and tracks streaming lifecycle state.
 *
 * Usage:
 *   const { fullText, isStreaming, isDone, error, fullResult, startStreaming, stopStreaming } =
 *     useStreamingDiagnosis();
 *
 *   await startStreaming(formData);
 */

import { useState, useCallback, useRef } from 'react'
import { streamDiagnosis, type DiagnosisFormData } from '../services/diagnosisService'
import type { StreamingEvent } from '../types/streaming'

// =============================================================================
// Types
// =============================================================================

interface StreamingState {
  /** Ordered list of text chunks received so far */
  chunks: string[]
  /** Concatenation of all chunks — suitable for display */
  fullText: string
  /** True while the SSE stream is open */
  isStreaming: boolean
  /** True after the stream ends successfully */
  isDone: boolean
  /** Error message if the stream failed, null otherwise */
  error: string | null
  /** Full structured result delivered in the 'complete' event, null until done */
  fullResult: Record<string, unknown> | null
  /** Progress value 0–1, updated on each SSE event */
  progress: number
}

const INITIAL_STATE: StreamingState = {
  chunks: [],
  fullText: '',
  isStreaming: false,
  isDone: false,
  error: null,
  fullResult: null,
  progress: 0,
}

// =============================================================================
// Hook
// =============================================================================

export function useStreamingDiagnosis() {
  const [state, setState] = useState<StreamingState>(INITIAL_STATE)

  /**
   * AbortController returned by streamDiagnosis — calling .abort() cancels the fetch.
   * Stored in a ref so stopStreaming() can access it without a stale closure.
   */
  const abortRef = useRef<AbortController | null>(null)

  const startStreaming = useCallback(async (data: DiagnosisFormData) => {
    // Reset state before starting a new stream
    setState({ ...INITIAL_STATE, isStreaming: true })

    const controller = streamDiagnosis(data, {
      onStart: () => {
        // Stream has been acknowledged by the server; nothing extra needed
      },

      onAnalysis: (eventData: Record<string, unknown>) => {
        // The analysis event carries the incremental LLM text in the 'text' field
        const chunk = typeof eventData.text === 'string' ? eventData.text : ''
        if (chunk) {
          setState((prev) => ({
            ...prev,
            chunks: [...prev.chunks, chunk],
            fullText: prev.fullText + chunk,
          }))
        }
      },

      onComplete: (eventData: Record<string, unknown>) => {
        setState((prev) => ({
          ...prev,
          isStreaming: false,
          isDone: true,
          fullResult: eventData,
        }))
      },

      onError: (err: Error) => {
        setState((prev) => ({
          ...prev,
          isStreaming: false,
          error: err.message,
        }))
      },

      onProgress: (progress: number) => {
        setState((prev) => ({ ...prev, progress }))
      },
    })

    abortRef.current = controller
  }, [])

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    setState((prev) => ({ ...prev, isStreaming: false }))
  }, [])

  return {
    ...state,
    startStreaming,
    stopStreaming,
  }
}

// =============================================================================
// Async Generator variant
// =============================================================================

/**
 * StreamChunk — the shape yielded by streamDiagnosisGenerator.
 *
 * - While streaming: { chunk: string, done: false, full_result: undefined }
 * - On completion:  { chunk: '',    done: true,  full_result: {...} }
 */
export interface StreamChunk {
  chunk: string
  done: boolean
  full_result?: Record<string, unknown>
}

/**
 * Async generator that wraps the callback-based streamDiagnosis service.
 *
 * Yields StreamChunk objects as the SSE stream progresses.
 * The generator completes (returns) when the 'complete' or 'error' event fires,
 * or when the caller breaks out of the for-await loop.
 *
 * @param data  Diagnosis form data (same as streamDiagnosis)
 * @yields StreamChunk
 *
 * @example
 * const gen = streamDiagnosisGenerator(formData)
 * for await (const event of gen) {
 *   if (event.done) console.log('Full result:', event.full_result)
 *   else process.stdout.write(event.chunk)
 * }
 */
export async function* streamDiagnosisGenerator(
  data: DiagnosisFormData
): AsyncGenerator<StreamChunk, void, unknown> {
  // Bridge the callback-based API to an async generator using a shared queue.
  type QueueItem =
    | { type: 'chunk'; text: string }
    | { type: 'done'; result: Record<string, unknown> }
    | { type: 'error'; message: string }

  const queue: QueueItem[] = []
  let resolve: (() => void) | null = null
  let finished = false

  function notify() {
    if (resolve) {
      const r = resolve
      resolve = null
      r()
    }
  }

  const controller = streamDiagnosis(data, {
    onAnalysis: (eventData: Record<string, unknown>) => {
      const text = typeof eventData.text === 'string' ? eventData.text : ''
      if (text) {
        queue.push({ type: 'chunk', text })
        notify()
      }
    },

    onComplete: (eventData: Record<string, unknown>) => {
      queue.push({ type: 'done', result: eventData })
      finished = true
      notify()
    },

    onError: (err: Error) => {
      queue.push({ type: 'error', message: err.message })
      finished = true
      notify()
    },
  })

  try {
    while (true) {
      // Drain queued items
      while (queue.length > 0) {
        const item = queue.shift()!
        if (item.type === 'chunk') {
          yield { chunk: item.text, done: false }
        } else if (item.type === 'done') {
          yield { chunk: '', done: true, full_result: item.result }
          return
        } else {
          throw new Error(item.message)
        }
      }

      if (finished) break

      // Wait for the next callback to arrive
      await new Promise<void>((r) => {
        resolve = r
      })
    }
  } finally {
    // If the caller breaks early, cancel the in-flight request
    controller.abort()
  }
}

// Re-export DiagnosisFormData so consumers can import it from one place
export type { DiagnosisFormData, StreamingEvent }
