/**
 * Chat Service
 * SSE streaming client for the AI Chat Assistant endpoint.
 * Follows the same fetch + ReadableStream pattern as streamDiagnosis.
 */

import { ApiError } from './api'

// =============================================================================
// Types
// =============================================================================

export interface ChatSource {
  type: 'dtc' | 'recall' | 'complaint' | 'tsb' | 'manual' | 'database'
  title: string
  url?: string
  relevance_score?: number
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: ChatSource[]
}

export interface VehicleContext {
  make?: string
  model?: string
  year?: number
  engine?: string
  vin?: string
  dtc_codes?: string[]
}

export interface ChatStreamCallbacks {
  onStart?: () => void
  onToken?: (token: string) => void
  onSource?: (source: ChatSource) => void
  onSuggestion?: (suggestion: string) => void
  onComplete?: (fullContent: string) => void
  onError?: (error: ApiError) => void
}

// =============================================================================
// SSE Parser
// =============================================================================

interface ChatSSEEvent {
  event_type: string
  data: Record<string, unknown>
}

function parseChatSSEEvents(buffer: string): {
  events: ChatSSEEvent[]
  remaining: string
} {
  const events: ChatSSEEvent[] = []
  const parts = buffer.split(/\r?\n\r?\n/)
  const remaining = parts.pop() || ''

  for (const part of parts) {
    const trimmed = part.trim()
    if (!trimmed) continue

    let dataLine: string | null = null
    for (const line of trimmed.split('\n')) {
      if (line.startsWith('data: ')) {
        dataLine = line.slice(6)
      }
    }

    if (!dataLine) continue

    try {
      const parsed = JSON.parse(dataLine) as ChatSSEEvent
      events.push(parsed)
    } catch {
      // Skip malformed JSON lines
    }
  }

  return { events, remaining }
}

// =============================================================================
// Streaming Chat
// =============================================================================

/**
 * Stream a chat message using Server-Sent Events (SSE).
 *
 * Uses fetch() with POST (not EventSource) because the endpoint requires a JSON body.
 * Reads the response as a ReadableStream and parses SSE events incrementally.
 *
 * @param message - User message text
 * @param conversationId - Optional conversation ID for multi-turn context
 * @param vehicleContext - Optional vehicle context for diagnosis-aware chat
 * @param diagnosisId - Optional diagnosis ID to reference a previous diagnosis
 * @param callbacks - Event callbacks for streaming lifecycle
 * @returns AbortController to cancel the stream
 */
export function streamChatMessage(
  message: string,
  conversationId?: string,
  vehicleContext?: VehicleContext,
  diagnosisId?: string,
  callbacks: ChatStreamCallbacks = {}
): AbortController {
  const controller = new AbortController()

  if (!message.trim()) {
    queueMicrotask(() => {
      callbacks.onError?.(
        new ApiError('Az uzenet nem lehet ures', 400, 'Az uzenet nem lehet ures', 'VALIDATION_ERROR')
      )
    })
    return controller
  }

  const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const url = `${apiBaseUrl}/api/v1/chat/message`

  const requestBody: Record<string, unknown> = {
    message: message.trim(),
  }
  if (conversationId) requestBody.conversation_id = conversationId
  if (vehicleContext) requestBody.vehicle_context = vehicleContext
  if (diagnosisId) requestBody.diagnosis_id = diagnosisId

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  const token = localStorage.getItem('access_token')
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const streamPromise = async () => {
    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined

    try {
      let response: Response

      try {
        response = await fetch(url, {
          method: 'POST',
          headers,
          body: JSON.stringify(requestBody),
          signal: controller.signal,
        })
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          return
        }
        callbacks.onError?.(
          new ApiError(
            'Halozati hiba - ellenorizze az internetkapcsolatot',
            0,
            'Halozati hiba',
            'NETWORK_ERROR',
            undefined,
            true
          )
        )
        return
      }

      if (!response.ok) {
        let detail = `Szerver hiba (${response.status})`
        try {
          const errorBody = await response.json()
          if (errorBody?.detail) {
            detail = errorBody.detail
          }
        } catch {
          // Ignore JSON parse errors on error response
        }
        callbacks.onError?.(new ApiError(detail, response.status, detail))
        return
      }

      if (!response.body) {
        callbacks.onError?.(new ApiError('A szerver nem tamogatja a streaminget', 500))
        return
      }

      reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const { events, remaining } = parseChatSSEEvents(buffer)
        buffer = remaining

        for (const event of events) {
          switch (event.event_type) {
            case 'start':
              callbacks.onStart?.()
              break
            case 'token': {
              const token = (event.data?.token as string) || ''
              callbacks.onToken?.(token)
              break
            }
            case 'source': {
              const source = event.data as unknown as ChatSource
              callbacks.onSource?.(source)
              break
            }
            case 'suggestion': {
              const suggestion = (event.data?.text as string) || ''
              callbacks.onSuggestion?.(suggestion)
              break
            }
            case 'complete': {
              const fullContent = (event.data?.content as string) || ''
              callbacks.onComplete?.(fullContent)
              break
            }
            case 'error': {
              const errorMsg =
                (event.data?.message as string) || 'Ismeretlen streaming hiba'
              callbacks.onError?.(new ApiError(errorMsg, 500, errorMsg))
              return
            }
          }
        }
      }

      // Process any remaining buffer content
      if (buffer.trim()) {
        const { events } = parseChatSSEEvents(buffer + '\n\n')
        for (const event of events) {
          if (event.event_type === 'token') {
            callbacks.onToken?.((event.data?.token as string) || '')
          } else if (event.event_type === 'complete') {
            callbacks.onComplete?.((event.data?.content as string) || '')
          } else if (event.event_type === 'error') {
            const errorMsg =
              (event.data?.message as string) || 'Ismeretlen streaming hiba'
            callbacks.onError?.(new ApiError(errorMsg, 500, errorMsg))
            return
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return
      }
      const apiErr =
        err instanceof ApiError
          ? err
          : new ApiError(
              err instanceof Error ? err.message : 'Streaming hiba',
              500,
              err instanceof Error ? err.message : 'Streaming hiba'
            )
      callbacks.onError?.(apiErr)
    } finally {
      if (reader) {
        reader.releaseLock()
      }
    }
  }

  streamPromise()

  return controller
}

// =============================================================================
// Export
// =============================================================================

export const chatService = {
  streamMessage: streamChatMessage,
}

export default chatService
