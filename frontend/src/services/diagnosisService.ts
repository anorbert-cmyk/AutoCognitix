/**
 * Diagnosis Service
 * Handles all diagnosis-related API operations
 */

import api, {
  ApiError,
  DiagnosisHistoryItem,
  DiagnosisRequest,
  DiagnosisResponse,
  QuickAnalyzeResult,
} from './api'
import { isValidDTCFormat } from './dtcService'
import type {
  StreamingCallbacks,
  StreamingEvent,
  StreamingEventType,
} from '../types/streaming'

// =============================================================================
// Types
// =============================================================================

export interface DiagnosisFormData {
  vehicleMake: string
  vehicleModel: string
  vehicleYear: number
  vehicleEngine?: string
  vin?: string
  dtcCodes: string[]
  symptoms: string
  additionalContext?: string
}

export interface HistoryParams {
  skip?: number
  limit?: number
  vehicleMake?: string
  vehicleModel?: string
  vehicleYear?: number
  dtcCode?: string
  dateFrom?: string
  dateTo?: string
}

export interface PaginatedHistoryResponse {
  items: DiagnosisHistoryItem[]
  total: number
  skip: number
  limit: number
  has_more: boolean
}

export interface DiagnosisStats {
  total_diagnoses: number
  avg_confidence: number
  most_diagnosed_vehicles: Array<{
    make: string
    model: string
    count: number
  }>
  most_common_dtcs: Array<{
    code: string
    count: number
  }>
  diagnoses_by_month: Array<{
    month: string
    count: number
  }>
}

export interface DeleteResponse {
  success: boolean
  message: string
  deleted_id?: string
}

// =============================================================================
// Validation
// =============================================================================

/**
 * Validate diagnosis request data
 * @param data Form data to validate
 * @returns Validation errors or empty array if valid
 */
export function validateDiagnosisRequest(data: DiagnosisFormData): string[] {
  const errors: string[] = []

  if (!data.vehicleMake || data.vehicleMake.trim().length === 0) {
    errors.push('Gyarto megadasa kotelezo')
  }

  if (!data.vehicleModel || data.vehicleModel.trim().length === 0) {
    errors.push('Modell megadasa kotelezo')
  }

  if (!data.vehicleYear || data.vehicleYear < 1900 || data.vehicleYear > 2030) {
    errors.push('Ervenytelen evjarat')
  }

  if (!data.dtcCodes || data.dtcCodes.length === 0) {
    errors.push('Legalabb egy DTC kod megadasa kotelezo')
  } else if (data.dtcCodes.length > 20) {
    errors.push('Maximum 20 DTC kodot adhat meg')
  } else {
    // Validate each DTC code
    const invalidCodes = data.dtcCodes.filter((code) => !isValidDTCFormat(code))
    if (invalidCodes.length > 0) {
      errors.push(`Ervenytelen DTC kodok: ${invalidCodes.join(', ')}`)
    }
  }

  if (!data.symptoms || data.symptoms.trim().length < 10) {
    errors.push('A tunetleiras legalabb 10 karakter kell legyen')
  } else if (data.symptoms.length > 2000) {
    errors.push('A tunetleiras maximum 2000 karakter lehet')
  }

  if (data.vin && data.vin.length !== 17) {
    errors.push('A VIN pontosan 17 karakter kell legyen')
  }

  return errors
}

// =============================================================================
// Service Functions
// =============================================================================

/**
 * Submit a full diagnosis request
 * @param data Diagnosis request data
 * @returns Diagnosis response with probable causes and recommendations
 * @throws ApiError on request failure or validation error
 */
export async function analyzeDiagnosis(data: DiagnosisFormData): Promise<DiagnosisResponse> {
  // Validate before sending
  const validationErrors = validateDiagnosisRequest(data)
  if (validationErrors.length > 0) {
    throw new ApiError(
      validationErrors.join('. '),
      400,
      validationErrors.join('. '),
      'VALIDATION_ERROR'
    )
  }

  // Transform to API format
  const request: DiagnosisRequest = {
    vehicle_make: data.vehicleMake.trim(),
    vehicle_model: data.vehicleModel.trim(),
    vehicle_year: data.vehicleYear,
    vehicle_engine: data.vehicleEngine?.trim(),
    vin: data.vin?.toUpperCase().trim(),
    dtc_codes: data.dtcCodes.map((code) => code.toUpperCase().trim()),
    symptoms: data.symptoms.trim(),
    additional_context: data.additionalContext?.trim(),
  }

  const response = await api.post<DiagnosisResponse>('/diagnosis/analyze', request)
  return response.data
}

/**
 * Get a specific diagnosis by ID
 * @param id Diagnosis UUID
 * @returns Diagnosis details
 * @throws ApiError on request failure or not found
 */
export async function getDiagnosisById(id: string): Promise<DiagnosisResponse> {
  if (!id || id.trim().length === 0) {
    throw new ApiError('Diagnozis ID megadasa kotelezo', 400, 'Diagnozis ID megadasa kotelezo')
  }

  const response = await api.get<DiagnosisResponse>(`/diagnosis/${id}`)
  return response.data
}

/**
 * Get diagnosis history for the current user with filters
 * @param params Pagination and filter parameters
 * @returns Paginated list of previous diagnoses
 * @throws ApiError on request failure
 */
export async function getDiagnosisHistory(
  params: HistoryParams = {}
): Promise<PaginatedHistoryResponse> {
  const {
    skip = 0,
    limit = 10,
    vehicleMake,
    vehicleModel,
    vehicleYear,
    dtcCode,
    dateFrom,
    dateTo,
  } = params

  const queryParams: Record<string, unknown> = { skip, limit }

  if (vehicleMake) queryParams.vehicle_make = vehicleMake
  if (vehicleModel) queryParams.vehicle_model = vehicleModel
  if (vehicleYear) queryParams.vehicle_year = vehicleYear
  if (dtcCode) queryParams.dtc_code = dtcCode
  if (dateFrom) queryParams.date_from = dateFrom
  if (dateTo) queryParams.date_to = dateTo

  const response = await api.get<PaginatedHistoryResponse>('/diagnosis/history/list', {
    params: queryParams,
  })

  return response.data
}

/**
 * Delete a diagnosis by ID (soft delete)
 * @param id Diagnosis UUID
 * @returns Delete confirmation
 * @throws ApiError on request failure
 */
export async function deleteDiagnosis(id: string): Promise<DeleteResponse> {
  if (!id || id.trim().length === 0) {
    throw new ApiError('Diagnozis ID megadasa kotelezo', 400, 'Diagnozis ID megadasa kotelezo')
  }

  const response = await api.delete<DeleteResponse>(`/diagnosis/${id}`)
  return response.data
}

/**
 * Get diagnosis statistics for the current user
 * @returns Statistics including totals, averages, and trends
 * @throws ApiError on request failure
 */
export async function getDiagnosisStats(): Promise<DiagnosisStats> {
  const response = await api.get<DiagnosisStats>('/diagnosis/stats/summary')
  return response.data
}

/**
 * Quick analyze DTC codes without full vehicle info
 * @param dtcCodes List of DTC codes to analyze
 * @returns Quick analysis results
 * @throws ApiError on request failure
 */
export async function quickAnalyze(dtcCodes: string[]): Promise<QuickAnalyzeResult> {
  if (!dtcCodes || dtcCodes.length === 0) {
    throw new ApiError('Legalabb egy DTC kod megadasa kotelezo', 400)
  }

  if (dtcCodes.length > 10) {
    throw new ApiError('Maximum 10 DTC kod elemzese lehetseges egyszerre', 400)
  }

  // Validate codes
  const normalizedCodes = dtcCodes.map((code) => code.toUpperCase().trim())
  const invalidCodes = normalizedCodes.filter((code) => !isValidDTCFormat(code))
  if (invalidCodes.length > 0) {
    throw new ApiError(`Ervenytelen DTC kodok: ${invalidCodes.join(', ')}`, 400)
  }

  const response = await api.post<QuickAnalyzeResult>('/diagnosis/quick-analyze', null, {
    params: { dtc_codes: normalizedCodes },
  })

  return response.data
}

// =============================================================================
// Streaming Diagnosis
// =============================================================================

/**
 * Parse SSE text buffer into individual events.
 * Handles the "event: {type}\ndata: {json}\n\n" format from the backend.
 */
function parseSSEEvents(buffer: string): { events: StreamingEvent[]; remaining: string } {
  const events: StreamingEvent[] = []
  // Split on double newline (SSE event boundary)
  const parts = buffer.split(/\r?\n\r?\n/)
  // Last part may be incomplete - keep it as remaining
  const remaining = parts.pop() || ''

  for (const part of parts) {
    const trimmed = part.trim()
    if (!trimmed) continue

    // Extract the data line from the SSE event
    let dataLine: string | null = null
    for (const line of trimmed.split('\n')) {
      if (line.startsWith('data: ')) {
        dataLine = line.slice(6)
      }
    }

    if (!dataLine) continue

    try {
      const parsed = JSON.parse(dataLine) as StreamingEvent
      events.push(parsed)
    } catch {
      // Skip malformed JSON lines
    }
  }

  return { events, remaining }
}

/**
 * Stream a diagnosis using Server-Sent Events (SSE).
 *
 * Uses fetch() with POST (not EventSource) because the endpoint requires a JSON body.
 * Reads the response as a ReadableStream and parses SSE events incrementally.
 *
 * @param data - Diagnosis form data (same as analyzeDiagnosis)
 * @param callbacks - Event callbacks for each streaming event type
 * @returns AbortController to cancel the stream
 * @throws ApiError on validation failure (before streaming starts)
 */
export function streamDiagnosis(
  data: DiagnosisFormData,
  callbacks: StreamingCallbacks
): AbortController {
  const controller = new AbortController()

  // Validate before sending
  const validationErrors = validateDiagnosisRequest(data)
  if (validationErrors.length > 0) {
    // Call onError asynchronously to maintain consistent behavior
    queueMicrotask(() => {
      callbacks.onError?.(
        new ApiError(
          validationErrors.join('. '),
          400,
          validationErrors.join('. '),
          'VALIDATION_ERROR'
        )
      )
    })
    return controller
  }

  // Build API base URL (same logic as api.ts)
  const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const url = `${apiBaseUrl}/api/v1/diagnosis/analyze/stream`

  // Build request body matching DiagnosisStreamRequest
  const requestBody = {
    vehicle_make: data.vehicleMake.trim(),
    vehicle_model: data.vehicleModel.trim(),
    vehicle_year: data.vehicleYear,
    vehicle_engine: data.vehicleEngine?.trim() || undefined,
    vin: data.vin?.toUpperCase().trim() || undefined,
    dtc_codes: data.dtcCodes.map((code) => code.toUpperCase().trim()),
    symptoms: data.symptoms.trim(),
    additional_context: data.additionalContext?.trim() || undefined,
    include_context: true,
    include_progress: true,
  }

  // Build headers
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  const token = localStorage.getItem('access_token')
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  // Start streaming in the background
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
          return // User cancelled, no error callback needed
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

      const callbackMap: Record<StreamingEventType, ((data: Record<string, unknown>) => void) | undefined> = {
        start: callbacks.onStart,
        context: callbacks.onContext,
        analysis: callbacks.onAnalysis,
        cause: callbacks.onCause,
        repair: callbacks.onRepair,
        warning: callbacks.onWarning,
        complete: callbacks.onComplete,
        error: undefined, // Handled separately
      }

      let streamDone = false
      while (!streamDone) {
        const { done, value } = await reader.read()
        if (done) {
          streamDone = true
          continue
        }

        buffer += decoder.decode(value, { stream: true })
        const { events, remaining } = parseSSEEvents(buffer)
        buffer = remaining

        for (const event of events) {
          // Handle progress callback
          if (callbacks.onProgress && event.progress != null) {
            const stepName = typeof event.event_type === 'string' ? event.event_type : undefined
            callbacks.onProgress(event.progress, stepName)
          }

          // Handle error events specially
          if (event.event_type === 'error') {
            const errorMsg = (event.data?.message as string) || 'Ismeretlen streaming hiba'
            callbacks.onError?.(new ApiError(errorMsg, 500, errorMsg))
            return
          }

          // Dispatch to typed callback
          const handler = callbackMap[event.event_type]
          if (handler) {
            handler(event.data)
          }
        }
      }

      // Process any remaining buffer content
      if (buffer.trim()) {
        const { events } = parseSSEEvents(buffer + '\n\n')
        for (const event of events) {
          if (callbacks.onProgress && event.progress != null) {
            const stepName = typeof event.event_type === 'string' ? event.event_type : undefined
            callbacks.onProgress(event.progress, stepName)
          }
          if (event.event_type === 'error') {
            const errorMsg = (event.data?.message as string) || 'Ismeretlen streaming hiba'
            callbacks.onError?.(new ApiError(errorMsg, 500, errorMsg))
            return
          }
          const handler = callbackMap[event.event_type]
          if (handler) {
            handler(event.data)
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return // User cancelled
      }
      const apiErr = err instanceof ApiError ? err : new ApiError(
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
// Utility Functions
// =============================================================================

/**
 * Format confidence score for display
 * @param score Confidence score (0-1)
 * @returns Formatted percentage string
 */
export function formatConfidenceScore(score: number): string {
  return `${Math.round(score * 100)}%`
}

/**
 * Get confidence level label in Hungarian
 * @param score Confidence score (0-1)
 * @returns Hungarian label
 */
export function getConfidenceLevelHu(score: number): string {
  if (score >= 0.8) return 'Nagyon magas'
  if (score >= 0.6) return 'Magas'
  if (score >= 0.4) return 'Kozepes'
  if (score >= 0.2) return 'Alacsony'
  return 'Nagyon alacsony'
}

/**
 * Get confidence color class
 * @param score Confidence score (0-1)
 * @returns Tailwind color class
 */
export function getConfidenceColorClass(score: number): string {
  if (score >= 0.8) return 'text-green-600'
  if (score >= 0.6) return 'text-emerald-600'
  if (score >= 0.4) return 'text-yellow-600'
  if (score >= 0.2) return 'text-orange-600'
  return 'text-red-600'
}

/**
 * Get difficulty label in Hungarian
 * @param difficulty Difficulty level
 * @returns Hungarian label
 */
export function getDifficultyLabelHu(
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'professional'
): string {
  switch (difficulty) {
    case 'beginner':
      return 'Kezdo'
    case 'intermediate':
      return 'Kozephalado'
    case 'advanced':
      return 'Halado'
    case 'professional':
      return 'Szakszerviz'
    default:
      return 'Ismeretlen'
  }
}

/**
 * Get difficulty color class
 * @param difficulty Difficulty level
 * @returns Tailwind color class
 */
export function getDifficultyColorClass(
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'professional'
): string {
  switch (difficulty) {
    case 'beginner':
      return 'text-green-600 bg-green-100'
    case 'intermediate':
      return 'text-yellow-600 bg-yellow-100'
    case 'advanced':
      return 'text-orange-600 bg-orange-100'
    case 'professional':
      return 'text-red-600 bg-red-100'
    default:
      return 'text-gray-600 bg-gray-100'
  }
}

/**
 * Format cost range in Hungarian Forint
 * @param min Minimum cost
 * @param max Maximum cost
 * @param currency Currency code (default: HUF)
 * @returns Formatted cost range string
 */
export function formatCostRange(
  min?: number,
  max?: number,
  currency: string = 'HUF'
): string {
  if (!min && !max) {
    return 'Nincs becslés'
  }

  const formatter = new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  })

  if (min && max) {
    return `${formatter.format(min)} - ${formatter.format(max)}`
  }

  if (min) {
    return `${formatter.format(min)}-tol`
  }

  return `${formatter.format(max!)}-ig`
}

/**
 * Format time in minutes to human readable format
 * @param minutes Time in minutes
 * @returns Formatted time string in Hungarian
 */
export function formatTime(minutes?: number): string {
  if (!minutes) {
    return 'Nincs becslés'
  }

  if (minutes < 60) {
    return `${minutes} perc`
  }

  const hours = Math.floor(minutes / 60)
  const remainingMinutes = minutes % 60

  if (remainingMinutes === 0) {
    return `${hours} ora`
  }

  return `${hours} ora ${remainingMinutes} perc`
}

/**
 * Format date for display
 * @param dateString ISO date string
 * @returns Formatted date in Hungarian locale
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString('hu-HU', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// =============================================================================
// Export service object for convenience
// =============================================================================

export const diagnosisService = {
  analyze: analyzeDiagnosis,
  stream: streamDiagnosis,
  getById: getDiagnosisById,
  getHistory: getDiagnosisHistory,
  delete: deleteDiagnosis,
  getStats: getDiagnosisStats,
  quickAnalyze,
  validate: validateDiagnosisRequest,
  formatConfidenceScore,
  getConfidenceLevelHu,
  getConfidenceColorClass,
  getDifficultyLabelHu,
  getDifficultyColorClass,
  formatCostRange,
  formatTime,
  formatDate,
}

export default diagnosisService
