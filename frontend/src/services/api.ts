import axios, { AxiosError, AxiosResponse } from 'axios'

// =============================================================================
// API Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
  withCredentials: true, // Send httpOnly cookies with every request
})

// =============================================================================
// Error Types
// =============================================================================

export interface ApiErrorDetail {
  detail: string
  code?: string
  field?: string
}

export class ApiError extends Error {
  status: number
  detail: string
  code?: string
  field?: string
  isNetworkError: boolean

  constructor(
    message: string,
    status: number = 500,
    detail?: string,
    code?: string,
    field?: string,
    isNetworkError = false
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail || message
    this.code = code
    this.field = field
    this.isNetworkError = isNetworkError
  }

  static fromAxiosError(error: AxiosError): ApiError {
    if (!error.response) {
      return new ApiError(
        'Hálózati hiba - ellenőrizze az internetkapcsolatot',
        0,
        'Hálózati hiba - ellenőrizze az internetkapcsolatot',
        'NETWORK_ERROR',
        undefined,
        true
      )
    }

    const { status } = error.response
    const data = error.response.data as Record<string, unknown> | undefined

    // Structured error format: { detail: { error: { code, message, message_hu, details } } }
    const structured = data?.detail as Record<string, unknown> | undefined
    const errorObj = (
      structured !== null && typeof structured === 'object' && !Array.isArray(structured)
        ? structured?.error
        : undefined
    ) as Record<string, unknown> | undefined

    // code and field: check errorObj first, then top-level data
    const code =
      (errorObj?.code as string) ||
      (data?.code as string) ||
      String(status)
    const field = (errorObj?.field as string) || (data?.field as string) || undefined

    // Security-sensitive statuses: always use Hungarian message (never expose server detail)
    const securityOverrides: Record<number, string> = {
      401: 'Bejelentkezés szükséges',
      403: 'Nincs jogosultság',
    }

    const message =
      (errorObj?.message_hu as string) ||
      (errorObj?.message as string) ||
      securityOverrides[status] ||
      (typeof data?.detail === 'string' ? data.detail : null) ||
      error.message

    return new ApiError(message, status, message, code, field)
  }
}

// =============================================================================
// CSRF Token Storage (in-memory only - not accessible to XSS)
// =============================================================================

let csrfToken: string | null = null

export function setCsrfToken(token: string | null) {
  csrfToken = token
}

export function getCsrfToken(): string | null {
  return csrfToken
}

// =============================================================================
// Request Interceptors
// =============================================================================

// Add CSRF token header for state-changing requests (cookies are sent automatically)
api.interceptors.request.use(
  (config) => {
    // Attach CSRF token for state-changing methods
    if (csrfToken && config.method && ['post', 'put', 'patch', 'delete'].includes(config.method)) {
      config.headers['X-CSRF-Token'] = csrfToken
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// =============================================================================
// Response Interceptors
// =============================================================================

let isRefreshing = false
let failedQueue: Array<{ resolve: (value: unknown) => void; reject: (reason?: unknown) => void }> = []

const processQueue = (error: unknown) => {
  failedQueue.forEach(prom => {
    if (error) {
      prom.reject(error)
    } else {
      prom.resolve(undefined)
    }
  })
  failedQueue = []
}

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError<ApiErrorDetail>) => {
    const originalRequest = error.config as typeof error.config & { _retry?: boolean }

    // Handle 401 errors (token expired) - attempt silent refresh via cookie
    if (error.response?.status === 401 && !originalRequest?._retry) {
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject })
        }).then(() => {
          if (originalRequest) {
            return api(originalRequest)
          }
        })
      }

      originalRequest._retry = true
      isRefreshing = true

      try {
        // Refresh token is sent automatically via httpOnly cookie
        const response = await api.post('/auth/refresh')
        const { csrf_token } = response.data

        // Update CSRF token in memory
        if (csrf_token) {
          setCsrfToken(csrf_token)
        }

        processQueue(null)

        // Retry the original request (cookies are updated automatically)
        if (originalRequest) {
          return api(originalRequest)
        }
      } catch (refreshError) {
        processQueue(refreshError)
        // Refresh failed, clear CSRF token and redirect to login
        setCsrfToken(null)
        // Dispatch custom event so AuthContext can nullify user state and
        // redirect via React Router (avoids bypassing the router and stale UI state)
        window.dispatchEvent(new CustomEvent('auth:unauthorized'))
      } finally {
        isRefreshing = false
      }
    }

    // Convert to ApiError for consistent error handling
    throw ApiError.fromAxiosError(error)
  }
)

// =============================================================================
// API Response Types - Diagnosis
// =============================================================================

export interface DiagnosisRequest {
  vehicle_make: string
  vehicle_model: string
  vehicle_year: number
  vehicle_engine?: string
  vin?: string
  dtc_codes: string[]
  symptoms: string
  additional_context?: string
}

export interface ProbableCause {
  title: string
  description: string
  confidence: number
  related_dtc_codes: string[]
  components: string[]
}

export interface ToolNeeded {
  name: string
  icon_hint: string
}

export interface PartWithPrice {
  id: string
  name: string
  name_en?: string
  category: string
  price_range_min: number
  price_range_max: number
  labor_hours: number
  currency: string
}

export interface TotalCostEstimate {
  parts_min: number
  parts_max: number
  labor_min: number
  labor_max: number
  total_min: number
  total_max: number
  currency: string
  estimated_hours: number
  difficulty: string
  disclaimer: string
}

export interface RepairRecommendation {
  title: string
  description: string
  estimated_cost_min?: number
  estimated_cost_max?: number
  estimated_cost_currency: string
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'professional'
  parts_needed: string[]
  estimated_time_minutes?: number
  tools_needed: ToolNeeded[]
  expert_tips: string[]
  root_cause_explanation?: string
}

export interface Source {
  type: 'tsb' | 'forum' | 'video' | 'manual' | 'database'
  title: string
  url?: string
  relevance_score: number
}

export interface RelatedRecall {
  campaign_number: string
  component: string
  summary: string
  consequence?: string
  remedy?: string
  recall_date?: string
}

export interface RelatedComplaint {
  complaint_id?: string
  summary?: string
  incident_date?: string
  severity?: string
}

export interface DiagnosisResponse {
  id: string
  vehicle_make: string
  vehicle_model: string
  vehicle_year: number
  dtc_codes: string[]
  symptoms: string
  probable_causes: ProbableCause[]
  recommended_repairs: RepairRecommendation[]
  confidence_score: number
  sources: Source[]
  created_at: string
  parts_with_prices: PartWithPrice[]
  total_cost_estimate?: TotalCostEstimate
  root_cause_analysis?: string
  related_recalls?: RelatedRecall[]
  similar_complaints?: RelatedComplaint[] | string[]
  // Extended fields
  urgency_level?: string
  safety_warnings?: string[]
  diagnostic_steps?: string[]
  processing_time_ms?: number
  model_used?: string
  save_error?: boolean
  used_fallback?: boolean
  ai_disclaimer?: string  // EU AI Act
}

export interface DiagnosisHistoryItem {
  id: string
  vehicle_make: string
  vehicle_model: string
  vehicle_year: number
  dtc_codes: string[]
  confidence_score: number
  created_at: string
}

export interface QuickAnalyzeResult {
  dtc_codes: Array<{
    code: string
    description: string
    severity: string
    symptoms: string[]
    possible_causes: string[]
  }>
  message: string
}

// =============================================================================
// API Response Types - DTC
// =============================================================================

export type DTCCategory = 'powertrain' | 'body' | 'chassis' | 'network'
export type DTCSeverity = 'low' | 'medium' | 'high' | 'critical'

export interface DTCCode {
  code: string
  description_en: string
  description_hu?: string
  category: DTCCategory
  is_generic: boolean
}

export interface DTCSearchResult extends DTCCode {
  severity: DTCSeverity
  relevance_score?: number
}

export interface DTCCodeDetail extends DTCCode {
  severity: DTCSeverity
  system?: string
  symptoms: string[]
  possible_causes: string[]
  diagnostic_steps: string[]
  related_codes: string[]
  common_vehicles: string[]
  manufacturer_code?: string
  freeze_frame_data?: string[]
}

export interface DTCCategoryInfo {
  code: string
  name: string
  name_hu: string
  description: string
  description_hu: string
}

// =============================================================================
// API Response Types - Vehicle
// =============================================================================

export interface VehicleMake {
  id: string
  name: string
  country?: string
  logo_url?: string
}

export interface VehicleModel {
  id: string
  name: string
  make_id: string
  year_start: number
  year_end?: number
  body_types?: string[]
}

export interface VINDecodeRequest {
  vin: string
}

export interface VINDecodeResponse {
  vin: string
  make: string
  model: string
  year: number
  trim?: string
  engine?: string
  transmission?: string
  drive_type?: string
  body_type?: string
  fuel_type?: string
  region?: string
  country_of_origin?: string
}

export interface Recall {
  campaign_number: string
  report_received_date?: string
  component: string
  summary: string
  consequence?: string
  remedy?: string
  manufacturer?: string
  recall_date?: string
}

export interface Complaint {
  odiNumber: string
  crash: boolean
  fire: boolean
  numberOfInjuries: number
  numberOfDeaths: number
  dateOfIncident?: string
  dateComplaintFiled?: string
  components: string[]
  summary: string
}

// =============================================================================
// API Response Types - Auth
// =============================================================================

export interface LoginRequest {
  email: string
  password: string
}

export interface LoginResponse {
  access_token: string
  refresh_token: string
  token_type: string
  csrf_token?: string
}

export interface RefreshTokenRequest {
  refresh_token: string
}

export interface UserResponse {
  id: string
  email: string
  is_active: boolean
  created_at: string
}

// =============================================================================
// Utility Types
// =============================================================================

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  skip: number
  limit: number
}

export interface ApiResponse<T> {
  data: T
  status: number
}

// =============================================================================
// Loading State Helper
// =============================================================================

export interface LoadingState<T> {
  data: T | null
  isLoading: boolean
  error: ApiError | null
}

export function createLoadingState<T>(): LoadingState<T> {
  return {
    data: null,
    isLoading: false,
    error: null,
  }
}

export default api
