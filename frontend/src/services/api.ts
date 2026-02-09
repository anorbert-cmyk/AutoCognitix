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

  static fromAxiosError(error: AxiosError<ApiErrorDetail>): ApiError {
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

    const { status, data } = error.response
    const detail = data?.detail || error.message
    let message = detail

    // Hungarian error messages based on status
    switch (status) {
      case 400:
        message = detail || 'Hibás kérés'
        break
      case 401:
        message = 'Bejelentkezés szükséges'
        break
      case 403:
        message = 'Nincs jogosultság'
        break
      case 404:
        message = detail || 'Az erőforrás nem található'
        break
      case 422:
        message = detail || 'Érvénytelen adatok'
        break
      case 429:
        message = 'Túl sok kérés - kérjük várjon'
        break
      case 500:
        message = 'Szerver hiba - kérjük próbálja újra később'
        break
      case 502:
        message = detail || 'Külső szolgáltatás nem elérhető'
        break
      case 503:
        message = 'A szolgáltatás átmenetileg nem elérhető'
        break
      default:
        message = detail || `Ismeretlen hiba (${status})`
    }

    return new ApiError(message, status, detail, data?.code, data?.field)
  }
}

// =============================================================================
// Request Interceptors
// =============================================================================

// Add auth token to requests
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('access_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
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

api.interceptors.response.use(
  (response: AxiosResponse) => response,
  async (error: AxiosError<ApiErrorDetail>) => {
    const originalRequest = error.config as typeof error.config & { _retry?: boolean }

    // Handle 401 errors (token expired)
    if (error.response?.status === 401 && !originalRequest?._retry) {
      originalRequest._retry = true

      const refreshToken = localStorage.getItem('refresh_token')
      if (refreshToken) {
        try {
          const response = await api.post('/auth/refresh', {
            refresh_token: refreshToken,
          })
          const { access_token, refresh_token: newRefreshToken } = response.data

          localStorage.setItem('access_token', access_token)
          localStorage.setItem('refresh_token', newRefreshToken)

          if (originalRequest) {
            originalRequest.headers.Authorization = `Bearer ${access_token}`
            return api(originalRequest)
          }
        } catch (refreshError) {
          // Refresh failed, clear tokens and redirect to login
          localStorage.removeItem('access_token')
          localStorage.removeItem('refresh_token')
          window.location.href = '/login'
        }
      } else {
        // No refresh token, redirect to login
        localStorage.removeItem('access_token')
        window.location.href = '/login'
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
