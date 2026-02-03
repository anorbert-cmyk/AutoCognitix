import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth token
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

// Response interceptor for handling errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config

    // Handle 401 errors (token expired)
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true

      const refreshToken = localStorage.getItem('refresh_token')
      if (refreshToken) {
        try {
          const response = await api.post('/auth/refresh', {
            refresh_token: refreshToken,
          })
          const { access_token, refresh_token } = response.data

          localStorage.setItem('access_token', access_token)
          localStorage.setItem('refresh_token', refresh_token)

          originalRequest.headers.Authorization = `Bearer ${access_token}`
          return api(originalRequest)
        } catch (refreshError) {
          // Refresh failed, clear tokens and redirect to login
          localStorage.removeItem('access_token')
          localStorage.removeItem('refresh_token')
          window.location.href = '/login'
        }
      }
    }

    return Promise.reject(error)
  }
)

// API Types
export interface DiagnosisRequest {
  vehicle_make: string
  vehicle_model: string
  vehicle_year: number
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

export interface RepairRecommendation {
  title: string
  description: string
  estimated_cost_min?: number
  estimated_cost_max?: number
  estimated_cost_currency: string
  difficulty: string
  parts_needed: string[]
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
  created_at: string
}

export interface DTCCode {
  code: string
  description_en: string
  description_hu?: string
  category: string
  severity: string
  is_generic: boolean
}

export interface DTCCodeDetail extends DTCCode {
  system?: string
  symptoms: string[]
  possible_causes: string[]
  diagnostic_steps: string[]
  related_codes: string[]
  common_vehicles: string[]
}

export interface VehicleMake {
  id: string
  name: string
  country?: string
}

export interface VehicleModel {
  id: string
  name: string
  make_id: string
  year_start: number
  year_end?: number
}

// API Functions
export const diagnosisApi = {
  analyze: (data: DiagnosisRequest) =>
    api.post<DiagnosisResponse>('/diagnosis/analyze', data),

  getById: (id: string) =>
    api.get<DiagnosisResponse>(`/diagnosis/${id}`),

  getHistory: (skip = 0, limit = 10) =>
    api.get<DiagnosisResponse[]>('/diagnosis/history', { params: { skip, limit } }),
}

export const dtcApi = {
  search: (query: string, category?: string, limit = 20) =>
    api.get<DTCCode[]>('/dtc/search', { params: { q: query, category, limit } }),

  getDetail: (code: string) =>
    api.get<DTCCodeDetail>(`/dtc/${code}`),

  getRelated: (code: string) =>
    api.get<DTCCode[]>(`/dtc/${code}/related`),
}

export const vehicleApi = {
  getMakes: (search?: string) =>
    api.get<VehicleMake[]>('/vehicles/makes', { params: { search } }),

  getModels: (makeId: string, year?: number) =>
    api.get<VehicleModel[]>(`/vehicles/models/${makeId}`, { params: { year } }),

  decodeVin: (vin: string) =>
    api.post('/vehicles/decode-vin', { vin }),

  getYears: () =>
    api.get<{ years: number[] }>('/vehicles/years'),
}

export default api
