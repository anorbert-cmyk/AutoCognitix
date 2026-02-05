/**
 * Services Index
 * Re-exports all services, types, and hooks for easy importing
 */

// API core and types
export {
  api,
  ApiError,
  createLoadingState,
  default,
} from './api'

// Type exports
export type {
  ApiErrorDetail,
  ApiResponse,
  Complaint,
  DiagnosisHistoryItem,
  DiagnosisRequest,
  DiagnosisResponse,
  DTCCategory,
  DTCCategoryInfo,
  DTCCode,
  DTCCodeDetail,
  DTCSearchResult,
  DTCSeverity,
  LoadingState,
  LoginRequest,
  LoginResponse,
  PaginatedResponse,
  ProbableCause,
  QuickAnalyzeResult,
  Recall,
  RefreshTokenRequest,
  RepairRecommendation,
  Source,
  UserResponse,
  VehicleMake,
  VehicleModel,
  VINDecodeRequest,
  VINDecodeResponse,
} from './api'

// DTC Service
export {
  dtcService,
  formatDTCCode,
  getCategoryFromCode,
  getCategoryNameHu,
  getDTCCategories,
  getDTCCodeDetail,
  getRelatedDTCCodes,
  getSeverityColorClass,
  getSeverityLabelHu,
  isValidDTCFormat,
  searchDTCCodes,
} from './dtcService'

export type { DTCSearchParams } from './dtcService'

// Diagnosis Service
export {
  analyzeDiagnosis,
  diagnosisService,
  formatConfidenceScore,
  formatCostRange,
  formatDate,
  formatTime,
  getConfidenceColorClass,
  getConfidenceLevelHu,
  getDiagnosisById,
  getDiagnosisHistory,
  getDifficultyColorClass,
  getDifficultyLabelHu,
  quickAnalyze,
  validateDiagnosisRequest,
} from './diagnosisService'

export type { DiagnosisFormData, HistoryParams } from './diagnosisService'

// Vehicle Service
export {
  decodeVIN,
  formatRecallDate,
  formatVehicleInfo,
  formatVINForDisplay,
  getRecallSeverity,
  getRegionFromVIN,
  getVehicleComplaints,
  getVehicleMakes,
  getVehicleModels,
  getVehicleRecalls,
  getVehicleYears,
  getYearFromVIN,
  validateVIN,
  vehicleService,
} from './vehicleService'

export type { VehicleInfo } from './vehicleService'

// Auth Service
export {
  authService,
  changePassword,
  clearTokens,
  forgotPassword,
  getAccessToken,
  getCurrentUser,
  getRefreshToken,
  isAuthenticated,
  login,
  logout,
  refreshTokens,
  register,
  resetPassword,
  setTokens,
  updateProfile,
} from './authService'

export type {
  AuthTokens,
  ChangePasswordData,
  ForgotPasswordData,
  LoginCredentials,
  RegisterData,
  ResetPasswordData,
  UpdateProfileData,
  User,
} from './authService'

// Hooks
export * from './hooks'
