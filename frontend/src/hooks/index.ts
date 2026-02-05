/**
 * Custom hooks for AutoCognitix frontend
 */

export {
  useErrorHandler,
  useApiMutation,
  isNetworkError,
  isRecoverableError,
  getErrorMessage,
  getErrorCode,
} from './useErrorHandler'

export {
  useVehicleMakes,
  useVehicleModels,
  useVINDecode,
  useVINDecodeQuery,
  usePrefetchMakes,
  usePrefetchModels,
  generateYearOptions,
  vehicleSelectorKeys,
} from './useVehicles'

// Re-export toast hook for convenience
export { useToast } from '../contexts/ToastContext'
