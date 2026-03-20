/**
 * React Query hooks for API services
 */

// Diagnosis hooks
export {
  diagnosisKeys,
  useAnalyzeDiagnosis,
  useDeleteDiagnosis,
  useDiagnosisDetail,
  useDiagnosisHistory,
  useDiagnosisStats,
  useQuickAnalyze,
  useQuickAnalyzeQuery,
} from './useDiagnosis'

// DTC hooks
export {
  dtcKeys,
  useDebouncedDTCSearch,
  useDTCCategories,
  useDTCDetail,
  useDTCSearch,
  useRelatedDTCCodes,
} from './useDTC'

// Vehicle hooks
export {
  useDecodeVIN,
  usePrefetchVehicleMakes,
  usePrefetchVehicleYears,
  useVehicleComplaints,
  useVehicleMakes,
  useVehicleModels,
  useVehicleRecalls,
  useVehicleYears,
  useVINDecodeQuery,
  vehicleKeys,
} from './useVehicle'

// Inspection hooks
export {
  inspectionKeys,
  useEvaluateInspection,
} from './useInspection'

// Calculator hooks
export {
  calculatorKeys,
  useEvaluateCalculator,
} from './useCalculator'

// Chat hooks
export { useChat } from './useChat'

// Service Shop hooks
export {
  serviceShopKeys,
  useRegions,
  useServiceShops,
  useShopDetail,
} from './useServiceShops'
