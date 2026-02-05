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
