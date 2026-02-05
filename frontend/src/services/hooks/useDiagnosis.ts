/**
 * Diagnosis hooks for React Query integration
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '../api'
import {
  analyzeDiagnosis,
  deleteDiagnosis,
  DiagnosisFormData,
  getDiagnosisById,
  getDiagnosisHistory,
  getDiagnosisStats,
  HistoryParams,
  quickAnalyze,
} from '../diagnosisService'

// =============================================================================
// Query Keys
// =============================================================================

export const diagnosisKeys = {
  all: ['diagnosis'] as const,
  history: (params?: HistoryParams) => [...diagnosisKeys.all, 'history', params] as const,
  detail: (id: string) => [...diagnosisKeys.all, 'detail', id] as const,
  quickAnalyze: (codes: string[]) => [...diagnosisKeys.all, 'quick', codes] as const,
  stats: () => [...diagnosisKeys.all, 'stats'] as const,
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook for submitting a diagnosis analysis
 */
export function useAnalyzeDiagnosis() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: DiagnosisFormData) => analyzeDiagnosis(data),
    onSuccess: (data) => {
      // Invalidate history to include new diagnosis
      queryClient.invalidateQueries({ queryKey: diagnosisKeys.all })
      // Cache the new diagnosis detail
      queryClient.setQueryData(diagnosisKeys.detail(data.id), data)
    },
    onError: (error: ApiError) => {
      console.error('Diagnosis analysis failed:', error.message)
    },
  })
}

/**
 * Hook for fetching diagnosis by ID
 */
export function useDiagnosisDetail(id: string | undefined) {
  return useQuery({
    queryKey: diagnosisKeys.detail(id || ''),
    queryFn: () => getDiagnosisById(id!),
    enabled: !!id,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: (failureCount, error) => {
      // Don't retry on 404
      if (error instanceof ApiError && error.status === 404) {
        return false
      }
      return failureCount < 3
    },
  })
}

/**
 * Hook for fetching diagnosis history
 */
export function useDiagnosisHistory(params: HistoryParams = {}) {
  return useQuery({
    queryKey: diagnosisKeys.history(params),
    queryFn: () => getDiagnosisHistory(params),
    staleTime: 2 * 60 * 1000, // 2 minutes
  })
}

/**
 * Hook for quick DTC analysis
 */
export function useQuickAnalyze() {
  return useMutation({
    mutationFn: (dtcCodes: string[]) => quickAnalyze(dtcCodes),
    onError: (error: ApiError) => {
      console.error('Quick analysis failed:', error.message)
    },
  })
}

/**
 * Hook for quick DTC analysis with caching
 */
export function useQuickAnalyzeQuery(dtcCodes: string[], enabled = true) {
  return useQuery({
    queryKey: diagnosisKeys.quickAnalyze(dtcCodes),
    queryFn: () => quickAnalyze(dtcCodes),
    enabled: enabled && dtcCodes.length > 0,
    staleTime: 10 * 60 * 1000, // 10 minutes
  })
}

/**
 * Hook for deleting a diagnosis
 */
export function useDeleteDiagnosis() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (id: string) => deleteDiagnosis(id),
    onSuccess: () => {
      // Invalidate history and stats to reflect the deletion
      queryClient.invalidateQueries({ queryKey: diagnosisKeys.all })
    },
    onError: (error: ApiError) => {
      console.error('Diagnosis deletion failed:', error.message)
    },
  })
}

/**
 * Hook for fetching diagnosis statistics
 */
export function useDiagnosisStats() {
  return useQuery({
    queryKey: diagnosisKeys.stats(),
    queryFn: () => getDiagnosisStats(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: (failureCount, error) => {
      // Don't retry on 401 (unauthorized)
      if (error instanceof ApiError && error.status === 401) {
        return false
      }
      return failureCount < 3
    },
  })
}
