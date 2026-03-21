/**
 * Inspection hooks for React Query integration
 */

import { useMutation } from '@tanstack/react-query'
import { ApiError } from '../api'
import { evaluateInspection, InspectionRequest } from '../inspectionService'

// =============================================================================
// Query Keys
// =============================================================================

export const inspectionKeys = {
  all: ['inspection'] as const,
  evaluate: () => [...inspectionKeys.all, 'evaluate'] as const,
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook for evaluating technical inspection risk
 *
 * Usage:
 *   const inspection = useEvaluateInspection()
 *   inspection.mutate(requestData)
 */
export function useEvaluateInspection() {
  return useMutation({
    mutationFn: (data: InspectionRequest) => evaluateInspection(data),
    onError: (error: ApiError) => {
      console.error('Inspection evaluation failed:', error.message)
    },
  })
}
