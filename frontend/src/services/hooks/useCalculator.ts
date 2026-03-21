/**
 * Calculator hooks for React Query integration
 */

import { useMutation } from '@tanstack/react-query'
import { ApiError } from '../api'
import {
  evaluateCalculator,
  type CalculatorRequest,
} from '../calculatorService'

// =============================================================================
// Query Keys
// =============================================================================

export const calculatorKeys = {
  all: ['calculator'] as const,
  evaluate: (params?: CalculatorRequest) => [...calculatorKeys.all, 'evaluate', params] as const,
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook for evaluating whether a vehicle is worth repairing.
 * Uses mutation since this is a POST with user-submitted data.
 */
export function useEvaluateCalculator() {
  return useMutation({
    mutationFn: (data: CalculatorRequest) => evaluateCalculator(data),
    onError: (error: ApiError) => {
      console.error('Calculator evaluation failed:', error.message)
    },
  })
}
