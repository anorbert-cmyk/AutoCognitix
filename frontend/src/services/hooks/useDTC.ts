/**
 * DTC hooks for React Query integration
 */

import { useQuery } from '@tanstack/react-query'
import { ApiError, DTCCategory } from '../api'
import {
  getDTCCategories,
  getDTCCodeDetail,
  getRelatedDTCCodes,
  searchDTCCodes,
} from '../dtcService'

// =============================================================================
// Query Keys
// =============================================================================

export const dtcKeys = {
  all: ['dtc'] as const,
  search: (query: string, category?: DTCCategory, make?: string) =>
    [...dtcKeys.all, 'search', { query, category, make }] as const,
  detail: (code: string) => [...dtcKeys.all, 'detail', code] as const,
  related: (code: string) => [...dtcKeys.all, 'related', code] as const,
  categories: () => [...dtcKeys.all, 'categories'] as const,
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook for searching DTC codes
 */
export function useDTCSearch(
  query: string,
  options?: {
    category?: DTCCategory
    make?: string
    limit?: number
    enabled?: boolean
  }
) {
  const { category, make, limit = 20, enabled = true } = options || {}

  return useQuery({
    queryKey: dtcKeys.search(query, category, make),
    queryFn: () => searchDTCCodes({ query, category, make, limit }),
    enabled: enabled && query.trim().length > 0,
    staleTime: 5 * 60 * 1000, // 5 minutes
    placeholderData: (previousData) => previousData, // Keep previous results while loading
  })
}

/**
 * Hook for fetching DTC code details
 */
export function useDTCDetail(code: string | undefined) {
  return useQuery({
    queryKey: dtcKeys.detail(code || ''),
    queryFn: () => getDTCCodeDetail(code!),
    enabled: !!code && code.trim().length > 0,
    staleTime: 30 * 60 * 1000, // 30 minutes (DTC details rarely change)
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
 * Hook for fetching related DTC codes
 */
export function useRelatedDTCCodes(code: string | undefined) {
  return useQuery({
    queryKey: dtcKeys.related(code || ''),
    queryFn: () => getRelatedDTCCodes(code!),
    enabled: !!code && code.trim().length > 0,
    staleTime: 30 * 60 * 1000, // 30 minutes
  })
}

/**
 * Hook for fetching DTC categories
 */
export function useDTCCategories() {
  return useQuery({
    queryKey: dtcKeys.categories(),
    queryFn: getDTCCategories,
    staleTime: 60 * 60 * 1000, // 1 hour (categories never change)
  })
}

/**
 * Hook for debounced DTC search (for autocomplete)
 * Note: debounceMs is not used directly here - debouncing should be implemented at the component level
 * or via a debounce utility before calling this hook.
 */
export function useDebouncedDTCSearch(
  query: string,
  _debounceMs: number = 300,
  options?: {
    category?: DTCCategory
    make?: string
    limit?: number
  }
) {
  // Note: You'll need to implement debouncing at the component level
  // or use a debounce utility. This hook provides the query functionality.
  const { category, make, limit = 10 } = options || {}

  return useQuery({
    queryKey: dtcKeys.search(query, category, make),
    queryFn: () => searchDTCCodes({ query, category, make, limit }),
    enabled: query.trim().length >= 2, // Only search with 2+ characters
    staleTime: 5 * 60 * 1000, // 5 minutes
    placeholderData: (previousData) => previousData,
  })
}
