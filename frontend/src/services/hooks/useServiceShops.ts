/**
 * Service Shop hooks for React Query integration
 */

import { useQuery } from '@tanstack/react-query'
import { ApiError } from '../api'
import {
  getRegions,
  getShopById,
  searchShops,
} from '../serviceShopService'
import type { ServiceSearchParams } from '../serviceShopService'

// =============================================================================
// Query Keys
// =============================================================================

export const serviceShopKeys = {
  all: ['serviceShops'] as const,
  search: (params: ServiceSearchParams) =>
    [...serviceShopKeys.all, 'search', params] as const,
  detail: (id: string) => [...serviceShopKeys.all, 'detail', id] as const,
  regions: () => [...serviceShopKeys.all, 'regions'] as const,
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook for searching service shops with filters
 */
export function useServiceShops(params: ServiceSearchParams) {
  return useQuery({
    queryKey: serviceShopKeys.search(params),
    queryFn: () => searchShops(params),
    staleTime: 5 * 60 * 1000, // 5 minutes
    placeholderData: (previousData) => previousData,
  })
}

/**
 * Hook for fetching available regions
 */
export function useRegions() {
  return useQuery({
    queryKey: serviceShopKeys.regions(),
    queryFn: getRegions,
    staleTime: 60 * 60 * 1000, // 1 hour (regions rarely change)
  })
}

/**
 * Hook for fetching a single shop's details
 */
export function useShopDetail(id: string | undefined) {
  return useQuery({
    queryKey: serviceShopKeys.detail(id || ''),
    queryFn: () => getShopById(id!),
    enabled: !!id && id.trim().length > 0,
    staleTime: 10 * 60 * 1000, // 10 minutes
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status === 404) {
        return false
      }
      return failureCount < 3
    },
  })
}
