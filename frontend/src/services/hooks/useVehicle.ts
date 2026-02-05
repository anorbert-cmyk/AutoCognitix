/**
 * Vehicle hooks for React Query integration
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '../api'
import {
  decodeVIN,
  getVehicleComplaints,
  getVehicleMakes,
  getVehicleModels,
  getVehicleRecalls,
  getVehicleYears,
} from '../vehicleService'

// =============================================================================
// Query Keys
// =============================================================================

export const vehicleKeys = {
  all: ['vehicle'] as const,
  makes: (search?: string) => [...vehicleKeys.all, 'makes', search] as const,
  models: (makeId: string, year?: number) => [...vehicleKeys.all, 'models', makeId, year] as const,
  years: () => [...vehicleKeys.all, 'years'] as const,
  vinDecode: (vin: string) => [...vehicleKeys.all, 'vin', vin] as const,
  recalls: (make: string, model: string, year: number) =>
    [...vehicleKeys.all, 'recalls', make, model, year] as const,
  complaints: (make: string, model: string, year: number) =>
    [...vehicleKeys.all, 'complaints', make, model, year] as const,
}

// =============================================================================
// VIN Decoding Hooks
// =============================================================================

/**
 * Hook for decoding VIN (mutation - for form submission)
 */
export function useDecodeVIN() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (vin: string) => decodeVIN(vin),
    onSuccess: (data, vin) => {
      // Cache the result
      queryClient.setQueryData(vehicleKeys.vinDecode(vin), data)
    },
    onError: (error: ApiError) => {
      console.error('VIN decode failed:', error.message)
    },
  })
}

/**
 * Hook for VIN decode with caching (for auto-fill)
 */
export function useVINDecodeQuery(vin: string | undefined) {
  return useQuery({
    queryKey: vehicleKeys.vinDecode(vin || ''),
    queryFn: () => decodeVIN(vin!),
    enabled: !!vin && vin.length === 17,
    staleTime: 60 * 60 * 1000, // 1 hour (VIN data doesn't change)
    retry: (failureCount, error) => {
      // Don't retry on validation errors
      if (error instanceof ApiError && error.status === 400) {
        return false
      }
      return failureCount < 2
    },
  })
}

// =============================================================================
// Makes and Models Hooks
// =============================================================================

/**
 * Hook for fetching vehicle makes
 */
export function useVehicleMakes(search?: string) {
  return useQuery({
    queryKey: vehicleKeys.makes(search),
    queryFn: () => getVehicleMakes(search),
    staleTime: 60 * 60 * 1000, // 1 hour (makes list rarely changes)
  })
}

/**
 * Hook for fetching vehicle models
 */
export function useVehicleModels(makeId: string | undefined, year?: number) {
  return useQuery({
    queryKey: vehicleKeys.models(makeId || '', year),
    queryFn: () => getVehicleModels(makeId!, year),
    enabled: !!makeId && makeId.trim().length > 0,
    staleTime: 60 * 60 * 1000, // 1 hour
  })
}

/**
 * Hook for fetching available years
 */
export function useVehicleYears() {
  return useQuery({
    queryKey: vehicleKeys.years(),
    queryFn: getVehicleYears,
    staleTime: 24 * 60 * 60 * 1000, // 24 hours (only changes once a year)
  })
}

// =============================================================================
// Recalls and Complaints Hooks
// =============================================================================

/**
 * Hook for fetching vehicle recalls
 */
export function useVehicleRecalls(
  make: string | undefined,
  model: string | undefined,
  year: number | undefined
) {
  return useQuery({
    queryKey: vehicleKeys.recalls(make || '', model || '', year || 0),
    queryFn: () => getVehicleRecalls(make!, model!, year!),
    enabled: !!make && !!model && !!year,
    staleTime: 30 * 60 * 1000, // 30 minutes
    retry: (failureCount, error) => {
      // Don't retry on external API errors
      if (error instanceof ApiError && error.status === 502) {
        return false
      }
      return failureCount < 2
    },
  })
}

/**
 * Hook for fetching vehicle complaints
 */
export function useVehicleComplaints(
  make: string | undefined,
  model: string | undefined,
  year: number | undefined
) {
  return useQuery({
    queryKey: vehicleKeys.complaints(make || '', model || '', year || 0),
    queryFn: () => getVehicleComplaints(make!, model!, year!),
    enabled: !!make && !!model && !!year,
    staleTime: 30 * 60 * 1000, // 30 minutes
    retry: (failureCount, error) => {
      // Don't retry on external API errors
      if (error instanceof ApiError && error.status === 502) {
        return false
      }
      return failureCount < 2
    },
  })
}

// =============================================================================
// Prefetch Hooks
// =============================================================================

/**
 * Prefetch vehicle makes for faster initial load
 */
export function usePrefetchVehicleMakes() {
  const queryClient = useQueryClient()

  return () => {
    queryClient.prefetchQuery({
      queryKey: vehicleKeys.makes(),
      queryFn: () => getVehicleMakes(),
      staleTime: 60 * 60 * 1000,
    })
  }
}

/**
 * Prefetch vehicle years for faster initial load
 */
export function usePrefetchVehicleYears() {
  const queryClient = useQueryClient()

  return () => {
    queryClient.prefetchQuery({
      queryKey: vehicleKeys.years(),
      queryFn: getVehicleYears,
      staleTime: 24 * 60 * 60 * 1000,
    })
  }
}
