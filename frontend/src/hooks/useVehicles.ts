/**
 * Vehicle hooks for the VehicleSelector component
 * Uses TanStack Query for caching and state management
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError, VehicleMake, VehicleModel, VINDecodeResponse } from '../services/api'
import {
  decodeVIN,
  getVehicleMakes,
  getVehicleModels,
} from '../services/vehicleService'
import type { VINDecodeResult } from '../types/vehicle'

// =============================================================================
// Query Keys
// =============================================================================

export const vehicleSelectorKeys = {
  all: ['vehicleSelector'] as const,
  makes: (search?: string) => [...vehicleSelectorKeys.all, 'makes', search] as const,
  models: (makeId: string, year?: number) => [...vehicleSelectorKeys.all, 'models', makeId, year] as const,
  vinDecode: (vin: string) => [...vehicleSelectorKeys.all, 'vin', vin] as const,
}

// =============================================================================
// useVehicleMakes - Fetch all vehicle makes
// =============================================================================

interface UseVehicleMakesOptions {
  search?: string
  enabled?: boolean
}

/**
 * Hook for fetching vehicle makes with optional search filtering
 * @param options - Configuration options
 * @returns Query result with makes data
 */
export function useVehicleMakes(options: UseVehicleMakesOptions = {}) {
  const { search, enabled = true } = options

  return useQuery<VehicleMake[], ApiError>({
    queryKey: vehicleSelectorKeys.makes(search),
    queryFn: () => getVehicleMakes(search),
    enabled,
    staleTime: 60 * 60 * 1000, // 1 hour - makes list rarely changes
    gcTime: 24 * 60 * 60 * 1000, // 24 hours cache
    retry: 2,
    refetchOnWindowFocus: false,
  })
}

// =============================================================================
// useVehicleModels - Fetch models for a specific make
// =============================================================================

interface UseVehicleModelsOptions {
  makeId: string | undefined
  year?: number
  enabled?: boolean
}

/**
 * Hook for fetching vehicle models for a specific make
 * @param options - Configuration options including makeId
 * @returns Query result with models data
 */
export function useVehicleModels(options: UseVehicleModelsOptions) {
  const { makeId, year, enabled = true } = options

  return useQuery<VehicleModel[], ApiError>({
    queryKey: vehicleSelectorKeys.models(makeId || '', year),
    queryFn: () => getVehicleModels(makeId!, year),
    enabled: enabled && !!makeId && makeId.trim().length > 0,
    staleTime: 60 * 60 * 1000, // 1 hour
    gcTime: 24 * 60 * 60 * 1000, // 24 hours cache
    retry: 2,
    refetchOnWindowFocus: false,
  })
}

// =============================================================================
// useVINDecode - Decode VIN (mutation)
// =============================================================================

interface VINDecodeCallbacks {
  onSuccess?: (data: VINDecodeResult) => void
  onError?: (error: ApiError) => void
}

/**
 * Hook for decoding VIN numbers
 * Uses mutation for manual triggering on button click
 * @param callbacks - Optional success/error callbacks
 * @returns Mutation object for VIN decoding
 */
export function useVINDecode(callbacks: VINDecodeCallbacks = {}) {
  const queryClient = useQueryClient()

  return useMutation<VINDecodeResponse, ApiError, string>({
    mutationFn: (vin: string) => decodeVIN(vin),
    onSuccess: (data, vin) => {
      // Cache the result for future lookups
      queryClient.setQueryData(vehicleSelectorKeys.vinDecode(vin), data)
      callbacks.onSuccess?.(data as VINDecodeResult)
    },
    onError: (error) => {
      console.error('VIN decode failed:', error.message)
      callbacks.onError?.(error)
    },
  })
}

// =============================================================================
// useVINDecodeQuery - Decode VIN (query - for auto-decode)
// =============================================================================

interface UseVINDecodeQueryOptions {
  vin: string | undefined
  enabled?: boolean
}

/**
 * Hook for auto-decoding VIN when it reaches 17 characters
 * Uses query for automatic fetching
 * @param options - Configuration options including VIN
 * @returns Query result with decoded vehicle data
 */
export function useVINDecodeQuery(options: UseVINDecodeQueryOptions) {
  const { vin, enabled = true } = options

  return useQuery<VINDecodeResponse, ApiError>({
    queryKey: vehicleSelectorKeys.vinDecode(vin || ''),
    queryFn: () => decodeVIN(vin!),
    enabled: enabled && !!vin && vin.length === 17,
    staleTime: 60 * 60 * 1000, // 1 hour - VIN data doesn't change
    gcTime: 24 * 60 * 60 * 1000, // 24 hours cache
    retry: (failureCount, error) => {
      // Don't retry on validation errors (400)
      if (error.status === 400) {
        return false
      }
      return failureCount < 2
    },
    refetchOnWindowFocus: false,
  })
}

// =============================================================================
// Helper: Generate year options based on model data
// =============================================================================

/**
 * Model year range for generating year options
 */
interface ModelYearRange {
  year_start: number
  year_end?: number | null
}

/**
 * Generate year options based on model's production years
 * @param model - Vehicle model with year range (optional)
 * @param defaultYearRange - Number of years to show by default
 * @returns Array of year numbers in descending order
 */
export function generateYearOptions(
  model?: ModelYearRange | null,
  defaultYearRange = 35
): number[] {
  const currentYear = new Date().getFullYear()

  if (model) {
    const startYear = model.year_start
    const endYear = model.year_end ?? currentYear + 1
    const years: number[] = []

    for (let year = Math.min(endYear, currentYear + 1); year >= startYear; year--) {
      years.push(year)
    }

    return years
  }

  // Default range: current year down to (currentYear - defaultYearRange)
  return Array.from(
    { length: defaultYearRange + 1 },
    (_, i) => currentYear - i
  )
}

// =============================================================================
// Helper: Prefetch functions
// =============================================================================

/**
 * Hook that returns a function to prefetch vehicle makes
 * Useful for improving perceived performance
 */
export function usePrefetchMakes() {
  const queryClient = useQueryClient()

  return () => {
    queryClient.prefetchQuery({
      queryKey: vehicleSelectorKeys.makes(),
      queryFn: () => getVehicleMakes(),
      staleTime: 60 * 60 * 1000,
    })
  }
}

/**
 * Hook that returns a function to prefetch models for a make
 * Useful for preloading on hover
 */
export function usePrefetchModels() {
  const queryClient = useQueryClient()

  return (makeId: string) => {
    if (makeId) {
      queryClient.prefetchQuery({
        queryKey: vehicleSelectorKeys.models(makeId),
        queryFn: () => getVehicleModels(makeId),
        staleTime: 60 * 60 * 1000,
      })
    }
  }
}
