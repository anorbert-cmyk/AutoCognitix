/**
 * Service Shop Service
 * Handles all service shop comparison API operations
 */

import api from './api'

// =============================================================================
// Types
// =============================================================================

export interface ServiceShop {
  id: string
  name: string
  address: string
  city: string
  region: string
  lat: number
  lng: number
  phone?: string
  website?: string
  rating: number
  review_count: number
  price_level: number // 1-3
  specializations: string[]
  accepted_makes: string[]
  services: string[]
  has_inspection: boolean
  has_courtesy_car: boolean
  opening_hours?: string
  distance_km?: number
}

export interface Region {
  id: string
  name: string
  county: string
  lat: number
  lng: number
}

export interface ServiceSearchParams {
  region?: string
  vehicle_make?: string
  service_type?: string
  sort_by?: 'rating' | 'price' | 'distance'
  limit?: number
  offset?: number
}

export interface ServiceSearchResponse {
  shops: ServiceShop[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

// =============================================================================
// Service Functions
// =============================================================================

/**
 * Search for service shops with filters
 * @param params Search and filter parameters
 * @returns Paginated list of matching service shops
 */
export async function searchShops(params: ServiceSearchParams): Promise<ServiceSearchResponse> {
  const response = await api.get<ServiceSearchResponse>('/services/search', {
    params: {
      region: params.region,
      vehicle_make: params.vehicle_make,
      service_type: params.service_type,
      sort_by: params.sort_by,
      limit: params.limit || 20,
      offset: params.offset || 0,
    },
  })

  return response.data
}

/**
 * Get list of available regions
 * @returns List of Hungarian regions
 */
export async function getRegions(): Promise<Region[]> {
  const response = await api.get<Region[]>('/services/regions')
  return response.data
}

/**
 * Get detailed information about a specific service shop
 * @param id The shop ID
 * @returns Full shop details
 */
export async function getShopById(id: string): Promise<ServiceShop> {
  const response = await api.get<ServiceShop>(`/services/${id}`)
  return response.data
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format price level as currency symbols
 * @param level Price level (1-3)
 * @returns Euro symbols string
 */
export function formatPriceLevel(level: number): string {
  return '\u20AC'.repeat(Math.max(1, Math.min(3, level)))
}

/**
 * Get price level label in Hungarian
 * @param level Price level (1-3)
 * @returns Hungarian label
 */
export function getPriceLevelLabelHu(level: number): string {
  switch (level) {
    case 1:
      return 'Kedvez\u0151 \u00E1rak'
    case 2:
      return '\u00C1tlagos \u00E1rak'
    case 3:
      return 'Pr\u00E9mium \u00E1rak'
    default:
      return 'Ismeretlen'
  }
}

/**
 * Get service type label in Hungarian
 * @param type Service type key
 * @returns Hungarian label
 */
export function getServiceTypeLabelHu(type: string): string {
  const labels: Record<string, string> = {
    general: '\u00C1ltal\u00E1nos',
    diagnostics: 'Diagnosztika',
    bodywork: 'Karossz\u00E9ria',
    inspection: 'M\u0171szaki vizsga',
    electric: 'Elektromos',
  }
  return labels[type] || type
}

// =============================================================================
// Export service object for convenience
// =============================================================================

export const serviceShopService = {
  search: searchShops,
  getRegions,
  getById: getShopById,
  formatPriceLevel,
  getPriceLevelLabelHu,
  getServiceTypeLabelHu,
}

export default serviceShopService
