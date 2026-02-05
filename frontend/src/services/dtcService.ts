/**
 * DTC (Diagnostic Trouble Code) Service
 * Handles all DTC-related API operations
 */

import api, {
  ApiError,
  DTCCategory,
  DTCCategoryInfo,
  DTCCodeDetail,
  DTCSearchResult,
} from './api'

// =============================================================================
// Types
// =============================================================================

export interface DTCSearchParams {
  query: string
  category?: DTCCategory
  make?: string
  limit?: number
}

// =============================================================================
// Service Functions
// =============================================================================

/**
 * Search for DTC codes by code or description
 * @param params Search parameters
 * @returns List of matching DTC codes
 * @throws ApiError on request failure
 */
export async function searchDTCCodes(params: DTCSearchParams): Promise<DTCSearchResult[]> {
  const { query, category, make, limit = 20 } = params

  if (!query || query.trim().length === 0) {
    return []
  }

  const response = await api.get<DTCSearchResult[]>('/dtc/search', {
    params: {
      q: query.trim(),
      category,
      make,
      limit,
    },
  })

  return response.data
}

/**
 * Get detailed information about a specific DTC code
 * @param code The DTC code (e.g., P0101)
 * @returns Detailed DTC code information
 * @throws ApiError on request failure
 */
export async function getDTCCodeDetail(code: string): Promise<DTCCodeDetail> {
  if (!code || code.trim().length === 0) {
    throw new ApiError('DTC kod megadasa kotelezo', 400, 'DTC kod megadasa kotelezo')
  }

  const normalizedCode = code.toUpperCase().trim()

  // Validate DTC code format
  if (!isValidDTCFormat(normalizedCode)) {
    throw new ApiError(
      'Ervenytelen DTC kod formatum. Peldaul: P0101, B1234, C0567, U0100',
      400,
      'Ervenytelen DTC kod formatum'
    )
  }

  const response = await api.get<DTCCodeDetail>(`/dtc/${normalizedCode}`)
  return response.data
}

/**
 * Get DTC codes related to the specified code
 * @param code The DTC code
 * @returns List of related DTC codes
 * @throws ApiError on request failure
 */
export async function getRelatedDTCCodes(code: string): Promise<DTCSearchResult[]> {
  if (!code || code.trim().length === 0) {
    return []
  }

  const normalizedCode = code.toUpperCase().trim()
  const response = await api.get<DTCSearchResult[]>(`/dtc/${normalizedCode}/related`)
  return response.data
}

/**
 * Get list of DTC categories with descriptions
 * @returns List of DTC categories
 * @throws ApiError on request failure
 */
export async function getDTCCategories(): Promise<DTCCategoryInfo[]> {
  const response = await api.get<DTCCategoryInfo[]>('/dtc/categories/list')
  return response.data
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Validate DTC code format
 * @param code The DTC code to validate
 * @returns true if valid format
 */
export function isValidDTCFormat(code: string): boolean {
  if (!code || code.length !== 5) {
    return false
  }

  const prefix = code[0].toUpperCase()
  const digits = code.substring(1)

  // Valid prefixes: P (Powertrain), B (Body), C (Chassis), U (Network)
  if (!['P', 'B', 'C', 'U'].includes(prefix)) {
    return false
  }

  // Check if remaining characters are digits
  return /^\d{4}$/.test(digits)
}

/**
 * Get category from DTC code
 * @param code The DTC code
 * @returns Category or undefined if invalid
 */
export function getCategoryFromCode(code: string): DTCCategory | undefined {
  if (!code || code.length < 1) {
    return undefined
  }

  const prefix = code[0].toUpperCase()
  switch (prefix) {
    case 'P':
      return 'powertrain'
    case 'B':
      return 'body'
    case 'C':
      return 'chassis'
    case 'U':
      return 'network'
    default:
      return undefined
  }
}

/**
 * Get Hungarian category name
 * @param category The category
 * @returns Hungarian name
 */
export function getCategoryNameHu(category: DTCCategory): string {
  switch (category) {
    case 'powertrain':
      return 'Hajtaslanc'
    case 'body':
      return 'Karosszeria'
    case 'chassis':
      return 'Alvaz'
    case 'network':
      return 'Halozat'
    default:
      return 'Ismeretlen'
  }
}

/**
 * Get severity label in Hungarian
 * @param severity The severity level
 * @returns Hungarian label
 */
export function getSeverityLabelHu(severity: string): string {
  switch (severity) {
    case 'low':
      return 'Alacsony'
    case 'medium':
      return 'Kozepes'
    case 'high':
      return 'Magas'
    case 'critical':
      return 'Kritikus'
    default:
      return 'Ismeretlen'
  }
}

/**
 * Get severity color class for UI
 * @param severity The severity level
 * @returns Tailwind color class
 */
export function getSeverityColorClass(severity: string): string {
  switch (severity) {
    case 'low':
      return 'text-green-600 bg-green-100'
    case 'medium':
      return 'text-yellow-600 bg-yellow-100'
    case 'high':
      return 'text-orange-600 bg-orange-100'
    case 'critical':
      return 'text-red-600 bg-red-100'
    default:
      return 'text-gray-600 bg-gray-100'
  }
}

/**
 * Format DTC code for display
 * @param code The DTC code
 * @returns Formatted code with description prefix
 */
export function formatDTCCode(code: string): string {
  const category = getCategoryFromCode(code)
  if (!category) {
    return code
  }

  const categoryName = getCategoryNameHu(category)
  return `${code} (${categoryName})`
}

// =============================================================================
// Export service object for convenience
// =============================================================================

export const dtcService = {
  search: searchDTCCodes,
  getDetail: getDTCCodeDetail,
  getRelated: getRelatedDTCCodes,
  getCategories: getDTCCategories,
  isValidFormat: isValidDTCFormat,
  getCategoryFromCode,
  getCategoryNameHu,
  getSeverityLabelHu,
  getSeverityColorClass,
  formatCode: formatDTCCode,
}

export default dtcService
