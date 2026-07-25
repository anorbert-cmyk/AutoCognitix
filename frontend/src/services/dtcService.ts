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
 *
 * The query here is a PARTIAL, in-progress input: `DTCAutocomplete` fires this
 * from two characters (`"P0"`), and it also accepts free-text descriptions.
 * It is therefore deliberately NOT run through `isValidDTCFormat()`, which is
 * an anchored whole-code check - applying it here would break
 * search-as-you-type by rejecting every prefix short of a full 5-char code.
 * Guarded by `searchDTCCodes > partial-input regression guard` in
 * `__tests__/dtcService.test.ts`.
 *
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
 * Canonical shape of a single, already-normalised DTC code.
 *
 * Canonical rule: backend/app/core/dtc_codes.py (`DTC_CODE_STRICT`, SAE J2012)
 * - P/B/C/U, then 0-3, then 3 HEX digits. There is no shared module across the
 * language boundary, so this regex deliberately mirrors it character for
 * character (the same way `DiagnosisForm.tsx` does).
 *
 * Both halves are load-bearing:
 * - characters 3-5 are HEX, not decimal. The old `\d{4}` tail rejected every
 *   real hex code a scan tool reports (`P26B7`, `P090C`, `P0A94`, `B00A0`,
 *   `P17F1`), so `getDTCCodeDetail()` threw before the request was even sent
 *   and the user could not open the detail page at all.
 * - character 2 is constrained to `0-3`. Without it, P/B/C/U + hex letters
 *   spells ordinary English (`PEACE`, `PACED`) and matches campaign ids
 *   (`P9324`, `UA80E`, `U760E`), which is how junk got into the corpus before.
 *
 * Deliberately un-flagged (no `/g`): a `RegExp` with `g` carries `lastIndex`
 * state across `.test()` calls, which would make this shared constant answer
 * differently on alternate invocations.
 */
const DTC_CODE_STRICT = /^[PBCU][0-3][0-9A-F]{3}$/

/**
 * Validate a COMPLETE DTC code.
 *
 * Mirrors `is_valid_dtc_code()` in backend/app/core/dtc_codes.py, including its
 * tolerance of surrounding whitespace and lower case: normalisation lives here
 * (once) so the helper is total for every caller. `validateDiagnosisRequest()`
 * passes raw, un-normalised user input, while `getDTCCodeDetail()` and
 * `quickAnalyze()` pass already-uppercased strings - upper-casing an
 * upper-cased string is idempotent, so there is one rule and no drift.
 *
 * This is an ANCHORED, whole-string check. Do NOT use it to gate a
 * partial/in-progress input such as an autocomplete query: `"P0"` is a
 * perfectly good search prefix but is not a code, and search-as-you-type
 * (`searchDTCCodes`) is intentionally not format-validated.
 *
 * @param code The DTC code to validate
 * @returns true if valid format
 */
export function isValidDTCFormat(code: string): boolean {
  if (!code) {
    return false
  }

  return DTC_CODE_STRICT.test(code.trim().toUpperCase())
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
