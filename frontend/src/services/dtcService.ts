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
  DTCSeverity,
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
 * A severity level rendered for the user: Hungarian label + chip colours.
 */
export interface SeverityChip {
  label: string
  className: string
}

/**
 * SINGLE SOURCE OF TRUTH for rendering a `DTCSeverity`.
 *
 * This mapping previously existed FOUR times — here, in `DTCDetailPage`
 * (`getSeverityColor`, a verbatim copy of the old `getSeverityColorClass`
 * body), in `CommonIssuesPanel` (`SEVERITY_CHIPS`) and inline in
 * `DTCAutocomplete` — with three different palettes and two different spellings
 * of the same word. All four now read from here.
 *
 * LABELS are accented Hungarian. `Kozepes` was an ASCII-folding artifact of this
 * file's older era (cf. its `Hajtaslanc` / `Karosszeria` neighbours); every
 * recently written screen spells it `Közepes` (`ResultPage.tsx`,
 * `DemoResultPage.tsx`, `PasswordStrengthMeter.tsx`), and
 * `ResultPage.test.tsx` asserts that spelling.
 *
 * COLOURS are the `-800` text shades on `-100` backgrounds, MEASURED against
 * WCAG 2.1 AA for normal text (>= 4.5:1) — these chips render at 10-14px, so the
 * 3:1 large-text allowance never applies to them:
 *
 *   green-800  #166534 on green-100  #dcfce7 =  6.49:1  PASS
 *   yellow-800 #854d0e on yellow-100 #fef9c3 =  6.38:1  PASS
 *   orange-800 #9a3412 on orange-100 #ffedd5 =  6.38:1  PASS
 *   red-800    #991b1b on red-100    #fee2e2 =  6.80:1  PASS
 *
 * The `-600` shades this file used to return FAILED at every level — red 3.95:1,
 * orange 3.11:1, green 3.00:1, and yellow 2.74:1, which misses even the 3:1
 * non-text floor. `CommonIssuesPanel` had already worked around that by defining
 * the AA-safe values locally; promoting ITS values (rather than inventing new
 * ones) is what makes deduplication and the accessibility fix the same edit, and
 * leaves that panel's pixels untouched.
 */
const SEVERITY_CHIPS: Record<DTCSeverity, SeverityChip> = {
  low: { label: 'Alacsony', className: 'bg-green-100 text-green-800' },
  medium: { label: 'Közepes', className: 'bg-yellow-100 text-yellow-800' },
  high: { label: 'Magas', className: 'bg-orange-100 text-orange-800' },
  critical: { label: 'Kritikus', className: 'bg-red-100 text-red-800' },
}

/**
 * Fallback for a severity value outside the `DTCSeverity` union.
 *
 * Kept at `gray-600` rather than following the `-800` family: it already
 * measures 6.87:1 on `gray-100` (AA PASS), and "unknown" must read QUIETER than
 * a real severity, not heavier. Unchanged from the previous behaviour.
 */
const UNKNOWN_SEVERITY_CHIP: SeverityChip = {
  label: 'Ismeretlen',
  className: 'bg-gray-100 text-gray-600',
}

/**
 * Look up the chip for a severity, or `undefined` if the value is missing or
 * unrecognised.
 *
 * PARTIAL on purpose. `VehicleCommonIssue.severity` is `string | null`, and
 * `CommonIssuesPanel` renders NO chip rather than a placeholder one when the
 * backend omits it — the same "never render a fabricated value" rule that file
 * applies to its frequency labels. Callers holding a required `DTCSeverity`
 * should use `getSeverityLabelHu` / `getSeverityColorClass`, which are total.
 *
 * @param severity The severity level
 * @returns The chip, or undefined for an unknown/absent severity
 */
export function getSeverityChip(severity: string | null | undefined): SeverityChip | undefined {
  if (!severity) {
    return undefined
  }

  return SEVERITY_CHIPS[severity as DTCSeverity]
}

/**
 * Get severity label in Hungarian
 * @param severity The severity level
 * @returns Hungarian label
 */
export function getSeverityLabelHu(severity: string): string {
  return (getSeverityChip(severity) ?? UNKNOWN_SEVERITY_CHIP).label
}

/**
 * Get severity color class for UI
 * @param severity The severity level
 * @returns Tailwind color class
 */
export function getSeverityColorClass(severity: string): string {
  return (getSeverityChip(severity) ?? UNKNOWN_SEVERITY_CHIP).className
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
  getSeverityChip,
  getSeverityLabelHu,
  getSeverityColorClass,
  formatCode: formatDTCCode,
}

export default dtcService
