/**
 * Vehicle Service
 * Handles all vehicle-related API operations
 */

import api, {
  ApiError,
  Complaint,
  Recall,
  VehicleMake,
  VehicleModel,
  VINDecodeResponse,
} from './api'

// =============================================================================
// Types
// =============================================================================

export interface VehicleInfo {
  make: string
  model: string
  year: number
  engine?: string
  vin?: string
}

// =============================================================================
// VIN Decoding
// =============================================================================

/**
 * Decode a VIN to get vehicle details
 * @param vin 17-character VIN
 * @returns Decoded vehicle information
 * @throws ApiError on request failure or invalid VIN
 */
export async function decodeVIN(vin: string): Promise<VINDecodeResponse> {
  const normalizedVin = vin.toUpperCase().trim()

  // Client-side validation
  const validationError = validateVIN(normalizedVin)
  if (validationError) {
    throw new ApiError(validationError, 400, validationError, 'VIN_VALIDATION_ERROR')
  }

  const response = await api.post<VINDecodeResponse>('/vehicles/decode-vin', {
    vin: normalizedVin,
  })

  return response.data
}

/**
 * Validate VIN format
 * @param vin The VIN to validate
 * @returns Error message or null if valid
 */
export function validateVIN(vin: string): string | null {
  if (!vin) {
    return 'VIN megadasa kotelezo'
  }

  if (vin.length !== 17) {
    return 'A VIN pontosan 17 karakter kell legyen'
  }

  // Check for invalid characters (I, O, Q are not allowed in VINs)
  if (/[IOQ]/i.test(vin)) {
    return 'A VIN nem tartalmazhat I, O vagy Q karaktereket'
  }

  // VIN should only contain alphanumeric characters
  if (!/^[A-HJ-NPR-Z0-9]{17}$/i.test(vin)) {
    return 'A VIN csak betűket (I, O, Q kivetelevel) es szamokat tartalmazhat'
  }

  return null
}

// =============================================================================
// Vehicle Makes and Models
// =============================================================================

/**
 * Get list of vehicle makes
 * @param search Optional search term to filter makes
 * @returns List of vehicle makes
 * @throws ApiError on request failure
 */
export async function getVehicleMakes(search?: string): Promise<VehicleMake[]> {
  const response = await api.get<VehicleMake[]>('/vehicles/makes', {
    params: search ? { search: search.trim() } : undefined,
  })

  return response.data
}

/**
 * Get list of models for a specific make
 * @param makeId Make identifier
 * @param year Optional year to filter models
 * @returns List of vehicle models
 * @throws ApiError on request failure
 */
export async function getVehicleModels(makeId: string, year?: number): Promise<VehicleModel[]> {
  if (!makeId || makeId.trim().length === 0) {
    return []
  }

  const response = await api.get<VehicleModel[]>(`/vehicles/models/${makeId}`, {
    params: year ? { year } : undefined,
  })

  return response.data
}

/**
 * Get list of available vehicle years
 * @returns Object with years array
 * @throws ApiError on request failure
 */
export async function getVehicleYears(): Promise<number[]> {
  const response = await api.get<{ years: number[] }>('/vehicles/years')
  return response.data.years
}

// =============================================================================
// Recalls and Complaints
// =============================================================================

/**
 * Get recalls for a specific vehicle
 * @param make Vehicle manufacturer
 * @param model Vehicle model
 * @param year Model year
 * @returns List of recalls
 * @throws ApiError on request failure
 */
export async function getVehicleRecalls(
  make: string,
  model: string,
  year: number
): Promise<Recall[]> {
  if (!make || !model || !year) {
    return []
  }

  const response = await api.get<Recall[]>(
    `/vehicles/${encodeURIComponent(make)}/${encodeURIComponent(model)}/${year}/recalls`
  )

  return response.data
}

/**
 * Get complaints for a specific vehicle
 * @param make Vehicle manufacturer
 * @param model Vehicle model
 * @param year Model year
 * @returns List of complaints
 * @throws ApiError on request failure
 */
export async function getVehicleComplaints(
  make: string,
  model: string,
  year: number
): Promise<Complaint[]> {
  if (!make || !model || !year) {
    return []
  }

  const response = await api.get<Complaint[]>(
    `/vehicles/${encodeURIComponent(make)}/${encodeURIComponent(model)}/${year}/complaints`
  )

  return response.data
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format vehicle info for display
 * @param info Vehicle information
 * @returns Formatted string like "2018 Volkswagen Golf"
 */
export function formatVehicleInfo(info: VehicleInfo): string {
  const parts = [info.year.toString(), info.make, info.model]
  if (info.engine) {
    parts.push(info.engine)
  }
  return parts.join(' ')
}

/**
 * Format VIN for display (with spaces for readability)
 * @param vin The VIN
 * @returns Formatted VIN (WVW ZZZ 3CZ WE1 234 56)
 */
export function formatVINForDisplay(vin: string): string {
  if (!vin || vin.length !== 17) {
    return vin || ''
  }

  // Standard VIN grouping: WMI (3) + VDS (6) + VIS (8)
  return `${vin.slice(0, 3)} ${vin.slice(3, 9)} ${vin.slice(9, 11)} ${vin.slice(11, 14)} ${vin.slice(14)}`
}

/**
 * Get region from VIN first character
 * @param vin The VIN
 * @returns Region name in Hungarian
 */
export function getRegionFromVIN(vin: string): string {
  if (!vin || vin.length < 1) {
    return 'Ismeretlen'
  }

  const firstChar = vin[0].toUpperCase()

  // North America
  if ('12345'.includes(firstChar)) {
    return 'Eszak-Amerika'
  }

  // Europe
  if ('SALFGHJKLMNPRSTUVWXYZ'.substring(0, 12).includes(firstChar)) {
    return 'Europa'
  }

  // Asia
  if ('SJKLMNPR'.includes(firstChar)) {
    return 'Azsia'
  }

  // Africa
  if ('ABC'.includes(firstChar)) {
    return 'Afrika'
  }

  // Oceania
  if ('6789'.includes(firstChar)) {
    return 'Oceania'
  }

  // South America
  if ('8'.includes(firstChar)) {
    return 'Del-Amerika'
  }

  return 'Ismeretlen'
}

/**
 * Parse year from VIN (position 10)
 * @param vin The VIN
 * @returns Model year or null if cannot be determined
 */
export function getYearFromVIN(vin: string): number | null {
  if (!vin || vin.length < 10) {
    return null
  }

  const yearCode = vin[9].toUpperCase()

  // Year codes for 1980-2000 (A-Y, excluding I, O, Q, U, Z)
  const yearCodes1980: Record<string, number> = {
    A: 1980,
    B: 1981,
    C: 1982,
    D: 1983,
    E: 1984,
    F: 1985,
    G: 1986,
    H: 1987,
    J: 1988,
    K: 1989,
    L: 1990,
    M: 1991,
    N: 1992,
    P: 1993,
    R: 1994,
    S: 1995,
    T: 1996,
    V: 1997,
    W: 1998,
    X: 1999,
    Y: 2000,
  }

  // Year codes for 2001-2009 (1-9)
  const yearCodes2001: Record<string, number> = {
    '1': 2001,
    '2': 2002,
    '3': 2003,
    '4': 2004,
    '5': 2005,
    '6': 2006,
    '7': 2007,
    '8': 2008,
    '9': 2009,
  }

  // Year codes for 2010-2030 (A-Y again, but we add 30)
  const yearCodes2010: Record<string, number> = {
    A: 2010,
    B: 2011,
    C: 2012,
    D: 2013,
    E: 2014,
    F: 2015,
    G: 2016,
    H: 2017,
    J: 2018,
    K: 2019,
    L: 2020,
    M: 2021,
    N: 2022,
    P: 2023,
    R: 2024,
    S: 2025,
    T: 2026,
    V: 2027,
    W: 2028,
    X: 2029,
    Y: 2030,
  }

  // Try 2010+ first (most common for current vehicles)
  if (yearCodes2010[yearCode]) {
    return yearCodes2010[yearCode]
  }

  // Then try 2001-2009
  if (yearCodes2001[yearCode]) {
    return yearCodes2001[yearCode]
  }

  // Finally try 1980-2000
  if (yearCodes1980[yearCode]) {
    return yearCodes1980[yearCode]
  }

  return null
}

/**
 * Get recall severity based on content
 * @param recall Recall information
 * @returns Severity level
 */
export function getRecallSeverity(recall: Recall): 'low' | 'medium' | 'high' | 'critical' {
  const summary = (recall.summary + (recall.consequence || '')).toLowerCase()

  // Critical keywords
  if (
    summary.includes('fire') ||
    summary.includes('crash') ||
    summary.includes('injury') ||
    summary.includes('death')
  ) {
    return 'critical'
  }

  // High severity keywords
  if (
    summary.includes('brake') ||
    summary.includes('steering') ||
    summary.includes('airbag') ||
    summary.includes('fuel leak')
  ) {
    return 'high'
  }

  // Medium severity (default for most recalls)
  return 'medium'
}

/**
 * Format recall date for display
 * @param dateString Date string from NHTSA
 * @returns Formatted date in Hungarian locale
 */
export function formatRecallDate(dateString?: string): string {
  if (!dateString) {
    return 'Ismeretlen datum'
  }

  try {
    const date = new Date(dateString)
    return date.toLocaleDateString('hu-HU', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  } catch {
    return dateString
  }
}

// =============================================================================
// Export service object for convenience
// =============================================================================

export const vehicleService = {
  decodeVIN,
  validateVIN,
  getMakes: getVehicleMakes,
  getModels: getVehicleModels,
  getYears: getVehicleYears,
  getRecalls: getVehicleRecalls,
  getComplaints: getVehicleComplaints,
  formatVehicleInfo,
  formatVINForDisplay,
  getRegionFromVIN,
  getYearFromVIN,
  getRecallSeverity,
  formatRecallDate,
}

export default vehicleService
