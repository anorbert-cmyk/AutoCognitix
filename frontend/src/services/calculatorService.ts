/**
 * Calculator Service — "Megeri megjavitani?" (Worth Repairing?)
 * Handles repair vs. sell/scrap evaluation API operations
 */

import api, { ApiError } from './api'

// =============================================================================
// Types
// =============================================================================

export interface CalculatorRequest {
  vehicle_make: string
  vehicle_model: string
  vehicle_year: number
  mileage_km: number
  condition: 'excellent' | 'good' | 'fair' | 'poor'
  repair_cost_huf?: number
  diagnosis_id?: string
  fuel_type?: 'petrol' | 'diesel' | 'hybrid' | 'electric' | 'lpg'
}

export interface CostBreakdown {
  parts_cost: number
  labor_cost: number
  additional_costs: number
}

export interface Factor {
  name: string
  impact: 'positive' | 'negative'
  description: string
}

export interface AlternativeScenario {
  scenario: string
  description: string
  estimated_value: number
}

export interface CalculatorResponse {
  vehicle_value_min: number
  vehicle_value_max: number
  vehicle_value_avg: number
  repair_cost_min: number
  repair_cost_max: number
  ratio: number
  recommendation: 'repair' | 'sell' | 'scrap'
  recommendation_text: string
  breakdown: CostBreakdown
  factors: Factor[]
  alternative_scenarios: AlternativeScenario[]
  confidence_score: number
  currency: string
  ai_disclaimer: string
}

// =============================================================================
// Validation
// =============================================================================

export function validateCalculatorRequest(data: CalculatorRequest): string[] {
  const errors: string[] = []

  if (!data.vehicle_make || data.vehicle_make.trim().length === 0) {
    errors.push('Gyarto megadasa kotelezo')
  }

  if (!data.vehicle_model || data.vehicle_model.trim().length === 0) {
    errors.push('Modell megadasa kotelezo')
  }

  if (!data.vehicle_year || data.vehicle_year < 1990 || data.vehicle_year > 2030) {
    errors.push('Ervenytelen evjarat (1990-2030)')
  }

  if (data.mileage_km === undefined || data.mileage_km === null || data.mileage_km < 0 || data.mileage_km > 999_999) {
    errors.push('Ervenytelen kilometerora allas (0 - 999.999 km)')
  }

  if (!data.condition) {
    errors.push('Az allapot megadasa kotelezo')
  }

  if (data.repair_cost_huf !== undefined && data.repair_cost_huf < 0) {
    errors.push('A javitasi koltseg nem lehet negativ')
  }

  return errors
}

// =============================================================================
// Service Functions
// =============================================================================

/**
 * Evaluate whether a vehicle is worth repairing
 * @param data Calculator request data
 * @returns Calculator response with recommendation
 * @throws ApiError on request failure or validation error
 */
export async function evaluateCalculator(data: CalculatorRequest): Promise<CalculatorResponse> {
  const validationErrors = validateCalculatorRequest(data)
  if (validationErrors.length > 0) {
    throw new ApiError(
      validationErrors.join('. '),
      400,
      validationErrors.join('. '),
      'VALIDATION_ERROR'
    )
  }

  const request: CalculatorRequest = {
    vehicle_make: data.vehicle_make.trim(),
    vehicle_model: data.vehicle_model.trim(),
    vehicle_year: data.vehicle_year,
    mileage_km: data.mileage_km,
    condition: data.condition,
    repair_cost_huf: data.repair_cost_huf,
    diagnosis_id: data.diagnosis_id?.trim(),
    fuel_type: data.fuel_type,
  }

  const response = await api.post<CalculatorResponse>('/calculator/evaluate', request)
  return response.data
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format HUF value for display
 */
export function formatHUF(amount: number): string {
  return new Intl.NumberFormat('hu-HU', {
    maximumFractionDigits: 0,
  }).format(amount)
}

/**
 * Format HUF value with currency
 */
export function formatHUFCurrency(amount: number, currency: string = 'HUF'): string {
  return new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(amount)
}

/**
 * Get recommendation label in Hungarian
 */
export function getRecommendationLabelHu(recommendation: 'repair' | 'sell' | 'scrap'): string {
  switch (recommendation) {
    case 'repair':
      return 'Erdemes megjavitani'
    case 'sell':
      return 'Erdemes eladni'
    case 'scrap':
      return 'Bontasra javasolt'
    default:
      return 'Ismeretlen'
  }
}

/**
 * Get condition label in Hungarian
 */
export function getConditionLabelHu(condition: string): string {
  switch (condition) {
    case 'excellent':
      return 'Kivalo'
    case 'good':
      return 'Jo'
    case 'fair':
      return 'Elfogadhato'
    case 'poor':
      return 'Rossz'
    default:
      return condition
  }
}

// =============================================================================
// Export
// =============================================================================

export const calculatorService = {
  evaluate: evaluateCalculator,
  validate: validateCalculatorRequest,
  formatHUF,
  formatHUFCurrency,
  getRecommendationLabelHu,
  getConditionLabelHu,
}

export default calculatorService
