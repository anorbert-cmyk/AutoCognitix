/**
 * Inspection Service
 * Handles technical inspection (Műszaki Vizsga) risk evaluation API operations
 */

import api from './api'

// =============================================================================
// Types
// =============================================================================

export interface InspectionRequest {
  vehicle_make: string
  vehicle_model: string
  vehicle_year: number
  vehicle_engine?: string
  dtc_codes: string[]
  mileage_km?: number
  symptoms?: string
}

export interface FailingItem {
  category: string
  category_hu: string
  issue: string
  related_dtc: string
  severity: 'fail' | 'warning' | 'pass'
  fix_recommendation: string
  estimated_cost_min: number
  estimated_cost_max: number
}

export interface InspectionResponse {
  overall_risk: 'high' | 'medium' | 'low'
  risk_score: number
  failing_items: FailingItem[]
  passing_categories: string[]
  recommendations: string[]
  estimated_total_fix_cost_min: number
  estimated_total_fix_cost_max: number
  vehicle_info: string
  dtc_count: number
}

// =============================================================================
// Service Functions
// =============================================================================

/**
 * Evaluate technical inspection risk based on vehicle data and DTC codes
 * @param data Inspection request data
 * @returns Inspection evaluation with risk score and failing items
 */
export async function evaluateInspection(data: InspectionRequest): Promise<InspectionResponse> {
  const response = await api.post<InspectionResponse>('/inspection/evaluate', data)
  return response.data
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format cost range in Hungarian Forint for inspection
 * @param min Minimum cost
 * @param max Maximum cost
 * @returns Formatted cost range string
 */
export function formatInspectionCost(min: number, max: number): string {
  const formatter = new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency: 'HUF',
    maximumFractionDigits: 0,
  })

  if (min === 0 && max === 0) {
    return 'Nincs becsles'
  }

  return `${formatter.format(min)} - ${formatter.format(max)}`
}

/**
 * Get risk level label in Hungarian
 * @param risk Risk level
 * @returns Hungarian label
 */
export function getRiskLabelHu(risk: 'high' | 'medium' | 'low'): string {
  switch (risk) {
    case 'high':
      return 'Magas kockazat'
    case 'medium':
      return 'Kozepes kockazat'
    case 'low':
      return 'Alacsony kockazat'
    default:
      return 'Ismeretlen'
  }
}

// =============================================================================
// Export service object for convenience
// =============================================================================

export const inspectionService = {
  evaluate: evaluateInspection,
  formatCost: formatInspectionCost,
  getRiskLabelHu,
}

export default inspectionService
