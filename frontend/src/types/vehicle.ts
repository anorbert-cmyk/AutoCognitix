/**
 * Vehicle-related TypeScript interfaces
 * These types match the backend API schemas
 */

/**
 * Vehicle manufacturer (make) information
 */
export interface VehicleMake {
  id: string
  name: string
  country?: string
  logo_url?: string
}

/**
 * Vehicle model information
 */
export interface VehicleModel {
  id: string
  name: string
  make_id: string
  year_start: number
  year_end?: number | null
  body_types?: string[]
}

/**
 * VIN decode result from NHTSA API
 */
export interface VINDecodeResult {
  vin: string
  make: string
  model: string
  year: number
  trim?: string
  engine?: string
  transmission?: string
  drive_type?: string
  body_type?: string
  fuel_type?: string
  region?: string
  country_of_origin?: string
}

/**
 * Selected vehicle state for forms
 */
export interface SelectedVehicle {
  make: string
  makeId?: string
  model: string
  modelId?: string
  year: number
  vin?: string
  engine?: string
}

/**
 * Vehicle option for react-select dropdown
 */
export interface VehicleOption {
  value: string
  label: string
  data?: VehicleMake | VehicleModel
}

/**
 * Year option for react-select dropdown
 */
export interface YearOption {
  value: number
  label: string
}

/**
 * Vehicle recall information
 */
export interface VehicleRecall {
  campaign_number: string
  report_received_date?: string
  component: string
  summary: string
  consequence?: string
  remedy?: string
  manufacturer?: string
}

/**
 * Vehicle complaint information
 */
export interface VehicleComplaint {
  odiNumber: string
  crash: boolean
  fire: boolean
  numberOfInjuries: number
  numberOfDeaths: number
  dateOfIncident?: string
  dateComplaintFiled?: string
  components: string[]
  summary: string
}

/**
 * Props for VehicleSelector component
 */
export interface VehicleSelectorProps {
  /** Current selected vehicle (controlled component) */
  value?: SelectedVehicle
  /** Callback when vehicle selection changes */
  onChange: (vehicle: SelectedVehicle | null) => void
  /** Whether the selector is disabled */
  disabled?: boolean
  /** Whether vehicle selection is required */
  required?: boolean
  /** Custom class name for the container */
  className?: string
  /** Whether to show the VIN input section */
  showVINInput?: boolean
  /** Error message to display */
  error?: string
}
