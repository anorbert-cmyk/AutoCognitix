/**
 * Garage Service — API client for vehicle garage, reminders, and maintenance costs.
 */

import { api } from './api'

// ─── Types ────────────────────────────────────────────────────────────────────

export type FuelType = 'petrol' | 'diesel' | 'electric' | 'hybrid' | 'lpg' | 'other'

export type ReminderType =
  | 'oil_change'
  | 'tire_rotation'
  | 'mueszaki_vizsga'
  | 'kotelezo_biztositas'
  | 'coolant'
  | 'brake_fluid'
  | 'timing_belt'
  | 'air_filter'
  | 'brake_pads'
  | 'custom'

export type ReminderUrgency = 'overdue' | 'urgent' | 'upcoming' | 'ok'

export const REMINDER_TYPE_LABELS: Record<ReminderType, string> = {
  oil_change: 'Olajcsere',
  tire_rotation: 'Gumicsere / Forgatás',
  mueszaki_vizsga: 'Műszaki vizsga',
  kotelezo_biztositas: 'Kötelező biztosítás megújítás',
  coolant: 'Hűtőfolyadék csere',
  brake_fluid: 'Fékfolyadék csere',
  timing_belt: 'Vezérszíj csere',
  air_filter: 'Légszűrő csere',
  brake_pads: 'Fékbetét csere',
  custom: 'Egyedi emlékeztető',
}

export const FUEL_TYPE_LABELS: Record<FuelType, string> = {
  petrol: 'Benzin',
  diesel: 'Dízel',
  electric: 'Elektromos',
  hybrid: 'Hibrid',
  lpg: 'LPG / Gáz',
  other: 'Egyéb',
}

export interface UserVehicle {
  id: string
  user_id: string
  nickname?: string | null
  make: string
  model: string
  year: number
  vin?: string | null
  license_plate?: string | null
  mileage_km?: number | null
  fuel_type?: FuelType | null
  color?: string | null
  notes?: string | null
  is_active: boolean
  created_at: string
  updated_at: string
  health_score?: number | null
  upcoming_reminders_count?: number | null
}

export interface UserVehicleCreate {
  nickname?: string
  make: string
  model: string
  year: number
  vin?: string
  license_plate?: string
  mileage_km?: number
  fuel_type?: FuelType
  color?: string
  notes?: string
}

export interface UserVehicleUpdate {
  nickname?: string
  make?: string
  model?: string
  year?: number
  vin?: string
  license_plate?: string
  mileage_km?: number
  fuel_type?: FuelType
  color?: string
  notes?: string
}

export interface UserVehicleListResponse {
  vehicles: UserVehicle[]
  total: number
}

export interface VehicleHealthScore {
  vehicle_id: string
  score: number
  category: string
  category_color: 'green' | 'yellow' | 'orange' | 'red'
  factors: Array<{ type: 'positive' | 'negative'; label: string; impact: number }>
}

export interface MaintenanceReminder {
  id: string
  vehicle_id: string
  user_id: string
  reminder_type: ReminderType
  reminder_type_label: string
  title: string
  due_date?: string | null
  due_mileage_km?: number | null
  notes?: string | null
  is_completed: boolean
  completed_at?: string | null
  email_sent_at?: string | null
  created_at: string
  updated_at: string
  days_until_due?: number | null
  urgency?: ReminderUrgency | null
}

export interface MaintenanceReminderCreate {
  vehicle_id: string
  reminder_type: ReminderType
  title: string
  due_date?: string
  due_mileage_km?: number
  notes?: string
}

export interface MaintenanceReminderListResponse {
  reminders: MaintenanceReminder[]
  total: number
  overdue_count: number
  urgent_count: number
}

export interface MaintenanceCost {
  id: string
  vehicle_id: string
  user_id: string
  diagnosis_session_id?: string | null
  service_type: string
  cost_huf: number
  service_date: string
  mileage_km?: number | null
  workshop_name?: string | null
  notes?: string | null
  created_at: string
}

export interface MaintenanceCostCreate {
  vehicle_id: string
  service_type: string
  cost_huf: number
  service_date: string
  mileage_km?: number
  workshop_name?: string
  notes?: string
  diagnosis_session_id?: string
}

export interface MaintenanceCostListResponse {
  costs: MaintenanceCost[]
  total: number
  total_cost_huf: number
}

// ─── Vehicle API ──────────────────────────────────────────────────────────────

export async function getVehicles(): Promise<UserVehicleListResponse> {
  const response = await api.get<UserVehicleListResponse>('/garage/vehicles')
  return response.data
}

export async function createVehicle(data: UserVehicleCreate): Promise<UserVehicle> {
  const response = await api.post<UserVehicle>('/garage/vehicles', data)
  return response.data
}

export async function getVehicle(vehicleId: string): Promise<UserVehicle> {
  const response = await api.get<UserVehicle>(`/garage/vehicles/${vehicleId}`)
  return response.data
}

export async function updateVehicle(vehicleId: string, data: UserVehicleUpdate): Promise<UserVehicle> {
  const response = await api.put<UserVehicle>(`/garage/vehicles/${vehicleId}`, data)
  return response.data
}

export async function deleteVehicle(vehicleId: string): Promise<void> {
  await api.delete(`/garage/vehicles/${vehicleId}`)
}

export async function getVehicleHealth(vehicleId: string): Promise<VehicleHealthScore> {
  const response = await api.get<VehicleHealthScore>(`/garage/vehicles/${vehicleId}/health`)
  return response.data
}

// ─── Reminder API ─────────────────────────────────────────────────────────────

export interface GetRemindersParams {
  vehicle_id?: string
  include_completed?: boolean
}

export async function getReminders(params: GetRemindersParams = {}): Promise<MaintenanceReminderListResponse> {
  const response = await api.get<MaintenanceReminderListResponse>('/garage/reminders', { params })
  return response.data
}

export async function createReminder(data: MaintenanceReminderCreate): Promise<MaintenanceReminder> {
  const response = await api.post<MaintenanceReminder>('/garage/reminders', data)
  return response.data
}

export async function completeReminder(reminderId: string): Promise<MaintenanceReminder> {
  const response = await api.post<MaintenanceReminder>(`/garage/reminders/${reminderId}/complete`)
  return response.data
}

export async function deleteReminder(reminderId: string): Promise<void> {
  await api.delete(`/garage/reminders/${reminderId}`)
}

export async function getUpcomingReminders(daysAhead = 30): Promise<MaintenanceReminder[]> {
  const response = await api.get<{ reminders: MaintenanceReminder[] }>('/garage/reminders/upcoming', {
    params: { days_ahead: daysAhead },
  })
  return response.data.reminders
}

// ─── Cost API ─────────────────────────────────────────────────────────────────

export async function getCosts(vehicleId?: string): Promise<MaintenanceCostListResponse> {
  const response = await api.get<MaintenanceCostListResponse>('/garage/costs', {
    params: vehicleId ? { vehicle_id: vehicleId } : undefined,
  })
  return response.data
}

export async function createCost(data: MaintenanceCostCreate): Promise<MaintenanceCost> {
  const response = await api.post<MaintenanceCost>('/garage/costs', data)
  return response.data
}

// ─── Utilities ────────────────────────────────────────────────────────────────

export function formatHealthScore(score: number): string {
  if (score >= 80) return 'Kiváló'
  if (score >= 60) return 'Jó'
  if (score >= 40) return 'Figyelmet igényel'
  return 'Kritikus'
}

export function getHealthScoreColorClass(score: number): string {
  if (score >= 80) return 'text-green-600'
  if (score >= 60) return 'text-yellow-600'
  if (score >= 40) return 'text-orange-600'
  return 'text-red-600'
}

export function getUrgencyColorClass(urgency: ReminderUrgency): string {
  switch (urgency) {
    case 'overdue': return 'text-red-600 bg-red-50 border-red-200'
    case 'urgent': return 'text-orange-600 bg-orange-50 border-orange-200'
    case 'upcoming': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    default: return 'text-green-600 bg-green-50 border-green-200'
  }
}

export function formatCostHuf(amount: number): string {
  return new Intl.NumberFormat('hu-HU', { style: 'currency', currency: 'HUF', maximumFractionDigits: 0 }).format(amount)
}
