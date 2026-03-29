/**
 * React Query hooks for the Garage feature.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { ApiError } from '../api'
import {
  createCost,
  createReminder,
  createVehicle,
  completeReminder,
  deleteReminder,
  deleteVehicle,
  getCosts,
  getReminders,
  getUpcomingReminders,
  getVehicle,
  getVehicleHealth,
  getVehicleRecalls,
  getVehicles,
  updateVehicle,
  type GetRemindersParams,
  type MaintenanceCost,
  type MaintenanceCostCreate,
  type MaintenanceReminder,
  type MaintenanceReminderCreate,
  type UserVehicle,
  type UserVehicleCreate,
  type UserVehicleUpdate,
  type VehicleRecall,
} from '../garageService'

// ─── Query Keys ───────────────────────────────────────────────────────────────

export const garageKeys = {
  all: ['garage'] as const,
  vehicles: () => [...garageKeys.all, 'vehicles'] as const,
  vehicle: (id: string) => [...garageKeys.all, 'vehicle', id] as const,
  vehicleHealth: (id: string) => [...garageKeys.all, 'health', id] as const,
  reminders: (params?: GetRemindersParams) => [...garageKeys.all, 'reminders', params] as const,
  upcomingReminders: (days?: number) => [...garageKeys.all, 'upcoming', days] as const,
  costs: (vehicleId?: string) => [...garageKeys.all, 'costs', vehicleId] as const,
}

// ─── Vehicle Hooks ────────────────────────────────────────────────────────────

export function useVehicles() {
  return useQuery({
    queryKey: garageKeys.vehicles(),
    queryFn: () => getVehicles(),
    staleTime: 2 * 60 * 1000,
  })
}

export function useVehicle(vehicleId: string | undefined) {
  return useQuery({
    queryKey: garageKeys.vehicle(vehicleId || ''),
    queryFn: () => getVehicle(vehicleId!),
    enabled: !!vehicleId,
    staleTime: 2 * 60 * 1000,
  })
}

export function useVehicleHealth(vehicleId: string | undefined) {
  return useQuery({
    queryKey: garageKeys.vehicleHealth(vehicleId || ''),
    queryFn: () => getVehicleHealth(vehicleId!),
    enabled: !!vehicleId,
    staleTime: 5 * 60 * 1000,
  })
}

export function useCreateVehicle() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: UserVehicleCreate) => createVehicle(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: garageKeys.vehicles() })
    },
    onError: (error: ApiError) => {
      console.error('Jármű létrehozás sikertelen:', error.message)
    },
  })
}

export function useUpdateVehicle() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ vehicleId, data }: { vehicleId: string; data: UserVehicleUpdate }) =>
      updateVehicle(vehicleId, data),
    onSuccess: (updated: UserVehicle) => {
      queryClient.invalidateQueries({ queryKey: garageKeys.vehicles() })
      queryClient.setQueryData(garageKeys.vehicle(updated.id), updated)
    },
    onError: (error: ApiError) => {
      console.error('Jármű frissítés sikertelen:', error.message)
    },
  })
}

export function useDeleteVehicle() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (vehicleId: string) => deleteVehicle(vehicleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: garageKeys.vehicles() })
    },
    onError: (error: ApiError) => {
      console.error('Jármű törlés sikertelen:', error.message)
    },
  })
}

// ─── Reminder Hooks ───────────────────────────────────────────────────────────

export function useReminders(params: GetRemindersParams = {}) {
  return useQuery({
    queryKey: garageKeys.reminders(params),
    queryFn: () => getReminders(params),
    staleTime: 1 * 60 * 1000,
  })
}

export function useUpcomingReminders(daysAhead = 30) {
  return useQuery({
    queryKey: garageKeys.upcomingReminders(daysAhead),
    queryFn: () => getUpcomingReminders(daysAhead),
    staleTime: 5 * 60 * 1000,
  })
}

export function useCreateReminder() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: MaintenanceReminderCreate) => createReminder(data),
    onSuccess: (_: MaintenanceReminder, variables: MaintenanceReminderCreate) => {
      queryClient.invalidateQueries({ queryKey: garageKeys.reminders() })
      queryClient.invalidateQueries({ queryKey: garageKeys.upcomingReminders() })
      queryClient.invalidateQueries({ queryKey: garageKeys.vehicleHealth(variables.vehicle_id) })
    },
    onError: (error: ApiError) => {
      console.error('Emlékeztető létrehozás sikertelen:', error.message)
    },
  })
}

export function useCompleteReminder() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (reminderId: string) => completeReminder(reminderId),
    onSuccess: (completed: MaintenanceReminder) => {
      queryClient.invalidateQueries({ queryKey: garageKeys.reminders() })
      queryClient.invalidateQueries({ queryKey: garageKeys.upcomingReminders() })
      queryClient.invalidateQueries({ queryKey: garageKeys.vehicleHealth(completed.vehicle_id) })
    },
    onError: (error: ApiError) => {
      console.error('Emlékeztető teljesítés sikertelen:', error.message)
    },
  })
}

export function useDeleteReminder() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (reminderId: string) => deleteReminder(reminderId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: garageKeys.reminders() })
      queryClient.invalidateQueries({ queryKey: garageKeys.upcomingReminders() })
    },
    onError: (error: ApiError) => {
      console.error('Emlékeztető törlés sikertelen:', error.message)
    },
  })
}

// ─── Cost Hooks ───────────────────────────────────────────────────────────────

export function useCosts(vehicleId?: string) {
  return useQuery({
    queryKey: garageKeys.costs(vehicleId),
    queryFn: () => getCosts(vehicleId),
    staleTime: 2 * 60 * 1000,
  })
}

export function useCreateCost() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: MaintenanceCostCreate) => createCost(data),
    onSuccess: (_: MaintenanceCost, variables: MaintenanceCostCreate) => {
      queryClient.invalidateQueries({ queryKey: garageKeys.costs(variables.vehicle_id) })
    },
    onError: (error: ApiError) => {
      console.error('Költség rögzítés sikertelen:', error.message)
    },
  })
}

export function useVehicleRecalls(vehicleId: string | undefined) {
  return useQuery({
    queryKey: [...garageKeys.all, 'recalls', vehicleId] as const,
    queryFn: () => getVehicleRecalls(vehicleId!),
    enabled: !!vehicleId,
    staleTime: 10 * 60 * 1000,
  })
}

// Re-export VehicleRecall type for consumers
export type { VehicleRecall }
