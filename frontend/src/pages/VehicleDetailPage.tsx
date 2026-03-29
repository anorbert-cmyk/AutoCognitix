/**
 * VehicleDetailPage — Jármű részletek oldal
 * Egyedi jármű adatai, emlékeztetők és karbantartási előzmények.
 */

import { Link, useParams } from 'react-router-dom'
import { ArrowLeft, Car, Loader2, AlertCircle } from 'lucide-react'
import { useVehicle, useVehicleHealth, useReminders, useCosts } from '../services/hooks/useGarage'
import {
  formatHealthScore,
  getHealthScoreColorClass,
  getUrgencyColorClass,
  formatCostHuf,
  FUEL_TYPE_LABELS,
  REMINDER_TYPE_LABELS,
} from '../services/garageService'

export default function VehicleDetailPage() {
  const { vehicleId } = useParams<{ vehicleId: string }>()

  const { data: vehicle, isLoading: vehicleLoading, isError: vehicleError } = useVehicle(vehicleId)
  const { data: health } = useVehicleHealth(vehicleId)
  const { data: remindersData } = useReminders({ vehicle_id: vehicleId })
  const { data: costsData } = useCosts(vehicleId)

  if (vehicleLoading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <Loader2 className="h-8 w-8 animate-spin text-primary-600" />
      </div>
    )
  }

  if (vehicleError || !vehicle) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-slate-900 mb-2">Jármű nem található</h2>
        <p className="text-slate-500 mb-6">A keresett jármű nem létezik vagy nincs hozzáférési jogosultsága.</p>
        <Link to="/garage" className="inline-flex items-center gap-2 text-primary-600 hover:underline font-medium">
          <ArrowLeft className="h-4 w-4" />
          Vissza a garázs listához
        </Link>
      </div>
    )
  }

  const activeReminders = remindersData?.reminders.filter((r) => !r.is_completed) ?? []

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
      {/* Breadcrumb */}
      <Link to="/garage" className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-primary-600 transition-colors">
        <ArrowLeft className="h-4 w-4" />
        Garázs
      </Link>

      {/* Vehicle header */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
        <div className="flex items-start gap-4">
          <div className="h-14 w-14 rounded-full bg-primary-100 flex items-center justify-center flex-shrink-0">
            <Car className="h-7 w-7 text-primary-600" />
          </div>
          <div className="flex-1 min-w-0">
            <h1 className="text-2xl font-bold text-slate-900">
              {vehicle.nickname || `${vehicle.make} ${vehicle.model}`}
            </h1>
            <p className="text-slate-500 mt-1">
              {vehicle.year} · {vehicle.make} {vehicle.model}
              {vehicle.fuel_type ? ` · ${FUEL_TYPE_LABELS[vehicle.fuel_type]}` : ''}
            </p>
            {vehicle.license_plate && (
              <span className="inline-block mt-2 px-3 py-1 bg-slate-100 rounded-md text-sm font-mono text-slate-700">
                {vehicle.license_plate}
              </span>
            )}
          </div>
          {health && (
            <div className="text-right flex-shrink-0">
              <p className="text-sm text-slate-500">Állapot</p>
              <p className={`text-2xl font-bold ${getHealthScoreColorClass(health.score)}`}>
                {health.score}%
              </p>
              <p className={`text-sm font-medium ${getHealthScoreColorClass(health.score)}`}>
                {formatHealthScore(health.score)}
              </p>
            </div>
          )}
        </div>

        {vehicle.mileage_km && (
          <div className="mt-4 pt-4 border-t border-slate-100">
            <p className="text-sm text-slate-500">
              Futásteljesítmény: <span className="font-semibold text-slate-800">{vehicle.mileage_km.toLocaleString('hu-HU')} km</span>
            </p>
          </div>
        )}
      </div>

      {/* Reminders */}
      {activeReminders.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
          <h2 className="text-lg font-bold text-slate-900 mb-4">Emlékeztetők</h2>
          <div className="space-y-3">
            {activeReminders.map((reminder) => (
              <div
                key={reminder.id}
                className={`flex items-center justify-between p-3 rounded-xl border ${
                  reminder.urgency ? getUrgencyColorClass(reminder.urgency) : 'bg-slate-50 border-slate-200'
                }`}
              >
                <div>
                  <p className="font-medium text-slate-900 text-sm">{reminder.title}</p>
                  <p className="text-xs text-slate-500">
                    {REMINDER_TYPE_LABELS[reminder.reminder_type]}
                    {reminder.due_date ? ` · ${reminder.due_date}` : ''}
                  </p>
                </div>
                {reminder.days_until_due !== null && reminder.days_until_due !== undefined && (
                  <span className="text-xs font-semibold">
                    {reminder.days_until_due < 0
                      ? `${Math.abs(reminder.days_until_due)} napja lejárt`
                      : reminder.days_until_due === 0
                      ? 'Ma esedékes'
                      : `${reminder.days_until_due} nap`}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Costs */}
      {costsData && costsData.costs.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-slate-900">Karbantartási költségek</h2>
            <span className="text-sm font-semibold text-slate-700">
              Összesen: {formatCostHuf(costsData.total_cost_huf)}
            </span>
          </div>
          <div className="space-y-2">
            {costsData.costs.slice(0, 5).map((cost) => (
              <div key={cost.id} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
                <div>
                  <p className="text-sm font-medium text-slate-800">{cost.service_type}</p>
                  <p className="text-xs text-slate-500">
                    {cost.service_date}
                    {cost.workshop_name ? ` · ${cost.workshop_name}` : ''}
                  </p>
                </div>
                <span className="text-sm font-semibold text-slate-900">
                  {formatCostHuf(cost.cost_huf)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {activeReminders.length === 0 && (!costsData || costsData.costs.length === 0) && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-12 text-center">
          <Car className="h-12 w-12 text-slate-300 mx-auto mb-4" />
          <p className="text-slate-500">Nincsenek rögzített emlékeztetők vagy karbantartási adatok.</p>
        </div>
      )}
    </div>
  )
}
