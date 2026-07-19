/**
 * GaragePage — Garázs kezelő oldal
 * Jármű lista, hozzáadás, törlés és diagnosztika navigáció.
 */

import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Car, Plus, Loader2, Trash2, ChevronRight, Stethoscope, X } from 'lucide-react'
import { useToast } from '../contexts/ToastContext'
import {
  useVehicles,
  useCreateVehicle,
  useDeleteVehicle,
} from '../services/hooks/useGarage'
import {
  formatHealthScore,
  getHealthScoreColorClass,
  FUEL_TYPE_LABELS,
  type FuelType,
  type UserVehicleCreate,
} from '../services/garageService'

// =============================================================================
// Constants
// =============================================================================

const currentYear = new Date().getFullYear()

const EMPTY_FORM: UserVehicleCreate = {
  make: '',
  model: '',
  year: currentYear,
  license_plate: '',
  mileage_km: undefined,
  fuel_type: 'petrol' as FuelType,
  nickname: '',
}

// =============================================================================
// VehicleCard sub-component
// =============================================================================

interface VehicleCardProps {
  vehicle: {
    id: string
    make: string
    model: string
    year: number
    nickname?: string | null
    fuel_type?: FuelType | null
    mileage_km?: number | null
    health_score?: number | null
    upcoming_reminders_count?: number | null
  }
  onDelete: (id: string, name: string) => void
  isDeleting: boolean
}

function VehicleCard({ vehicle, onDelete, isDeleting }: VehicleCardProps) {
  const navigate = useNavigate()
  const displayName = vehicle.nickname || `${vehicle.make} ${vehicle.model}`
  const remindersCount = vehicle.upcoming_reminders_count ?? 0

  return (
    <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-lg font-bold text-slate-900 leading-tight">{displayName}</h3>
          <p className="text-sm text-slate-500 mt-0.5">
            {vehicle.year}
            {vehicle.fuel_type ? ` · ${FUEL_TYPE_LABELS[vehicle.fuel_type]}` : ''}
          </p>
        </div>
        <button
          onClick={() => onDelete(vehicle.id, displayName)}
          disabled={isDeleting}
          aria-label={`${displayName} törlése`}
          className="p-2 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors disabled:opacity-40"
        >
          <Trash2 className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>

      {/* Health score */}
      <div className="flex items-center gap-3">
        {vehicle.health_score != null ? (
          <>
            <div
              className={`text-3xl font-black leading-none ${getHealthScoreColorClass(vehicle.health_score)}`}
              aria-label={`Állapot: ${vehicle.health_score}`}
            >
              {vehicle.health_score}
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-700">Állapot</div>
              <div className="text-xs text-slate-500">{formatHealthScore(vehicle.health_score)}</div>
            </div>
          </>
        ) : (
          <>
            <div
              className="text-3xl font-black leading-none text-slate-400"
              aria-label="Állapot: nincs adat"
            >
              —
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-700">Állapot</div>
              <div className="text-xs text-slate-500">Nincs adat</div>
            </div>
          </>
        )}
      </div>

      {/* Stats row */}
      <div className="flex items-center gap-3 text-xs text-slate-500">
        {vehicle.mileage_km != null && (
          <span>{vehicle.mileage_km.toLocaleString('hu-HU')} km</span>
        )}
        {remindersCount > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-yellow-100 text-yellow-700 font-semibold border border-yellow-200">
            {remindersCount} emlékeztető
          </span>
        )}
        {remindersCount === 0 && (
          <span className="text-slate-400">0 emlékeztető</span>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex gap-2 pt-1">
        <button
          onClick={() => navigate(`/garage/${vehicle.id}`)}
          className="flex-1 flex items-center justify-center gap-1.5 h-9 rounded-xl border border-slate-200 text-slate-700 text-sm font-semibold hover:bg-slate-50 transition-colors"
        >
          Részletek
          <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
        <button
          onClick={() => navigate(`/diagnosis?vehicle=${vehicle.id}`)}
          className="flex-1 flex items-center justify-center gap-1.5 h-9 rounded-xl bg-[#2563eb] text-white text-sm font-semibold hover:bg-blue-700 transition-colors"
        >
          <Stethoscope className="h-3.5 w-3.5" aria-hidden="true" />
          Diagnosztika
        </button>
      </div>
    </div>
  )
}

// =============================================================================
// GaragePage
// =============================================================================

export default function GaragePage() {
  const toast = useToast()
  const [showAddModal, setShowAddModal] = useState(false)
  const [form, setForm] = useState<UserVehicleCreate>({ ...EMPTY_FORM })

  const { data: vehiclesData, isLoading, error } = useVehicles()
  const createVehicle = useCreateVehicle()
  const deleteVehicle = useDeleteVehicle()

  const vehicles = vehiclesData?.vehicles ?? []

  // ── Handlers ──────────────────────────────────────────────────────────────

  const handleOpenModal = () => {
    setForm({ ...EMPTY_FORM, year: currentYear })
    setShowAddModal(true)
  }

  const handleCloseModal = () => {
    setShowAddModal(false)
    setForm({ ...EMPTY_FORM, year: currentYear })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.make.trim() || !form.model.trim()) {
      toast.error('Gyártó és modell megadása kötelező')
      return
    }
    if (form.year < 1970 || form.year > currentYear + 1) {
      toast.error(`Érvényes évjárat szükséges (1970–${currentYear + 1})`)
      return
    }
    const payload: UserVehicleCreate = {
      make: form.make.trim(),
      model: form.model.trim(),
      year: form.year,
      fuel_type: form.fuel_type || undefined,
      license_plate: form.license_plate?.trim() || undefined,
      mileage_km: form.mileage_km || undefined,
      nickname: form.nickname?.trim() || undefined,
    }
    try {
      await createVehicle.mutateAsync(payload)
      toast.success('Jármű sikeresen hozzáadva!')
      handleCloseModal()
    } catch {
      toast.error('Nem sikerült hozzáadni a járművet')
    }
  }

  const handleDelete = async (vehicleId: string, vehicleName: string) => {
    if (!confirm(`Biztosan törölni szeretnéd: ${vehicleName}?`)) return
    try {
      await deleteVehicle.mutateAsync(vehicleId)
      toast.success('Jármű eltávolítva')
    } catch {
      toast.error('Törlés sikertelen')
    }
  }

  // ── Field helper ──────────────────────────────────────────────────────────

  const setField = <K extends keyof UserVehicleCreate>(key: K, value: UserVehicleCreate[K]) => {
    setForm((prev) => ({ ...prev, [key]: value }))
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      <main className="w-full max-w-5xl mx-auto p-4 md:p-8 lg:p-12">

        {/* Header */}
        <div className="flex items-start justify-between mb-10">
          <div>
            <h1 className="text-4xl font-black tracking-tight text-slate-900 mb-2">Garázs</h1>
            <p className="text-lg text-slate-600">Kezeld járműveid és karbantartási emlékeztetőidet</p>
          </div>
          <button
            onClick={handleOpenModal}
            className="flex items-center gap-2 h-11 px-5 rounded-xl bg-[#2563eb] text-white font-bold text-sm hover:bg-blue-700 transition-colors shadow-sm"
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
            Jármű hozzáadása
          </button>
        </div>

        {/* Loading state */}
        {isLoading && (
          <div className="flex items-center justify-center py-24">
            <Loader2 className="h-8 w-8 animate-spin text-[#2563eb]" aria-label="Betöltés..." />
          </div>
        )}

        {/* Error state */}
        {error && !isLoading && (
          <div className="bg-red-50 border border-red-200 rounded-2xl p-6 text-center text-red-700">
            <p className="font-semibold">Nem sikerült betölteni a járműveket.</p>
            <p className="text-sm mt-1 text-red-500">Próbáld újratölteni az oldalt.</p>
          </div>
        )}

        {/* Empty state */}
        {!isLoading && !error && vehicles.length === 0 && (
          <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-16 flex flex-col items-center gap-4 text-center">
            <div className="text-5xl" aria-hidden="true">🚗</div>
            <h2 className="text-xl font-bold text-slate-800">Még nincs jármű hozzáadva</h2>
            <p className="text-slate-500 max-w-xs">
              Add hozzá első járműved és kövesd nyomon a karbantartási teendőket!
            </p>
            <button
              onClick={handleOpenModal}
              className="mt-2 flex items-center gap-2 h-11 px-6 rounded-xl bg-[#2563eb] text-white font-bold text-sm hover:bg-blue-700 transition-colors"
            >
              <Plus className="h-4 w-4" aria-hidden="true" />
              Jármű hozzáadása
            </button>
          </div>
        )}

        {/* Vehicle grid */}
        {!isLoading && !error && vehicles.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            {vehicles.map((vehicle) => (
              <VehicleCard
                key={vehicle.id}
                vehicle={vehicle}
                onDelete={handleDelete}
                isDeleting={deleteVehicle.isPending}
              />
            ))}
          </div>
        )}
      </main>

      {/* Add Vehicle Modal */}
      {showAddModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="Új jármű hozzáadása"
        >
          <div className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-md mx-4">
            {/* Modal header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Car className="h-5 w-5 text-[#2563eb]" aria-hidden="true" />
                <h2 className="text-xl font-bold text-slate-900">Új jármű hozzáadása</h2>
              </div>
              <button
                onClick={handleCloseModal}
                aria-label="Bezárás"
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Make + Model */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label htmlFor="garage-make" className="block text-sm font-bold text-slate-600">
                    Gyártó <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="garage-make"
                    type="text"
                    value={form.make}
                    onChange={(e) => setField('make', e.target.value)}
                    placeholder="pl. Volkswagen"
                    required
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="garage-model" className="block text-sm font-bold text-slate-600">
                    Modell <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="garage-model"
                    type="text"
                    value={form.model}
                    onChange={(e) => setField('model', e.target.value)}
                    placeholder="pl. Golf VII"
                    required
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
              </div>

              {/* Year + Fuel */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label htmlFor="garage-year" className="block text-sm font-bold text-slate-600">
                    Évjárat <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="garage-year"
                    type="number"
                    value={form.year}
                    onChange={(e) => setField('year', parseInt(e.target.value, 10) || currentYear)}
                    min="1970"
                    max={currentYear + 1}
                    required
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="garage-fuel" className="block text-sm font-bold text-slate-600">
                    Üzemanyag
                  </label>
                  <select
                    id="garage-fuel"
                    value={form.fuel_type ?? 'petrol'}
                    onChange={(e) => setField('fuel_type', e.target.value as FuelType)}
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium appearance-none cursor-pointer"
                  >
                    {(Object.entries(FUEL_TYPE_LABELS) as [FuelType, string][]).map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* License plate + Mileage */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label htmlFor="garage-plate" className="block text-sm font-bold text-slate-600">
                    Rendszám
                  </label>
                  <input
                    id="garage-plate"
                    type="text"
                    value={form.license_plate ?? ''}
                    onChange={(e) => setField('license_plate', e.target.value.toUpperCase())}
                    placeholder="pl. ABC-123"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium uppercase"
                  />
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="garage-mileage" className="block text-sm font-bold text-slate-600">
                    Km állás
                  </label>
                  <input
                    id="garage-mileage"
                    type="number"
                    value={form.mileage_km ?? ''}
                    onChange={(e) =>
                      setField('mileage_km', e.target.value ? parseInt(e.target.value, 10) : undefined)
                    }
                    placeholder="pl. 87000"
                    min="0"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
              </div>

              {/* Nickname */}
              <div className="space-y-1.5">
                <label htmlFor="garage-nickname" className="block text-sm font-bold text-slate-600">
                  Becenév (opcionális)
                </label>
                <input
                  id="garage-nickname"
                  type="text"
                  value={form.nickname ?? ''}
                  onChange={(e) => setField('nickname', e.target.value)}
                  placeholder="pl. Fehér Golf"
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                />
              </div>

              {/* Action buttons */}
              <div className="flex gap-3 justify-end pt-2">
                <button
                  type="button"
                  onClick={handleCloseModal}
                  className="h-10 px-5 rounded-xl border border-slate-200 text-slate-700 text-sm font-semibold hover:bg-slate-50 transition-colors"
                >
                  Mégsem
                </button>
                <button
                  type="submit"
                  disabled={createVehicle.isPending}
                  className="h-10 px-5 rounded-xl bg-[#2563eb] text-white text-sm font-bold hover:bg-blue-700 transition-colors disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {createVehicle.isPending ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                      Mentés...
                    </>
                  ) : (
                    'Hozzáadás'
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
