/**
 * VehicleDetailPage — Jármű részletek oldal
 * Egyedi jármű adatai, emlékeztetők kezelése és karbantartási előzmények.
 *
 * URL: /garage/:vehicleId
 */

import { useState } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  Car,
  Loader2,
  AlertCircle,
  Plus,
  Check,
  Trash2,
  X,
  Stethoscope,
  Bell,
  Banknote,
  ChevronDown,
} from 'lucide-react'
import { useToast } from '../contexts/ToastContext'
import {
  useVehicle,
  useVehicleHealth,
  useReminders,
  useCosts,
  useCreateReminder,
  useCompleteReminder,
  useDeleteReminder,
  useCreateCost,
} from '../services/hooks/useGarage'
import {
  formatHealthScore,
  getHealthScoreColorClass,
  getUrgencyColorClass,
  formatCostHuf,
  FUEL_TYPE_LABELS,
  REMINDER_TYPE_LABELS,
  type ReminderType,
  type MaintenanceReminderCreate,
  type MaintenanceCostCreate,
} from '../services/garageService'

// =============================================================================
// Constants
// =============================================================================

type ActiveTab = 'reminders' | 'costs'

const EMPTY_REMINDER: Omit<MaintenanceReminderCreate, 'vehicle_id'> = {
  reminder_type: 'oil_change',
  title: REMINDER_TYPE_LABELS['oil_change'],
  due_date: '',
  due_mileage_km: undefined,
  notes: '',
}

// =============================================================================
// VehicleDetailPage
// =============================================================================

export default function VehicleDetailPage() {
  const { vehicleId } = useParams<{ vehicleId: string }>()
  const navigate = useNavigate()
  const toast = useToast()

  const [activeTab, setActiveTab] = useState<ActiveTab>('reminders')
  const [showAddReminder, setShowAddReminder] = useState(false)
  const [showAddCost, setShowAddCost] = useState(false)

  // Reminder form state
  const [reminderForm, setReminderForm] = useState<Omit<MaintenanceReminderCreate, 'vehicle_id'>>({
    ...EMPTY_REMINDER,
  })

  // Cost form state
  const today = new Date().toISOString().split('T')[0]
  const [costForm, setCostForm] = useState<Omit<MaintenanceCostCreate, 'vehicle_id'>>({
    service_type: '',
    cost_huf: 0,
    service_date: today,
    mileage_km: undefined,
    workshop_name: '',
    notes: '',
  })

  // ── Queries ──────────────────────────────────────────────────────────────────

  const { data: vehicle, isLoading: vehicleLoading, isError: vehicleError } = useVehicle(vehicleId)
  const { data: health } = useVehicleHealth(vehicleId)
  const { data: remindersData } = useReminders({ vehicle_id: vehicleId })
  const { data: costsData } = useCosts(vehicleId)

  // ── Mutations ────────────────────────────────────────────────────────────────

  const createReminder = useCreateReminder()
  const completeReminder = useCompleteReminder()
  const deleteReminder = useDeleteReminder()
  const createCost = useCreateCost()

  // ── Loading / Error ───────────────────────────────────────────────────────────

  if (vehicleLoading) {
    return (
      <div className="flex items-center justify-center min-h-[50vh]">
        <Loader2 className="h-8 w-8 animate-spin text-[#2563eb]" aria-label="Betöltés..." />
      </div>
    )
  }

  if (vehicleError || !vehicle) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-16 text-center">
        <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" aria-hidden="true" />
        <h2 className="text-xl font-semibold text-slate-900 mb-2">Jármű nem található</h2>
        <p className="text-slate-500 mb-6">
          A keresett jármű nem létezik vagy nincs hozzáférési jogosultsága.
        </p>
        <Link
          to="/garage"
          className="inline-flex items-center gap-2 text-[#2563eb] hover:underline font-medium"
        >
          <ArrowLeft className="h-4 w-4" />
          Vissza a garázs listához
        </Link>
      </div>
    )
  }

  const displayName = vehicle.nickname || `${vehicle.make} ${vehicle.model}`
  const activeReminders = remindersData?.reminders.filter((r) => !r.is_completed) ?? []
  const allCosts = costsData?.costs ?? []
  const healthScore = health?.score ?? vehicle.health_score ?? null

  // ── Reminder handlers ─────────────────────────────────────────────────────────

  const handleReminderTypeChange = (type: ReminderType) => {
    setReminderForm((prev) => ({
      ...prev,
      reminder_type: type,
      title: REMINDER_TYPE_LABELS[type],
    }))
  }

  const handleSubmitReminder = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!reminderForm.title.trim()) {
      toast.error('A cím megadása kötelező')
      return
    }
    if (!reminderForm.due_date) {
      toast.error('A határidő megadása kötelező')
      return
    }
    if (!vehicleId) return
    try {
      await createReminder.mutateAsync({
        vehicle_id: vehicleId,
        reminder_type: reminderForm.reminder_type,
        title: reminderForm.title.trim(),
        due_date: reminderForm.due_date,
        due_mileage_km: reminderForm.due_mileage_km || undefined,
        notes: reminderForm.notes?.trim() || undefined,
      })
      toast.success('Emlékeztető sikeresen létrehozva!')
      setShowAddReminder(false)
      setReminderForm({ ...EMPTY_REMINDER })
    } catch {
      toast.error('Nem sikerült létrehozni az emlékeztetőt')
    }
  }

  const handleCompleteReminder = async (reminderId: string, title: string) => {
    try {
      await completeReminder.mutateAsync(reminderId)
      toast.success(`"${title}" teljesítve!`)
    } catch {
      toast.error('Nem sikerült teljesítettnek jelölni')
    }
  }

  const handleDeleteReminder = async (reminderId: string, title: string) => {
    if (!confirm(`Biztosan törölni szeretnéd: "${title}"?`)) return
    try {
      await deleteReminder.mutateAsync(reminderId)
      toast.success('Emlékeztető törölve')
    } catch {
      toast.error('Törlés sikertelen')
    }
  }

  // ── Cost handlers ─────────────────────────────────────────────────────────────

  const handleSubmitCost = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!costForm.service_type.trim()) {
      toast.error('A szerviz típus megadása kötelező')
      return
    }
    if (!costForm.cost_huf || costForm.cost_huf <= 0) {
      toast.error('Érvényes összeg megadása kötelező')
      return
    }
    if (!costForm.service_date) {
      toast.error('A dátum megadása kötelező')
      return
    }
    if (!vehicleId) return
    try {
      await createCost.mutateAsync({
        vehicle_id: vehicleId,
        service_type: costForm.service_type.trim(),
        cost_huf: costForm.cost_huf,
        service_date: costForm.service_date,
        mileage_km: costForm.mileage_km || undefined,
        workshop_name: costForm.workshop_name?.trim() || undefined,
        notes: costForm.notes?.trim() || undefined,
      })
      toast.success('Kiadás sikeresen rögzítve!')
      setShowAddCost(false)
      setCostForm({
        service_type: '',
        cost_huf: 0,
        service_date: today,
        mileage_km: undefined,
        workshop_name: '',
        notes: '',
      })
    } catch {
      toast.error('Nem sikerült rögzíteni a kiadást')
    }
  }

  // ── Days until due helper ─────────────────────────────────────────────────────

  const formatDaysUntilDue = (days: number | null | undefined): string => {
    if (days === null || days === undefined) return ''
    if (days < 0) return `${Math.abs(days)} napja lejárt`
    if (days === 0) return 'Ma esedékes'
    return `${days} nap múlva`
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-[#f8fafc]">
      <main className="w-full max-w-4xl mx-auto p-4 md:p-8">

        {/* Breadcrumb */}
        <Link
          to="/garage"
          className="inline-flex items-center gap-2 text-sm text-slate-500 hover:text-[#2563eb] transition-colors mb-8"
        >
          <ArrowLeft className="h-4 w-4" />
          Vissza a garázzsba
        </Link>

        {/* Vehicle header card */}
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 mb-6">
          <div className="flex items-start gap-4">
            <div className="h-14 w-14 rounded-2xl bg-blue-50 flex items-center justify-center flex-shrink-0">
              <Car className="h-7 w-7 text-[#2563eb]" aria-hidden="true" />
            </div>
            <div className="flex-1 min-w-0">
              <h1 className="text-2xl font-black text-slate-900 leading-tight">{displayName}</h1>
              <p className="text-slate-500 mt-1 text-sm">
                {vehicle.year} · {vehicle.make} {vehicle.model}
                {vehicle.fuel_type ? ` · ${FUEL_TYPE_LABELS[vehicle.fuel_type]}` : ''}
                {vehicle.license_plate ? ` · ${vehicle.license_plate}` : ''}
                {vehicle.mileage_km != null
                  ? ` · ${vehicle.mileage_km.toLocaleString('hu-HU')} km`
                  : ''}
              </p>
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              {healthScore !== null && (
                <div className="text-right">
                  <div className="text-xs text-slate-500 font-medium mb-0.5">Állapot</div>
                  <div
                    className={`text-2xl font-black leading-none ${getHealthScoreColorClass(healthScore)}`}
                    aria-label={`Állapot: ${healthScore}`}
                  >
                    {healthScore}
                  </div>
                  <div className={`text-xs font-semibold mt-0.5 ${getHealthScoreColorClass(healthScore)}`}>
                    {formatHealthScore(healthScore)}
                  </div>
                </div>
              )}
              <button
                onClick={() => navigate(`/diagnosis?vehicle=${vehicle.id}`)}
                className="flex items-center gap-1.5 h-9 px-4 rounded-xl bg-[#2563eb] text-white text-sm font-bold hover:bg-blue-700 transition-colors"
              >
                <Stethoscope className="h-3.5 w-3.5" aria-hidden="true" />
                Diagnosztika
              </button>
            </div>
          </div>
        </div>

        {/* Tab navigation */}
        <div className="flex gap-1 p-1 bg-slate-100 rounded-xl mb-6 w-fit">
          <button
            onClick={() => setActiveTab('reminders')}
            className={`flex items-center gap-2 h-9 px-5 rounded-lg text-sm font-semibold transition-colors ${
              activeTab === 'reminders'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <Bell className="h-4 w-4" aria-hidden="true" />
            Emlékeztetők
            {activeReminders.length > 0 && (
              <span className="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1.5 rounded-full bg-yellow-100 text-yellow-700 text-xs font-bold border border-yellow-200">
                {activeReminders.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setActiveTab('costs')}
            className={`flex items-center gap-2 h-9 px-5 rounded-lg text-sm font-semibold transition-colors ${
              activeTab === 'costs'
                ? 'bg-white text-slate-900 shadow-sm'
                : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            <Banknote className="h-4 w-4" aria-hidden="true" />
            Karbantartási log
          </button>
        </div>

        {/* ── Reminders tab ─────────────────────────────────────────────────── */}
        {activeTab === 'reminders' && (
          <div className="space-y-4">
            {activeReminders.length === 0 ? (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-12 text-center">
                <Bell className="h-10 w-10 text-slate-300 mx-auto mb-3" aria-hidden="true" />
                <p className="text-slate-500 font-medium">Nincsenek aktív emlékeztetők</p>
                <p className="text-slate-400 text-sm mt-1">
                  Adj hozzá egy emlékeztetőt az olajcseréhez, műszaki vizsgához stb.
                </p>
              </div>
            ) : (
              activeReminders.map((reminder) => (
                <div
                  key={reminder.id}
                  className={`bg-white rounded-2xl border shadow-sm p-5 flex items-start gap-4 ${
                    reminder.urgency ? getUrgencyColorClass(reminder.urgency) : 'border-slate-200'
                  }`}
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-slate-900 text-sm">{reminder.title}</span>
                      {reminder.urgency && (
                        <span
                          className={`text-xs font-bold px-2 py-0.5 rounded-full border ${getUrgencyColorClass(
                            reminder.urgency
                          )}`}
                        >
                          {reminder.urgency === 'overdue'
                            ? 'Lejárt'
                            : reminder.urgency === 'urgent'
                            ? 'Sürgős'
                            : reminder.urgency === 'upcoming'
                            ? 'Közelgő'
                            : 'Rendben'}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-slate-500 mt-1">
                      {REMINDER_TYPE_LABELS[reminder.reminder_type]}
                      {reminder.due_date ? ` · ${reminder.due_date}` : ''}
                      {reminder.days_until_due !== null && reminder.days_until_due !== undefined
                        ? ` · ${formatDaysUntilDue(reminder.days_until_due)}`
                        : ''}
                    </p>
                    {reminder.notes && (
                      <p className="text-xs text-slate-400 mt-1 italic">{reminder.notes}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <button
                      onClick={() => handleCompleteReminder(reminder.id, reminder.title)}
                      disabled={completeReminder.isPending}
                      title="Teljesítve"
                      className="flex items-center gap-1.5 h-8 px-3 rounded-lg bg-green-50 border border-green-200 text-green-700 text-xs font-bold hover:bg-green-100 transition-colors disabled:opacity-50"
                    >
                      <Check className="h-3.5 w-3.5" aria-hidden="true" />
                      Teljesítve
                    </button>
                    <button
                      onClick={() => handleDeleteReminder(reminder.id, reminder.title)}
                      disabled={deleteReminder.isPending}
                      title="Törlés"
                      aria-label="Emlékeztető törlése"
                      className="flex items-center justify-center h-8 w-8 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 transition-colors disabled:opacity-50"
                    >
                      <Trash2 className="h-4 w-4" aria-hidden="true" />
                    </button>
                  </div>
                </div>
              ))
            )}

            {/* Add reminder button */}
            <button
              onClick={() => setShowAddReminder(true)}
              className="w-full flex items-center justify-center gap-2 h-12 rounded-2xl border-2 border-dashed border-slate-300 text-slate-500 text-sm font-semibold hover:border-[#2563eb] hover:text-[#2563eb] transition-colors"
            >
              <Plus className="h-4 w-4" aria-hidden="true" />
              Új emlékeztető
            </button>
          </div>
        )}

        {/* ── Costs tab ─────────────────────────────────────────────────────── */}
        {activeTab === 'costs' && (
          <div className="space-y-4">
            {allCosts.length === 0 ? (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-12 text-center">
                <Banknote className="h-10 w-10 text-slate-300 mx-auto mb-3" aria-hidden="true" />
                <p className="text-slate-500 font-medium">Nincsenek rögzített karbantartási kiadások</p>
                <p className="text-slate-400 text-sm mt-1">
                  Rögzítsd a szervizköltségeket a karbantartási előzmények nyomon követéséhez.
                </p>
              </div>
            ) : (
              <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
                {/* Costs header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-slate-100">
                  <h2 className="text-base font-bold text-slate-900">Karbantartási előzmények</h2>
                  <span className="text-sm font-semibold text-slate-700">
                    Összes kiadás:{' '}
                    <span className="text-[#2563eb]">{formatCostHuf(costsData?.total_cost_huf ?? 0)}</span>
                  </span>
                </div>
                {/* Costs table */}
                <div className="divide-y divide-slate-100">
                  {allCosts.map((cost) => (
                    <div key={cost.id} className="flex items-center justify-between px-6 py-4">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-slate-800">{cost.service_type}</p>
                        <p className="text-xs text-slate-500 mt-0.5">
                          {cost.service_date}
                          {cost.workshop_name ? ` · ${cost.workshop_name}` : ''}
                          {cost.mileage_km != null
                            ? ` · ${cost.mileage_km.toLocaleString('hu-HU')} km`
                            : ''}
                        </p>
                        {cost.notes && (
                          <p className="text-xs text-slate-400 mt-0.5 italic">{cost.notes}</p>
                        )}
                      </div>
                      <span className="text-sm font-black text-slate-900 flex-shrink-0 ml-4">
                        {formatCostHuf(cost.cost_huf)}
                      </span>
                    </div>
                  ))}
                </div>
                {/* Total footer */}
                <div className="flex items-center justify-end px-6 py-4 border-t border-slate-100 bg-slate-50">
                  <span className="text-sm font-bold text-slate-700">
                    Összesen:{' '}
                    <span className="text-[#2563eb] text-base">
                      {formatCostHuf(costsData?.total_cost_huf ?? 0)}
                    </span>
                  </span>
                </div>
              </div>
            )}

            {/* Add cost button */}
            <button
              onClick={() => setShowAddCost(true)}
              className="w-full flex items-center justify-center gap-2 h-12 rounded-2xl border-2 border-dashed border-slate-300 text-slate-500 text-sm font-semibold hover:border-[#2563eb] hover:text-[#2563eb] transition-colors"
            >
              <Plus className="h-4 w-4" aria-hidden="true" />
              Kiadás rögzítése
            </button>
          </div>
        )}
      </main>

      {/* ── Add Reminder Modal ─────────────────────────────────────────────────── */}
      {showAddReminder && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="Új emlékeztető hozzáadása"
        >
          <div className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-md mx-4">
            {/* Modal header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Bell className="h-5 w-5 text-[#2563eb]" aria-hidden="true" />
                <h2 className="text-xl font-bold text-slate-900">Új emlékeztető</h2>
              </div>
              <button
                onClick={() => {
                  setShowAddReminder(false)
                  setReminderForm({ ...EMPTY_REMINDER })
                }}
                aria-label="Bezárás"
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>

            <form onSubmit={handleSubmitReminder} className="space-y-4">
              {/* Reminder type */}
              <div className="space-y-1.5">
                <label htmlFor="reminder-type" className="block text-sm font-bold text-slate-600">
                  Típus
                </label>
                <div className="relative">
                  <select
                    id="reminder-type"
                    value={reminderForm.reminder_type}
                    onChange={(e) => handleReminderTypeChange(e.target.value as ReminderType)}
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium appearance-none cursor-pointer"
                  >
                    {(Object.entries(REMINDER_TYPE_LABELS) as [ReminderType, string][]).map(
                      ([value, label]) => (
                        <option key={value} value={value}>
                          {label}
                        </option>
                      )
                    )}
                  </select>
                  <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
                    <ChevronDown className="h-4 w-4 text-slate-400" aria-hidden="true" />
                  </div>
                </div>
              </div>

              {/* Title */}
              <div className="space-y-1.5">
                <label htmlFor="reminder-title" className="block text-sm font-bold text-slate-600">
                  Cím <span className="text-red-500">*</span>
                </label>
                <input
                  id="reminder-title"
                  type="text"
                  value={reminderForm.title}
                  onChange={(e) => setReminderForm((prev) => ({ ...prev, title: e.target.value }))}
                  placeholder="pl. Olajcsere"
                  required
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                />
              </div>

              {/* Due date + km */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label htmlFor="reminder-date" className="block text-sm font-bold text-slate-600">
                    Határidő <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="reminder-date"
                    type="date"
                    value={reminderForm.due_date ?? ''}
                    onChange={(e) =>
                      setReminderForm((prev) => ({ ...prev, due_date: e.target.value }))
                    }
                    required
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="reminder-km" className="block text-sm font-bold text-slate-600">
                    Km határig
                  </label>
                  <input
                    id="reminder-km"
                    type="number"
                    value={reminderForm.due_mileage_km ?? ''}
                    onChange={(e) =>
                      setReminderForm((prev) => ({
                        ...prev,
                        due_mileage_km: e.target.value ? parseInt(e.target.value, 10) : undefined,
                      }))
                    }
                    placeholder="pl. 100000"
                    min="0"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
              </div>

              {/* Notes */}
              <div className="space-y-1.5">
                <label htmlFor="reminder-notes" className="block text-sm font-bold text-slate-600">
                  Megjegyzés
                </label>
                <textarea
                  id="reminder-notes"
                  value={reminderForm.notes ?? ''}
                  onChange={(e) =>
                    setReminderForm((prev) => ({ ...prev, notes: e.target.value }))
                  }
                  placeholder="Opcionális megjegyzés..."
                  rows={2}
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white resize-none p-3 text-sm font-medium"
                />
              </div>

              {/* Action buttons */}
              <div className="flex gap-3 justify-end pt-2">
                <button
                  type="button"
                  onClick={() => {
                    setShowAddReminder(false)
                    setReminderForm({ ...EMPTY_REMINDER })
                  }}
                  className="h-10 px-5 rounded-xl border border-slate-200 text-slate-700 text-sm font-semibold hover:bg-slate-50 transition-colors"
                >
                  Mégsem
                </button>
                <button
                  type="submit"
                  disabled={createReminder.isPending}
                  className="h-10 px-5 rounded-xl bg-[#2563eb] text-white text-sm font-bold hover:bg-blue-700 transition-colors disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {createReminder.isPending ? (
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

      {/* ── Add Cost Modal ───────────────────────────────────────────────────────── */}
      {showAddCost && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="Kiadás rögzítése"
        >
          <div className="bg-white rounded-2xl shadow-xl p-6 w-full max-w-md mx-4">
            {/* Modal header */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2">
                <Banknote className="h-5 w-5 text-[#2563eb]" aria-hidden="true" />
                <h2 className="text-xl font-bold text-slate-900">Kiadás rögzítése</h2>
              </div>
              <button
                onClick={() => setShowAddCost(false)}
                aria-label="Bezárás"
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>

            <form onSubmit={handleSubmitCost} className="space-y-4">
              {/* Service type */}
              <div className="space-y-1.5">
                <label htmlFor="cost-type" className="block text-sm font-bold text-slate-600">
                  Szerviz típus <span className="text-red-500">*</span>
                </label>
                <input
                  id="cost-type"
                  type="text"
                  value={costForm.service_type}
                  onChange={(e) =>
                    setCostForm((prev) => ({ ...prev, service_type: e.target.value }))
                  }
                  placeholder="pl. Olajcsere, Fékbetét csere"
                  required
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                />
              </div>

              {/* Cost + Date */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label htmlFor="cost-amount" className="block text-sm font-bold text-slate-600">
                    Összeg (Ft) <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="cost-amount"
                    type="number"
                    value={costForm.cost_huf || ''}
                    onChange={(e) =>
                      setCostForm((prev) => ({
                        ...prev,
                        cost_huf: e.target.value ? parseInt(e.target.value, 10) : 0,
                      }))
                    }
                    placeholder="pl. 35000"
                    min="1"
                    required
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="cost-date" className="block text-sm font-bold text-slate-600">
                    Dátum <span className="text-red-500">*</span>
                  </label>
                  <input
                    id="cost-date"
                    type="date"
                    value={costForm.service_date}
                    onChange={(e) =>
                      setCostForm((prev) => ({ ...prev, service_date: e.target.value }))
                    }
                    required
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
              </div>

              {/* Workshop + km */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label htmlFor="cost-workshop" className="block text-sm font-bold text-slate-600">
                    Műhely
                  </label>
                  <input
                    id="cost-workshop"
                    type="text"
                    value={costForm.workshop_name ?? ''}
                    onChange={(e) =>
                      setCostForm((prev) => ({ ...prev, workshop_name: e.target.value }))
                    }
                    placeholder="pl. Toyota Szerviz"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
                <div className="space-y-1.5">
                  <label htmlFor="cost-km" className="block text-sm font-bold text-slate-600">
                    Km állás
                  </label>
                  <input
                    id="cost-km"
                    type="number"
                    value={costForm.mileage_km ?? ''}
                    onChange={(e) =>
                      setCostForm((prev) => ({
                        ...prev,
                        mileage_km: e.target.value ? parseInt(e.target.value, 10) : undefined,
                      }))
                    }
                    placeholder="pl. 87000"
                    min="0"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-11 px-3 text-sm font-medium"
                  />
                </div>
              </div>

              {/* Notes */}
              <div className="space-y-1.5">
                <label htmlFor="cost-notes" className="block text-sm font-bold text-slate-600">
                  Megjegyzés
                </label>
                <textarea
                  id="cost-notes"
                  value={costForm.notes ?? ''}
                  onChange={(e) =>
                    setCostForm((prev) => ({ ...prev, notes: e.target.value }))
                  }
                  placeholder="Opcionális megjegyzés..."
                  rows={2}
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white resize-none p-3 text-sm font-medium"
                />
              </div>

              {/* Action buttons */}
              <div className="flex gap-3 justify-end pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddCost(false)}
                  className="h-10 px-5 rounded-xl border border-slate-200 text-slate-700 text-sm font-semibold hover:bg-slate-50 transition-colors"
                >
                  Mégsem
                </button>
                <button
                  type="submit"
                  disabled={createCost.isPending}
                  className="h-10 px-5 rounded-xl bg-[#2563eb] text-white text-sm font-bold hover:bg-blue-700 transition-colors disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2"
                >
                  {createCost.isPending ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                      Mentés...
                    </>
                  ) : (
                    'Rögzítés'
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
