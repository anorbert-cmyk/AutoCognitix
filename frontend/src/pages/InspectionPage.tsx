/**
 * InspectionPage - Műszaki Vizsga kockázat elemző oldal
 * Magyar nyelvű technikai vizsgálati kockázatbecslés DTC kódok és jármű adatok alapján.
 */

import { useState, useCallback } from 'react'
import {
  AlertTriangle,
  Car,
  ChevronDown,
  CheckCircle,
  ClipboardList,
  Loader2,
  Sparkles,
  Shield,
  Gauge,
  Lightbulb,
  Banknote,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useToast } from '../contexts/ToastContext'
import { ApiError } from '../services/api'
import { useEvaluateInspection } from '../services/hooks/useInspection'
import type { InspectionRequest, InspectionResponse } from '../services/inspectionService'
import RiskGauge from '../components/features/inspection/RiskGauge'
import InspectionCategoryCard from '../components/features/inspection/InspectionCategoryCard'

// =============================================================================
// Constants
// =============================================================================

const currentYear = new Date().getFullYear()

const manufacturers = [
  'Toyota',
  'Volkswagen',
  'BMW',
  'Audi',
  'Mercedes-Benz',
  'Ford',
  'Opel',
  'Škoda',
  'Suzuki',
  'Dacia',
  'Hyundai',
  'Kia',
  'Renault',
  'Peugeot',
  'Honda',
  'Mazda',
  'Nissan',
  'Volvo',
  'Citroën',
  'Fiat',
]

// =============================================================================
// Component
// =============================================================================

export default function InspectionPage() {
  const toast = useToast()
  const evaluateInspection = useEvaluateInspection()

  // Form state
  const [vehicleMake, setVehicleMake] = useState('')
  const [vehicleModel, setVehicleModel] = useState('')
  const [vehicleYear, setVehicleYear] = useState('')
  const [dtcCodesInput, setDtcCodesInput] = useState('')
  const [mileage, setMileage] = useState('')
  const [symptoms, setSymptoms] = useState('')

  // Results
  const [result, setResult] = useState<InspectionResponse | null>(null)

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault()

      // Validation
      if (!vehicleMake) {
        toast.error('Gyarto valasztasa kotelezo')
        return
      }
      if (!vehicleModel.trim()) {
        toast.error('Modell megadasa kotelezo')
        return
      }
      if (!vehicleYear || parseInt(vehicleYear, 10) < 1990 || parseInt(vehicleYear, 10) > currentYear + 1) {
        toast.error('Ervenyes evjarat megadasa kotelezo (1990-tol)')
        return
      }
      if (!dtcCodesInput.trim()) {
        toast.error('Legalabb egy DTC kod megadasa kotelezo')
        return
      }

      // Parse DTC codes (comma, space, or semicolon separated)
      const dtcCodes = dtcCodesInput
        .toUpperCase()
        .split(/[,;\s]+/)
        .map((code) => code.trim())
        .filter((code) => code.length > 0)

      if (dtcCodes.length === 0) {
        toast.error('Legalabb egy ervenyes DTC kod megadasa kotelezo')
        return
      }

      const request: InspectionRequest = {
        vehicle_make: vehicleMake.trim(),
        vehicle_model: vehicleModel.trim(),
        vehicle_year: parseInt(vehicleYear, 10),
        dtc_codes: dtcCodes,
        mileage_km: mileage ? parseInt(mileage, 10) : undefined,
        symptoms: symptoms.trim() || undefined,
      }

      try {
        const response = await evaluateInspection.mutateAsync(request)
        setResult(response)
        toast.success('Vizsga kockazat elemzes kesz!')
      } catch (err) {
        if (err instanceof ApiError) {
          toast.error(err.detail, 'Elemzesi hiba')
        } else {
          toast.error('Ismeretlen hiba tortent az elemzes soran')
        }
      }
    },
    [vehicleMake, vehicleModel, vehicleYear, dtcCodesInput, mileage, symptoms, evaluateInspection, toast]
  )

  const costFormatter = new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency: 'HUF',
    maximumFractionDigits: 0,
  })

  return (
    <div className="min-h-screen bg-[#f8fafc] flex flex-col">
      <main className="flex-1 w-full max-w-5xl mx-auto p-4 md:p-8 lg:p-12">
        {/* Header */}
        <div className="mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border border-amber-200 bg-amber-50 text-amber-700 text-xs font-bold uppercase tracking-wider mb-4">
            <Shield className="h-3.5 w-3.5" aria-hidden="true" />
            Muszaki Vizsga Kockazat
          </div>
          <h2 className="text-4xl font-black tracking-tight text-slate-900 mb-3">
            Muszaki vizsga kockazat elemzes
          </h2>
          <p className="text-lg text-slate-600 max-w-3xl">
            Adja meg a jarmu adatait es a jelenlegi hibakodokat. Az AI rendszer elemzi, hogy a jarmu atmenhet-e a muszaki vizsgalaton, es milyen javitasok szuksegesek.
          </p>
        </div>

        {/* Form Card */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
          <form onSubmit={handleSubmit} className="p-6 md:p-8 lg:p-10 space-y-8">
            {/* Vehicle Section */}
            <section>
              <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                <Car className="h-5 w-5 text-[#2563eb]" aria-hidden="true" />
                Jarmu adatok
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Make - Select */}
                <div className="space-y-2">
                  <label htmlFor="inspection-make" className="block text-sm font-bold text-slate-600">
                    Gyarto
                  </label>
                  <div className="relative">
                    <select
                      id="inspection-make"
                      value={vehicleMake}
                      onChange={(e) => setVehicleMake(e.target.value)}
                      className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium appearance-none cursor-pointer"
                    >
                      <option value="">Valasszon gyartot</option>
                      {manufacturers.map((make) => (
                        <option key={make} value={make}>
                          {make}
                        </option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 pr-4 flex items-center pointer-events-none">
                      <ChevronDown className="h-5 w-5 text-slate-400" aria-hidden="true" />
                    </div>
                  </div>
                </div>

                {/* Model - Text */}
                <div className="space-y-2">
                  <label htmlFor="inspection-model" className="block text-sm font-bold text-slate-600">
                    Modell
                  </label>
                  <input
                    id="inspection-model"
                    type="text"
                    value={vehicleModel}
                    onChange={(e) => setVehicleModel(e.target.value)}
                    placeholder="pl. Golf VII"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium"
                  />
                </div>

                {/* Year - Number */}
                <div className="space-y-2">
                  <label htmlFor="inspection-year" className="block text-sm font-bold text-slate-600">
                    Evjarat
                  </label>
                  <input
                    id="inspection-year"
                    type="number"
                    value={vehicleYear}
                    onChange={(e) => setVehicleYear(e.target.value)}
                    placeholder="2018"
                    min="1990"
                    max={currentYear + 1}
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium"
                  />
                </div>
              </div>
            </section>

            <hr className="border-slate-200" />

            {/* DTC and Mileage Section */}
            <section>
              <h3 className="text-xl font-bold text-slate-900 mb-6 flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-amber-500" aria-hidden="true" />
                Hibakodok es allapot
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* DTC Codes */}
                <div className="space-y-2">
                  <label htmlFor="inspection-dtc" className="block text-sm font-bold text-slate-600">
                    DTC hibakodok
                  </label>
                  <input
                    id="inspection-dtc"
                    type="text"
                    value={dtcCodesInput}
                    onChange={(e) => setDtcCodesInput(e.target.value.toUpperCase())}
                    placeholder="pl. P0300, P0420, C0035"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium uppercase"
                  />
                  <p className="text-xs text-slate-500">
                    Tobb kodot vesszővel, pontosvesszővel vagy szokozzel valasszon el.
                  </p>
                </div>

                {/* Mileage */}
                <div className="space-y-2">
                  <label htmlFor="inspection-mileage" className="block text-sm font-bold text-slate-600">
                    Kilometerorallas (km)
                  </label>
                  <input
                    id="inspection-mileage"
                    type="number"
                    value={mileage}
                    onChange={(e) => setMileage(e.target.value)}
                    placeholder="pl. 156000"
                    min="0"
                    max="999999"
                    className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:border-[#2563eb] focus:ring-0 focus:bg-white h-14 px-4 font-medium"
                  />
                </div>
              </div>
            </section>

            {/* Symptoms Section */}
            <section>
              <div className="space-y-2">
                <label htmlFor="inspection-symptoms" className="block text-sm font-bold text-slate-600 flex items-center gap-2">
                  <ClipboardList className="h-4 w-4 text-slate-400" aria-hidden="true" />
                  Eszlelt tunetek, panaszok (opcionalis)
                </label>
                <textarea
                  id="inspection-symptoms"
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  placeholder="pl. Motorhiba lampa vilagit, egyenetlen alapjarat, fusteleg a kipufogo..."
                  rows={4}
                  className="block w-full rounded-xl border-2 border-slate-200 bg-slate-50 text-slate-900 placeholder:text-slate-400 focus:bg-white focus:border-[#2563eb] focus:ring-0 resize-none p-4 font-medium"
                />
              </div>
            </section>

            {/* Submit Button */}
            <div className="pt-6 flex justify-end border-t border-slate-200">
              <button
                type="submit"
                disabled={evaluateInspection.isPending}
                className={cn(
                  'h-14 px-8 rounded-xl bg-amber-600 hover:bg-amber-700 text-white font-bold transition-colors flex items-center justify-center gap-2 shadow-lg shadow-amber-200',
                  evaluateInspection.isPending && 'opacity-70 cursor-not-allowed'
                )}
              >
                {evaluateInspection.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                    Elemzes folyamatban...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" aria-hidden="true" />
                    Vizsga kockazat elemzese
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Results Section */}
        {result && (
          <div className="mt-10 space-y-8">
            {/* Risk Gauge - Center */}
            <div className="flex flex-col items-center">
              <RiskGauge score={result.risk_score} risk={result.overall_risk} />
              <p className="mt-4 text-sm text-slate-500 text-center">
                {result.vehicle_info} &middot; {result.dtc_count} hibakod elemezve
              </p>
            </div>

            {/* Failing Items Grid */}
            {result.failing_items.length > 0 && (
              <section>
                <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-red-500" aria-hidden="true" />
                  Talalt problemas tetelek ({result.failing_items.length})
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {result.failing_items.map((item, index) => (
                    <InspectionCategoryCard key={`${item.category}-${index}`} item={item} />
                  ))}
                </div>
              </section>
            )}

            {/* Passing Categories */}
            {result.passing_categories.length > 0 && (
              <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
                <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <CheckCircle className="h-5 w-5 text-green-500" aria-hidden="true" />
                  Megfelelt kategoriak
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                  {result.passing_categories.map((category) => (
                    <div
                      key={category}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg bg-green-50 border border-green-100"
                    >
                      <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" aria-hidden="true" />
                      <span className="text-sm font-medium text-green-800">{category}</span>
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Recommendations */}
            {result.recommendations.length > 0 && (
              <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
                <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <Lightbulb className="h-5 w-5 text-amber-500" aria-hidden="true" />
                  Javaslatok
                </h3>
                <ul className="space-y-3">
                  {result.recommendations.map((rec, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <Gauge className="h-4 w-4 text-amber-500 flex-shrink-0 mt-0.5" aria-hidden="true" />
                      <p className="text-sm text-slate-700 leading-relaxed">{rec}</p>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {/* Total Estimated Fix Cost */}
            <section className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
              <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                <Banknote className="h-5 w-5 text-[#2563eb]" aria-hidden="true" />
                Becsult osszes javitasi koltseg
              </h3>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-black text-slate-900">
                  {costFormatter.format(result.estimated_total_fix_cost_min)}
                </span>
                <span className="text-lg text-slate-500 font-medium">-</span>
                <span className="text-3xl font-black text-slate-900">
                  {costFormatter.format(result.estimated_total_fix_cost_max)}
                </span>
              </div>
              <p className="text-sm text-slate-500 mt-2">
                A becsult koltseg tajekoztatasi jellegu. A vegso ar a szerviz es az alkatreszek aratol fugg.
              </p>
            </section>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-auto py-8 text-center text-slate-500 text-sm font-medium border-t border-slate-200 bg-white">
        <p>&copy; {currentYear} MechanicAI. Fejlett diagnosztikai algoritmusokkal mukodik.</p>
      </footer>
    </div>
  )
}
