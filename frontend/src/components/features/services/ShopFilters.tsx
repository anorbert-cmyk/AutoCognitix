/**
 * ShopFilters Component
 *
 * Filter controls for the service comparison page.
 * Horizontal layout on desktop, stacked on mobile.
 */

import { SlidersHorizontal } from 'lucide-react'

export interface ShopFilterValues {
  vehicle_make: string
  service_type: string
  sort_by: string
}

interface ShopFiltersProps {
  filters: ShopFilterValues
  onFilterChange: (key: keyof ShopFilterValues, value: string) => void
  vehicleMakes?: string[]
}

const SERVICE_TYPES = [
  { value: '', label: '\u00D6sszes t\u00EDpus' },
  { value: 'general', label: '\u00C1ltal\u00E1nos' },
  { value: 'diagnostics', label: 'Diagnosztika' },
  { value: 'bodywork', label: 'Karossz\u00E9ria' },
  { value: 'inspection', label: 'M\u0171szaki vizsga' },
  { value: 'electric', label: 'Elektromos' },
]

const SORT_OPTIONS = [
  { value: 'rating', label: '\u00C9rt\u00E9kel\u00E9s' },
  { value: 'price', label: '\u00C1r' },
  { value: 'distance', label: 'T\u00E1vols\u00E1g' },
]

const DEFAULT_MAKES = [
  'Volkswagen',
  'Opel',
  'Suzuki',
  'Ford',
  'Toyota',
  'Skoda',
  'BMW',
  'Mercedes-Benz',
  'Audi',
  'Renault',
  'Peugeot',
  'Fiat',
  'Hyundai',
  'Kia',
  'Honda',
  'Mazda',
  'Nissan',
  'Volvo',
  'Citro\u00EBn',
  'Dacia',
]

export function ShopFilters({ filters, onFilterChange, vehicleMakes }: ShopFiltersProps) {
  const makes = vehicleMakes || DEFAULT_MAKES

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
      <div className="flex items-center gap-2 mb-3">
        <SlidersHorizontal className="w-4 h-4 text-slate-400" />
        <span className="text-sm font-semibold text-slate-700">Sz\u0171r\u00E9s</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {/* Vehicle Make */}
        <div>
          <label
            htmlFor="filter-make"
            className="block text-xs font-medium text-slate-500 mb-1"
          >
            J\u00E1rm\u0171 m\u00E1rka
          </label>
          <select
            id="filter-make"
            value={filters.vehicle_make}
            onChange={(e) => onFilterChange('vehicle_make', e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm transition-colors focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 hover:border-slate-400"
          >
            <option value="">\u00D6sszes m\u00E1rka</option>
            {makes.map((make) => (
              <option key={make} value={make}>
                {make}
              </option>
            ))}
          </select>
        </div>

        {/* Service Type */}
        <div>
          <label
            htmlFor="filter-service-type"
            className="block text-xs font-medium text-slate-500 mb-1"
          >
            Szerviz t\u00EDpus
          </label>
          <select
            id="filter-service-type"
            value={filters.service_type}
            onChange={(e) => onFilterChange('service_type', e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm transition-colors focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 hover:border-slate-400"
          >
            {SERVICE_TYPES.map((type) => (
              <option key={type.value} value={type.value}>
                {type.label}
              </option>
            ))}
          </select>
        </div>

        {/* Sort By */}
        <div>
          <label
            htmlFor="filter-sort"
            className="block text-xs font-medium text-slate-500 mb-1"
          >
            Rendez\u00E9s
          </label>
          <select
            id="filter-sort"
            value={filters.sort_by}
            onChange={(e) => onFilterChange('sort_by', e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm transition-colors focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 hover:border-slate-400"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  )
}

export default ShopFilters
