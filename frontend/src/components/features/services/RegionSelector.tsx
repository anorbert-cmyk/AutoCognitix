/**
 * RegionSelector Component
 *
 * Dropdown for selecting Hungarian regions in the service comparison page.
 * Shows "Összes régió" as default option.
 */

import { MapPin } from 'lucide-react'
import type { RegionData } from '../../../data/regions'

interface RegionSelectorProps {
  regions: RegionData[]
  selectedRegion: string
  onChange: (region: string) => void
  shopCountByRegion?: Record<string, number>
}

export function RegionSelector({
  regions,
  selectedRegion,
  onChange,
  shopCountByRegion,
}: RegionSelectorProps) {
  return (
    <div className="relative">
      <label
        htmlFor="region-selector"
        className="block text-sm font-medium text-slate-700 mb-1.5"
      >
        <MapPin className="inline-block w-4 h-4 mr-1 -mt-0.5 text-slate-400" />
        R\u00E9gi\u00F3
      </label>
      <select
        id="region-selector"
        value={selectedRegion}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm transition-colors focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 hover:border-slate-400"
      >
        <option value="">\u00D6sszes r\u00E9gi\u00F3</option>
        {regions.map((region) => {
          const count = shopCountByRegion?.[region.id]
          const suffix = count !== undefined ? ` (${count})` : ''
          return (
            <option key={region.id} value={region.id}>
              {region.name}{suffix}
            </option>
          )
        })}
      </select>
    </div>
  )
}

export default RegionSelector
