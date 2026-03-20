/**
 * ServiceComparisonPage
 *
 * Szerviz Összehasonlítás — full page with split view layout.
 * Desktop: left sidebar (filters + shop list) + right map placeholder.
 * Mobile: stacked filters → shop list → map placeholder.
 *
 * Leaflet is NOT installed yet — map area shows a placeholder.
 */

import { useCallback, useMemo, useState } from 'react'
import { Loader2, MapPin, Search, Wrench } from 'lucide-react'
import { regions } from '../data/regions'
import { RegionSelector } from '../components/features/services/RegionSelector'
import { ShopFilters } from '../components/features/services/ShopFilters'
import { ShopCard } from '../components/features/services/ShopCard'
import { useServiceShops } from '../services/hooks/useServiceShops'
import type { ShopFilterValues } from '../components/features/services/ShopFilters'
import type { ServiceSearchParams } from '../services/serviceShopService'

export function ServiceComparisonPage() {
  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const [selectedRegion, setSelectedRegion] = useState('')
  const [filters, setFilters] = useState<ShopFilterValues>({
    vehicle_make: '',
    service_type: '',
    sort_by: 'rating',
  })
  const [_selectedShopId, setSelectedShopId] = useState<string | null>(null)

  // ---------------------------------------------------------------------------
  // Build search params from state
  // ---------------------------------------------------------------------------
  const searchParams = useMemo<ServiceSearchParams>(() => ({
    region: selectedRegion || undefined,
    vehicle_make: filters.vehicle_make || undefined,
    service_type: filters.service_type || undefined,
    sort_by: (filters.sort_by as 'rating' | 'price' | 'distance') || 'rating',
    limit: 50,
    offset: 0,
  }), [selectedRegion, filters])

  // ---------------------------------------------------------------------------
  // Data fetching
  // ---------------------------------------------------------------------------
  const { data, isLoading, isError, error } = useServiceShops(searchParams)

  const shops = data?.shops || []
  const totalCount = data?.total || 0

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleRegionChange = useCallback((region: string) => {
    setSelectedRegion(region)
    setSelectedShopId(null)
  }, [])

  const handleFilterChange = useCallback(
    (key: keyof ShopFilterValues, value: string) => {
      setFilters((prev) => ({ ...prev, [key]: value }))
      setSelectedShopId(null)
    },
    []
  )

  const handleShopClick = useCallback((shopId: string) => {
    setSelectedShopId((prev) => (prev === shopId ? null : shopId))
  }, [])

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-slate-50">
      {/* Page Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-4 sm:py-5">
          <div className="flex items-center gap-3 mb-1">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-[#0D1B2A] text-white shadow-lg shadow-[#0D1B2A]/20">
              <Wrench className="w-5 h-5" />
            </div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900">
              Szerviz \u00D6sszehasonl\u00EDt\u00E1s
            </h1>
          </div>
          <p className="text-sm text-slate-500 ml-[52px]">
            Keresd meg a legjobb szervizeket Magyarorsz\u00E1g-szerte, \u00E9rt\u00E9kel\u00E9sek \u00E9s \u00E1rak alapj\u00E1n
          </p>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 py-4 sm:py-6">
        {/* Filters Section */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-4">
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
              <RegionSelector
                regions={regions}
                selectedRegion={selectedRegion}
                onChange={handleRegionChange}
              />
            </div>
          </div>
          <div className="lg:col-span-3">
            <ShopFilters
              filters={filters}
              onFilterChange={handleFilterChange}
            />
          </div>
        </div>

        {/* Result Count Badge */}
        <div className="flex items-center gap-2 mb-4">
          <Search className="w-4 h-4 text-slate-400" />
          <span className="text-sm font-medium text-slate-600">
            <span className="font-bold text-slate-900">{totalCount}</span> szerviz tal\u00E1lat
          </span>
          {selectedRegion && (
            <span className="text-xs text-slate-400">
              \u2014 {regions.find((r) => r.id === selectedRegion)?.name || selectedRegion}
            </span>
          )}
        </div>

        {/* Split View: Shop List + Map */}
        <div className="flex flex-col lg:flex-row gap-4" style={{ minHeight: '600px' }}>
          {/* Left: Shop List */}
          <div className="w-full lg:w-[400px] lg:flex-shrink-0 flex flex-col">
            <div className="flex-1 overflow-y-auto space-y-3 pr-1 max-h-[calc(100vh-280px)] lg:max-h-[700px]">
              {/* Loading State */}
              {isLoading && (
                <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                  <Loader2 className="w-8 h-8 animate-spin mb-3" />
                  <span className="text-sm font-medium">Szervizek bet\u00F6lt\u00E9se...</span>
                </div>
              )}

              {/* Error State */}
              {isError && !isLoading && (
                <div className="flex flex-col items-center justify-center py-16 text-red-500">
                  <span className="text-sm font-medium mb-1">Hiba t\u00F6rt\u00E9nt a bet\u00F6lt\u00E9s sor\u00E1n</span>
                  <span className="text-xs text-red-400">
                    {error instanceof Error ? error.message : 'Ismeretlen hiba'}
                  </span>
                </div>
              )}

              {/* Empty State */}
              {!isLoading && !isError && shops.length === 0 && (
                <div className="flex flex-col items-center justify-center py-16 text-slate-400">
                  <Search className="w-8 h-8 mb-3" />
                  <span className="text-sm font-medium text-slate-500">
                    Nincs tal\u00E1lat a sz\u0171r\u00E9si felt\u00E9teleknek megfelel\u0151en
                  </span>
                  <span className="text-xs text-slate-400 mt-1">
                    Pr\u00F3b\u00E1lj m\u00E1s r\u00E9gi\u00F3t vagy sz\u0171r\u0151t v\u00E1lasztani
                  </span>
                </div>
              )}

              {/* Shop Cards */}
              {!isLoading &&
                shops.map((shop) => (
                  <ShopCard
                    key={shop.id}
                    shop={shop}
                    onClick={() => handleShopClick(shop.id)}
                  />
                ))}
            </div>
          </div>

          {/* Right: Map Placeholder */}
          <div className="flex-1 min-h-[400px] lg:min-h-0">
            <div className="w-full h-full min-h-[400px] rounded-xl border border-slate-200 bg-white shadow-sm flex flex-col items-center justify-center text-slate-400">
              <div className="flex items-center justify-center w-16 h-16 rounded-full bg-slate-100 mb-4">
                <MapPin className="w-8 h-8 text-slate-300" />
              </div>
              <span className="text-sm font-medium text-slate-500">
                T\u00E9rk\u00E9p bet\u00F6lt\u00E9se...
              </span>
              <span className="text-xs text-slate-400 mt-1">
                Leaflet t\u00E9rk\u00E9p hamarosan el\u00E9rhet\u0151
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ServiceComparisonPage
