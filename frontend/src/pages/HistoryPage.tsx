import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  History,
  Filter,
  Search,
  ChevronLeft,
  ChevronRight,
  Download,
  BarChart2,
  AlertCircle,
  X,
  Calendar,
  Car,
} from 'lucide-react'
import DiagnosisCard from '../components/DiagnosisCard'
import { useDiagnosisHistory, useDiagnosisStats, useDeleteDiagnosis } from '../services/hooks'
import { ErrorMessage, LoadingSpinner } from '../components/ui'
import { formatConfidenceScore } from '../services/diagnosisService'

interface HistoryFilters {
  vehicleMake: string
  vehicleModel: string
  vehicleYear: string
  dtcCode: string
  dateFrom: string
  dateTo: string
}

const initialFilters: HistoryFilters = {
  vehicleMake: '',
  vehicleModel: '',
  vehicleYear: '',
  dtcCode: '',
  dateFrom: '',
  dateTo: '',
}

export default function HistoryPage() {
  const [page, setPage] = useState(0)
  const [limit] = useState(10)
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState<HistoryFilters>(initialFilters)
  const [appliedFilters, setAppliedFilters] = useState<HistoryFilters>(initialFilters)
  const [showStats, setShowStats] = useState(false)

  // Fetch history with filters
  const {
    data: historyData,
    isLoading,
    error,
    refetch,
  } = useDiagnosisHistory({
    skip: page * limit,
    limit,
    vehicleMake: appliedFilters.vehicleMake || undefined,
    vehicleModel: appliedFilters.vehicleModel || undefined,
    vehicleYear: appliedFilters.vehicleYear ? parseInt(appliedFilters.vehicleYear) : undefined,
    dtcCode: appliedFilters.dtcCode || undefined,
    dateFrom: appliedFilters.dateFrom || undefined,
    dateTo: appliedFilters.dateTo || undefined,
  })

  // Fetch stats
  const { data: stats, isLoading: statsLoading } = useDiagnosisStats()

  // Delete mutation
  const deleteMutation = useDeleteDiagnosis()

  const handleApplyFilters = () => {
    setAppliedFilters(filters)
    setPage(0)
    setShowFilters(false)
  }

  const handleClearFilters = () => {
    setFilters(initialFilters)
    setAppliedFilters(initialFilters)
    setPage(0)
  }

  const handleDelete = async (id: string) => {
    if (window.confirm('Biztosan torolni szeretne ezt a diagnozist?')) {
      try {
        await deleteMutation.mutateAsync(id)
        refetch()
      } catch (error) {
        console.error('Delete failed:', error)
      }
    }
  }

  const hasActiveFilters = Object.values(appliedFilters).some((v) => v !== '')

  // Calculate total pages
  const totalPages = historyData ? Math.ceil(historyData.total / limit) : 0

  // Export functionality (simplified - in production would generate actual CSV/PDF)
  const handleExport = (format: 'csv' | 'pdf') => {
    if (!historyData?.items.length) return

    if (format === 'csv') {
      const headers = ['Datum', 'Jarmu', 'DTC kodok', 'Megbizhatosag']
      const rows = historyData.items.map((item) => [
        new Date(item.created_at).toLocaleDateString('hu-HU'),
        `${item.vehicle_year} ${item.vehicle_make} ${item.vehicle_model}`,
        item.dtc_codes.join('; '),
        formatConfidenceScore(item.confidence_score),
      ])

      const csvContent = [headers, ...rows].map((row) => row.join(',')).join('\n')
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
      const link = document.createElement('a')
      link.href = URL.createObjectURL(blob)
      link.download = `diagnozis_tortenelem_${new Date().toISOString().split('T')[0]}.csv`
      link.click()
    } else {
      // PDF export would require a library like jsPDF
      alert('PDF export hamarosan elerheto!')
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
              <History className="h-7 w-7 text-primary-600" />
              Diagnozis tortenelem
            </h1>
            <p className="text-gray-600 mt-1">
              Tekintse at korabbi diagnozisait es statisztikait
            </p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowStats(!showStats)}
              className={`btn-outline flex items-center gap-2 ${showStats ? 'bg-primary-50 border-primary-300' : ''}`}
            >
              <BarChart2 className="h-4 w-4" />
              Statisztikak
            </button>
            <Link to="/diagnosis" className="btn-primary">
              Uj diagnozis
            </Link>
          </div>
        </div>

        {/* Stats Panel */}
        {showStats && (
          <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <BarChart2 className="h-5 w-5 text-primary-600" />
              Osszesito statisztikak
            </h2>
            {statsLoading ? (
              <LoadingSpinner text="Statisztikak betoltese..." />
            ) : stats ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                <div className="text-center">
                  <p className="text-3xl font-bold text-primary-600">{stats.total_diagnoses}</p>
                  <p className="text-sm text-gray-600">Osszes diagnozis</p>
                </div>
                <div className="text-center">
                  <p className="text-3xl font-bold text-green-600">
                    {formatConfidenceScore(stats.avg_confidence)}
                  </p>
                  <p className="text-sm text-gray-600">Atlagos megbizhatosag</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Leggyakoribb jarmuvek</p>
                  {stats.most_diagnosed_vehicles.slice(0, 3).map((v, i) => (
                    <div key={i} className="flex justify-between text-sm">
                      <span className="text-gray-600 truncate">
                        {v.make} {v.model}
                      </span>
                      <span className="text-gray-900 font-medium">{v.count}</span>
                    </div>
                  ))}
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Leggyakoribb hibakodok</p>
                  {stats.most_common_dtcs.slice(0, 3).map((d, i) => (
                    <div key={i} className="flex justify-between text-sm">
                      <span className="text-primary-600 font-mono">{d.code}</span>
                      <span className="text-gray-900 font-medium">{d.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-center py-4">Nincs eleg adat a statisztikakhoz</p>
            )}
          </div>
        )}

        {/* Filters */}
        <div className="bg-white rounded-lg shadow-sm border mb-6">
          <div className="p-4 flex items-center justify-between border-b">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className="flex items-center gap-2 text-gray-700 hover:text-gray-900"
              >
                <Filter className="h-5 w-5" />
                <span className="font-medium">Szurok</span>
                {hasActiveFilters && (
                  <span className="px-2 py-0.5 bg-primary-100 text-primary-700 text-xs rounded-full">
                    Aktiv
                  </span>
                )}
              </button>
              {hasActiveFilters && (
                <button
                  onClick={handleClearFilters}
                  className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
                >
                  <X className="h-4 w-4" />
                  Torles
                </button>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleExport('csv')}
                disabled={!historyData?.items.length}
                className="text-sm text-gray-600 hover:text-gray-900 flex items-center gap-1 disabled:opacity-50"
              >
                <Download className="h-4 w-4" />
                CSV
              </button>
              <button
                onClick={() => handleExport('pdf')}
                disabled={!historyData?.items.length}
                className="text-sm text-gray-600 hover:text-gray-900 flex items-center gap-1 disabled:opacity-50"
              >
                <Download className="h-4 w-4" />
                PDF
              </button>
            </div>
          </div>

          {showFilters && (
            <div className="p-4 bg-gray-50 border-b">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    <Car className="h-4 w-4 inline mr-1" />
                    Gyarto
                  </label>
                  <input
                    type="text"
                    value={filters.vehicleMake}
                    onChange={(e) => setFilters({ ...filters, vehicleMake: e.target.value })}
                    placeholder="pl. Volkswagen"
                    className="input text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Modell</label>
                  <input
                    type="text"
                    value={filters.vehicleModel}
                    onChange={(e) => setFilters({ ...filters, vehicleModel: e.target.value })}
                    placeholder="pl. Golf"
                    className="input text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Evjarat</label>
                  <input
                    type="number"
                    value={filters.vehicleYear}
                    onChange={(e) => setFilters({ ...filters, vehicleYear: e.target.value })}
                    placeholder="pl. 2018"
                    min="1900"
                    max="2030"
                    className="input text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    <AlertCircle className="h-4 w-4 inline mr-1" />
                    DTC kod
                  </label>
                  <input
                    type="text"
                    value={filters.dtcCode}
                    onChange={(e) => setFilters({ ...filters, dtcCode: e.target.value.toUpperCase() })}
                    placeholder="pl. P0101"
                    maxLength={5}
                    className="input text-sm font-mono"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    <Calendar className="h-4 w-4 inline mr-1" />
                    Datumtol
                  </label>
                  <input
                    type="date"
                    value={filters.dateFrom}
                    onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value })}
                    className="input text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Datumig</label>
                  <input
                    type="date"
                    value={filters.dateTo}
                    onChange={(e) => setFilters({ ...filters, dateTo: e.target.value })}
                    className="input text-sm"
                  />
                </div>
              </div>
              <div className="mt-4 flex justify-end gap-3">
                <button onClick={() => setShowFilters(false)} className="btn-outline text-sm">
                  Megse
                </button>
                <button onClick={handleApplyFilters} className="btn-primary text-sm">
                  <Search className="h-4 w-4 mr-1" />
                  Alkalmazas
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Content */}
        {isLoading ? (
          <div className="flex justify-center py-12">
            <LoadingSpinner size="lg" text="Diagnozisok betoltese..." />
          </div>
        ) : error ? (
          <ErrorMessage error={error} onRetry={() => refetch()} className="mb-6" />
        ) : historyData && historyData.items.length > 0 ? (
          <>
            {/* Results info */}
            <div className="flex items-center justify-between mb-4">
              <p className="text-sm text-gray-600">
                {historyData.total} diagnozis talalat
                {hasActiveFilters && ' a megadott szurokkel'}
              </p>
            </div>

            {/* List */}
            <div className="space-y-4">
              {historyData.items.map((diagnosis) => (
                <DiagnosisCard
                  key={diagnosis.id}
                  diagnosis={diagnosis}
                  onDelete={handleDelete}
                  isDeleting={deleteMutation.isPending}
                />
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="mt-8 flex items-center justify-center gap-4">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="btn-outline flex items-center gap-1 disabled:opacity-50"
                >
                  <ChevronLeft className="h-4 w-4" />
                  Elozo
                </button>
                <span className="text-sm text-gray-600">
                  {page + 1} / {totalPages} oldal
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={!historyData.has_more}
                  className="btn-outline flex items-center gap-1 disabled:opacity-50"
                >
                  Kovetkezo
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="bg-white rounded-lg shadow-sm border p-12 text-center">
            <History className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {hasActiveFilters ? 'Nincs talalat' : 'Meg nincs diagnozis'}
            </h3>
            <p className="text-gray-600 mb-6">
              {hasActiveFilters
                ? 'Probalkozzon mas szurokkel vagy torlje a szuroket'
                : 'Inditson egy uj diagnozist a jarmuve hibainak feltarasahoz'}
            </p>
            {hasActiveFilters ? (
              <button onClick={handleClearFilters} className="btn-outline">
                Szurok torlese
              </button>
            ) : (
              <Link to="/diagnosis" className="btn-primary">
                Uj diagnozis inditasa
              </Link>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
