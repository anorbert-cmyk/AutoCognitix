import { Link, useParams, useNavigate } from 'react-router-dom'
import {
  AlertTriangle,
  CheckCircle,
  Wrench,
  DollarSign,
  ArrowLeft,
  Clock,
  ExternalLink,
  AlertCircle,
  FileText,
  Loader2,
  Printer,
  Download,
  Share2,
} from 'lucide-react'
import { useToast } from '../contexts/ToastContext'
import { useDiagnosisDetail, useVehicleRecalls } from '../services/hooks'
import { ErrorMessage, LoadingSpinner } from '../components/ui'
import {
  formatConfidenceScore,
  getConfidenceColorClass,
  getDifficultyLabelHu,
  getDifficultyColorClass,
  formatCostRange,
  formatTime,
  formatDate,
} from '../services/diagnosisService'
import { DiagnosisResponse, Recall, Source } from '../services/api'

// Helper functions
function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-green-600 bg-green-100'
  if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100'
  return 'text-red-600 bg-red-100'
}

function getSourceIcon(type: Source['type']): JSX.Element {
  switch (type) {
    case 'tsb':
      return <FileText className="h-4 w-4" />
    case 'manual':
      return <FileText className="h-4 w-4" />
    case 'forum':
      return <ExternalLink className="h-4 w-4" />
    case 'video':
      return <ExternalLink className="h-4 w-4" />
    case 'database':
      return <AlertCircle className="h-4 w-4" />
    default:
      return <FileText className="h-4 w-4" />
  }
}

function getSourceLabel(type: Source['type']): string {
  switch (type) {
    case 'tsb':
      return 'Technikai Szerviz Bullettin'
    case 'manual':
      return 'Szerviz kezikoynv'
    case 'forum':
      return 'Forum'
    case 'video':
      return 'Video'
    case 'database':
      return 'Adatbazis'
    default:
      return 'Forras'
  }
}

function RecallCard({ recall }: { recall: Recall }) {
  return (
    <div className="border rounded-lg p-4 bg-orange-50 border-orange-200">
      <div className="flex items-start gap-3">
        <AlertTriangle className="h-5 w-5 text-orange-500 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <h4 className="font-medium text-gray-900">{recall.component}</h4>
          <p className="text-sm text-gray-600 mt-1">{recall.summary}</p>
          {recall.consequence && (
            <p className="text-sm text-orange-700 mt-2">
              <strong>Kovetkezmeny:</strong> {recall.consequence}
            </p>
          )}
          {recall.remedy && (
            <p className="text-sm text-green-700 mt-1">
              <strong>Megoldas:</strong> {recall.remedy}
            </p>
          )}
          <p className="text-xs text-gray-500 mt-2">
            Kampany: {recall.campaign_number}
          </p>
        </div>
      </div>
    </div>
  )
}

function DiagnosisContent({ result }: { result: DiagnosisResponse }) {
  // Fetch recalls for the vehicle
  const { data: recalls, isLoading: recallsLoading } = useVehicleRecalls(
    result.vehicle_make,
    result.vehicle_model,
    result.vehicle_year
  )

  return (
    <>
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              {result.vehicle_year} {result.vehicle_make} {result.vehicle_model}
            </h1>
            <p className="text-gray-600">
              Hibakodok: {result.dtc_codes.join(', ')}
            </p>
            <p className="text-sm text-gray-500 mt-1">
              {formatDate(result.created_at)}
            </p>
          </div>
          <div
            className={`inline-flex items-center gap-2 px-4 py-2 rounded-full font-medium ${getConfidenceColor(
              result.confidence_score
            )}`}
          >
            <CheckCircle className="h-5 w-5" />
            {formatConfidenceScore(result.confidence_score)} megbizhatosag
          </div>
        </div>
      </div>

      {/* Symptoms Summary */}
      {result.symptoms && (
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
            <FileText className="h-5 w-5 text-gray-500" />
            Leirtt tunetek
          </h2>
          <p className="text-gray-700">{result.symptoms}</p>
        </div>
      )}

      {/* Probable Causes */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-yellow-500" />
          Valoszinu okok
        </h2>
        <div className="space-y-4">
          {result.probable_causes.map((cause, index) => (
            <div key={index} className="card p-6">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    {cause.title}
                  </h3>
                  <p className="text-gray-600 mb-4">{cause.description}</p>
                  <div className="flex flex-wrap gap-2">
                    {cause.related_dtc_codes.map((code) => (
                      <Link
                        key={code}
                        to={`/dtc/${code}`}
                        className="inline-flex items-center px-2 py-1 rounded bg-primary-100 text-primary-700 text-sm font-medium hover:bg-primary-200"
                      >
                        {code}
                      </Link>
                    ))}
                    {cause.components.map((component) => (
                      <span
                        key={component}
                        className="inline-flex items-center px-2 py-1 rounded bg-gray-100 text-gray-700 text-sm"
                      >
                        {component}
                      </span>
                    ))}
                  </div>
                </div>
                <div
                  className={`px-3 py-1 rounded-full text-sm font-medium ${getConfidenceColor(
                    cause.confidence
                  )}`}
                >
                  {formatConfidenceScore(cause.confidence)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended Repairs */}
      <div className="mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Wrench className="h-5 w-5 text-primary-500" />
          Javasolt javitasok
        </h2>
        <div className="space-y-4">
          {result.recommended_repairs.map((repair, index) => (
            <div key={index} className="card p-6">
              <div className="flex items-start justify-between gap-4 mb-4">
                <h3 className="text-lg font-medium text-gray-900">
                  {repair.title}
                </h3>
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${getDifficultyColorClass(
                    repair.difficulty
                  )}`}
                >
                  {getDifficultyLabelHu(repair.difficulty)}
                </span>
              </div>
              <p className="text-gray-600 mb-4">{repair.description}</p>

              <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                {/* Cost estimate */}
                <div className="flex items-center gap-2 text-gray-700">
                  <DollarSign className="h-5 w-5 text-gray-400" />
                  <span className="font-medium">
                    {formatCostRange(
                      repair.estimated_cost_min,
                      repair.estimated_cost_max,
                      repair.estimated_cost_currency
                    )}
                  </span>
                </div>

                {/* Time estimate */}
                {repair.estimated_time_minutes && (
                  <div className="flex items-center gap-2 text-gray-700">
                    <Clock className="h-5 w-5 text-gray-400" />
                    <span>{formatTime(repair.estimated_time_minutes)}</span>
                  </div>
                )}
              </div>

              {/* Parts needed */}
              {repair.parts_needed.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm font-medium text-gray-700 mb-2">
                    Szukseges alkatreszek:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {repair.parts_needed.map((part) => (
                      <span
                        key={part}
                        className="inline-flex items-center px-2 py-1 rounded bg-gray-100 text-gray-700 text-sm"
                      >
                        {part}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Recalls Section */}
      {(recalls && recalls.length > 0) && (
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-orange-500" />
            Visszahivasok
            <span className="ml-2 px-2 py-0.5 bg-orange-100 text-orange-700 text-sm rounded-full">
              {recalls.length}
            </span>
          </h2>
          {recallsLoading ? (
            <LoadingSpinner text="Visszahivasok betoltese..." />
          ) : (
            <div className="space-y-4">
              {recalls.map((recall) => (
                <RecallCard key={recall.campaign_number} recall={recall} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Sources */}
      {result.sources && result.sources.length > 0 && (
        <div className="mb-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <FileText className="h-5 w-5 text-gray-500" />
            Forrasok
          </h2>
          <div className="grid md:grid-cols-2 gap-4">
            {result.sources.map((source, index) => (
              <div key={index} className="card p-4">
                <div className="flex items-start gap-3">
                  <div className="text-gray-400">
                    {getSourceIcon(source.type)}
                  </div>
                  <div className="flex-1">
                    <p className="text-xs text-gray-500 uppercase tracking-wide">
                      {getSourceLabel(source.type)}
                    </p>
                    <p className="font-medium text-gray-900">{source.title}</p>
                    {source.url && (
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm text-primary-600 hover:text-primary-700 inline-flex items-center gap-1 mt-1"
                      >
                        Megtekintes
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </div>
                  <div className={`text-sm ${getConfidenceColorClass(source.relevance_score)}`}>
                    {formatConfidenceScore(source.relevance_score)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="mt-8 flex flex-wrap justify-center gap-4 print:hidden">
        <Link to="/diagnosis" className="btn-primary">
          Uj diagnozis inditasa
        </Link>
      </div>
    </>
  )
}

export default function ResultPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const toast = useToast()

  const { data: result, isLoading, error, refetch } = useDiagnosisDetail(id)

  // Print function
  const handlePrint = () => {
    window.print()
    toast.info('Nyomtatas indult')
  }

  // Export to JSON function
  const handleExportJSON = () => {
    if (!result) return

    const dataStr = JSON.stringify(result, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `diagnosis-${result.id}-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
    toast.success('Diagnozis exportalva JSON formatumban')
  }

  // Share function (copies link to clipboard)
  const handleShare = async () => {
    const url = window.location.href
    try {
      if (navigator.share) {
        await navigator.share({
          title: `Diagnozis: ${result?.vehicle_year} ${result?.vehicle_make} ${result?.vehicle_model}`,
          url,
        })
        toast.success('Megosztas sikeres!')
      } else {
        await navigator.clipboard.writeText(url)
        toast.success('Link masolva a vagolapra!')
      }
    } catch (err) {
      // User cancelled share or error occurred
      console.error('Share failed:', err)
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="h-12 w-12 animate-spin text-primary-600 mb-4" />
            <p className="text-gray-600">Diagnozis betoltese...</p>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <Link
            to="/diagnosis"
            className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6"
          >
            <ArrowLeft className="h-4 w-4" />
            Uj diagnozis
          </Link>

          <ErrorMessage
            error={error}
            onRetry={() => refetch()}
            className="mb-6"
          />

          <div className="text-center py-12">
            <AlertCircle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              A diagnozis nem talalhato
            </h2>
            <p className="text-gray-600 mb-6">
              Lehetseges, hogy a diagnozis torolve lett vagy nem letezik.
            </p>
            <button
              onClick={() => navigate('/diagnosis')}
              className="btn-primary"
            >
              Uj diagnozis keszitese
            </button>
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return null
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
        {/* Header with actions */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6 print:hidden">
          <Link
            to="/diagnosis"
            className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft className="h-4 w-4" />
            Uj diagnozis
          </Link>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrint}
              className="inline-flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              title="Nyomtatas"
            >
              <Printer className="h-4 w-4" />
              <span className="hidden sm:inline">Nyomtatas</span>
            </button>
            <button
              onClick={handleExportJSON}
              className="inline-flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              title="Exportalas"
            >
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline">Exportalas</span>
            </button>
            <button
              onClick={handleShare}
              className="inline-flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              title="Megosztas"
            >
              <Share2 className="h-4 w-4" />
              <span className="hidden sm:inline">Megosztas</span>
            </button>
          </div>
        </div>

        <DiagnosisContent result={result} />
      </div>
    </div>
  )
}
