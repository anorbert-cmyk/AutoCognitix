import { useParams, Link, useNavigate } from 'react-router-dom'
import {
  ArrowLeft,
  AlertCircle,
  Wrench,
  Car,
  BookOpen,
  Loader2,
  Search,
} from 'lucide-react'
import { useDTCDetail, useRelatedDTCCodes } from '../services/hooks'
import { ErrorMessage, LoadingSpinner } from '../components/ui'
import {
  getSeverityLabelHu,
  getCategoryNameHu,
  getCategoryFromCode,
} from '../services/dtcService'
import { DTCCodeDetail } from '../services/api'

// Helper to get severity badge color
function getSeverityColor(severity: string): string {
  const colors: Record<string, string> = {
    low: 'text-green-600 bg-green-100',
    medium: 'text-yellow-600 bg-yellow-100',
    high: 'text-orange-600 bg-orange-100',
    critical: 'text-red-600 bg-red-100',
  }
  return colors[severity] || 'text-gray-600 bg-gray-100'
}

// Component to display DTC details content
function DTCContent({ dtc, code }: { dtc: DTCCodeDetail; code: string }) {
  const { data: relatedCodes, isLoading: relatedLoading } = useRelatedDTCCodes(code)
  const category = getCategoryFromCode(code)

  return (
    <>
      {/* Header */}
      <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="text-3xl font-mono font-bold text-primary-600">
                {code}
              </span>
              <span
                className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(
                  dtc.severity
                )}`}
              >
                {getSeverityLabelHu(dtc.severity)}
              </span>
            </div>
            <h1 className="text-xl font-medium text-gray-900">
              {dtc.description_hu || dtc.description_en}
            </h1>
            {dtc.description_hu && (
              <p className="text-gray-500">{dtc.description_en}</p>
            )}
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {category && (
            <span className="inline-flex items-center px-3 py-1 rounded bg-primary-50 text-primary-700 text-sm font-medium">
              {getCategoryNameHu(category)}
            </span>
          )}
          {dtc.system && (
            <span className="inline-flex items-center px-3 py-1 rounded bg-gray-100 text-gray-700 text-sm">
              {dtc.system}
            </span>
          )}
          {dtc.is_generic ? (
            <span className="inline-flex items-center px-3 py-1 rounded bg-blue-100 text-blue-700 text-sm">
              Generikus kod
            </span>
          ) : (
            <span className="inline-flex items-center px-3 py-1 rounded bg-purple-100 text-purple-700 text-sm">
              Gyarto specifikus
            </span>
          )}
        </div>
      </div>

      {/* Content grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Symptoms */}
        {dtc.symptoms && dtc.symptoms.length > 0 && (
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-yellow-500" />
              Tunetek
            </h2>
            <ul className="space-y-2">
              {dtc.symptoms.map((symptom, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-yellow-500 mt-1">&#8226;</span>
                  <span className="text-gray-700">{symptom}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Possible Causes */}
        {dtc.possible_causes && dtc.possible_causes.length > 0 && (
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Wrench className="h-5 w-5 text-primary-500" />
              Lehetseges okok
            </h2>
            <ul className="space-y-2">
              {dtc.possible_causes.map((cause, index) => (
                <li key={index} className="flex items-start gap-2">
                  <span className="text-primary-500 mt-1">&#8226;</span>
                  <span className="text-gray-700">{cause}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Diagnostic Steps */}
        {dtc.diagnostic_steps && dtc.diagnostic_steps.length > 0 && (
          <div className="card p-6 md:col-span-2">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <BookOpen className="h-5 w-5 text-green-500" />
              Diagnosztikai lepesek
            </h2>
            <ol className="space-y-3">
              {dtc.diagnostic_steps.map((step, index) => (
                <li key={index} className="flex items-start gap-3">
                  <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-100 text-primary-700 text-sm font-medium flex items-center justify-center">
                    {index + 1}
                  </span>
                  <span className="text-gray-700">{step}</span>
                </li>
              ))}
            </ol>
          </div>
        )}

        {/* Related Codes */}
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Kapcsolodo hibakodok
          </h2>
          {relatedLoading ? (
            <LoadingSpinner size="sm" text="Betoltes..." />
          ) : relatedCodes && relatedCodes.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {relatedCodes.map((related) => (
                <Link
                  key={related.code}
                  to={`/dtc/${related.code}`}
                  className="inline-flex flex-col items-start px-3 py-2 rounded bg-primary-50 text-primary-700 font-mono font-medium hover:bg-primary-100 transition-colors"
                >
                  <span>{related.code}</span>
                  {related.description_hu && (
                    <span className="text-xs text-primary-500 font-sans font-normal">
                      {related.description_hu.substring(0, 30)}...
                    </span>
                  )}
                </Link>
              ))}
            </div>
          ) : dtc.related_codes && dtc.related_codes.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {dtc.related_codes.map((relatedCode) => (
                <Link
                  key={relatedCode}
                  to={`/dtc/${relatedCode}`}
                  className="inline-flex items-center px-3 py-2 rounded bg-primary-50 text-primary-700 font-mono font-medium hover:bg-primary-100 transition-colors"
                >
                  {relatedCode}
                </Link>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">Nincs kapcsolodo hibakod.</p>
          )}
        </div>

        {/* Common Vehicles */}
        {dtc.common_vehicles && dtc.common_vehicles.length > 0 && (
          <div className="card p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Car className="h-5 w-5 text-gray-500" />
              Gyakran erintett jarmuvek
            </h2>
            <ul className="space-y-2">
              {dtc.common_vehicles.map((vehicle, index) => (
                <li key={index} className="text-gray-700">
                  {vehicle}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Freeze Frame Data */}
        {dtc.freeze_frame_data && dtc.freeze_frame_data.length > 0 && (
          <div className="card p-6 md:col-span-2">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Search className="h-5 w-5 text-blue-500" />
              Freeze Frame adatok
            </h2>
            <div className="flex flex-wrap gap-2">
              {dtc.freeze_frame_data.map((data, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-3 py-1 rounded bg-blue-50 text-blue-700 text-sm"
                >
                  {data}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* CTA */}
      <div className="mt-8 text-center">
        <Link
          to={`/diagnosis?dtc=${code}`}
          className="btn-primary"
        >
          Diagnozis inditasa ezzel a koddal
        </Link>
      </div>
    </>
  )
}

export default function DTCDetailPage() {
  const { code } = useParams<{ code: string }>()
  const navigate = useNavigate()

  const { data: dtc, isLoading, error, refetch } = useDTCDetail(code)

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="h-12 w-12 animate-spin text-primary-600 mb-4" />
            <p className="text-gray-600">DTC kod betoltese...</p>
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
            Vissza
          </Link>

          <ErrorMessage
            error={error}
            onRetry={() => refetch()}
            className="mb-6"
          />

          <div className="text-center py-12">
            <AlertCircle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              A hibakod nem talalhato
            </h2>
            <p className="text-gray-600 mb-6">
              Sajnos a(z) <span className="font-mono font-bold">{code}</span> hibakodot nem talaltuk az adatbazisban.
            </p>
            <div className="flex justify-center gap-4">
              <button
                onClick={() => navigate(-1)}
                className="btn-outline"
              >
                Vissza
              </button>
              <Link to="/diagnosis" className="btn-primary">
                Diagnozis inditasa
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (!dtc || !code) {
    return null
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
        {/* Back button */}
        <button
          onClick={() => navigate(-1)}
          className="inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6"
        >
          <ArrowLeft className="h-4 w-4" />
          Vissza
        </button>

        <DTCContent dtc={dtc} code={code} />
      </div>
    </div>
  )
}
