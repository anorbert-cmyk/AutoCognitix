import { Link } from 'react-router-dom'
import { Car, AlertCircle, Calendar, ChevronRight, Trash2 } from 'lucide-react'
import { formatConfidenceScore, formatDate } from '../services/diagnosisService'

export interface DiagnosisHistoryItemData {
  id: string
  vehicle_make: string
  vehicle_model: string
  vehicle_year: number
  vehicle_vin?: string | null
  dtc_codes: string[]
  symptoms_text?: string
  symptoms?: string
  confidence_score: number
  created_at: string
}

interface DiagnosisCardProps {
  diagnosis: DiagnosisHistoryItemData
  onDelete?: (id: string) => void
  isDeleting?: boolean
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-green-600 bg-green-100'
  if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100'
  return 'text-red-600 bg-red-100'
}

/**
 * Card component for displaying a diagnosis history item
 */
export default function DiagnosisCard({
  diagnosis,
  onDelete,
  isDeleting = false,
}: DiagnosisCardProps) {
  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (onDelete && !isDeleting) {
      onDelete(diagnosis.id)
    }
  }

  return (
    <Link
      to={`/diagnosis/${diagnosis.id}`}
      className="block bg-white rounded-lg shadow-sm border border-gray-200 hover:shadow-md hover:border-primary-200 transition-all duration-200"
    >
      <div className="p-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-4 mb-3">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary-100 rounded-lg">
              <Car className="h-5 w-5 text-primary-600" />
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">
                {diagnosis.vehicle_year} {diagnosis.vehicle_make} {diagnosis.vehicle_model}
              </h3>
              {diagnosis.vehicle_vin && (
                <p className="text-xs text-gray-500 font-mono mt-0.5">
                  VIN: {diagnosis.vehicle_vin}
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={`px-2.5 py-1 rounded-full text-xs font-medium ${getConfidenceColor(
                diagnosis.confidence_score
              )}`}
            >
              {formatConfidenceScore(diagnosis.confidence_score)}
            </span>
            {onDelete && (
              <button
                onClick={handleDelete}
                disabled={isDeleting}
                className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded transition-colors disabled:opacity-50"
                title="Diagnozis torlese"
              >
                <Trash2 className={`h-4 w-4 ${isDeleting ? 'animate-pulse' : ''}`} />
              </button>
            )}
          </div>
        </div>

        {/* DTC Codes */}
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle className="h-4 w-4 text-gray-400 flex-shrink-0" />
          <div className="flex flex-wrap gap-1.5">
            {diagnosis.dtc_codes.slice(0, 5).map((code) => (
              <span
                key={code}
                className="inline-flex items-center px-2 py-0.5 rounded bg-primary-100 text-primary-700 text-xs font-medium"
              >
                {code}
              </span>
            ))}
            {diagnosis.dtc_codes.length > 5 && (
              <span className="inline-flex items-center px-2 py-0.5 rounded bg-gray-100 text-gray-600 text-xs">
                +{diagnosis.dtc_codes.length - 5} tovabbi
              </span>
            )}
          </div>
        </div>

        {/* Symptoms preview */}
        {(diagnosis.symptoms_text || diagnosis.symptoms) && (
          <p className="text-sm text-gray-600 mb-3 line-clamp-2">
            {diagnosis.symptoms_text || diagnosis.symptoms}
          </p>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-100">
          <div className="flex items-center gap-1.5 text-xs text-gray-500">
            <Calendar className="h-3.5 w-3.5" />
            {formatDate(diagnosis.created_at)}
          </div>
          <div className="flex items-center gap-1 text-sm text-primary-600 font-medium">
            Reszletek
            <ChevronRight className="h-4 w-4" />
          </div>
        </div>
      </div>
    </Link>
  )
}
