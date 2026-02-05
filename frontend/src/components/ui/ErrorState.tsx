import { ReactNode } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  WifiOff,
  ServerOff,
  Clock,
  RefreshCw,
  Home,
  ArrowLeft,
  Search,
} from 'lucide-react'
import { ApiError } from '../../services/api'

/**
 * Error state type for determining display style
 */
export type ErrorType =
  | 'network'
  | 'server'
  | 'not_found'
  | 'unauthorized'
  | 'forbidden'
  | 'validation'
  | 'timeout'
  | 'rate_limit'
  | 'generic'

/**
 * Props for ErrorState component
 */
interface ErrorStateProps {
  /** Error object or message */
  error?: ApiError | Error | string | null
  /** Error type override (auto-detected from ApiError if not provided) */
  type?: ErrorType
  /** Custom title override */
  title?: string
  /** Custom message override */
  message?: string
  /** Retry callback */
  onRetry?: () => void
  /** Go back callback */
  onBack?: () => void
  /** Go home callback */
  onHome?: () => void
  /** Additional CSS classes */
  className?: string
  /** Compact mode for inline display */
  compact?: boolean
  /** Show technical details in development */
  showDetails?: boolean
  /** Custom actions */
  actions?: ReactNode
}

/**
 * Configuration for each error type
 */
const ERROR_CONFIG: Record<
  ErrorType,
  {
    icon: typeof AlertCircle
    iconColor: string
    bgColor: string
    borderColor: string
    title: string
    message: string
  }
> = {
  network: {
    icon: WifiOff,
    iconColor: 'text-orange-500',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
    title: 'Kapcsolati hiba',
    message: 'Nincs internetkapcsolat. Ellenorizze a halozati kapcsolatot es probalkozzon ujra.',
  },
  server: {
    icon: ServerOff,
    iconColor: 'text-red-500',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    title: 'Szerver hiba',
    message: 'A szerver atmeneti hibaba utkozott. Kerem, probalkozzon kesobb.',
  },
  not_found: {
    icon: Search,
    iconColor: 'text-gray-500',
    bgColor: 'bg-gray-50',
    borderColor: 'border-gray-200',
    title: 'Nem talalhato',
    message: 'A keresett eroforras nem talalhato.',
  },
  unauthorized: {
    icon: AlertTriangle,
    iconColor: 'text-yellow-500',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
    title: 'Bejelentkezes szukseges',
    message: 'A tartalom megtekintesehez jelentkezzen be.',
  },
  forbidden: {
    icon: AlertCircle,
    iconColor: 'text-red-500',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    title: 'Hozzaferes megtagadva',
    message: 'Nincs jogosultsaga a tartalom megtekintesehez.',
  },
  validation: {
    icon: AlertCircle,
    iconColor: 'text-yellow-500',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
    title: 'Ervenytelen adatok',
    message: 'A megadott adatok ervenytelenek. Kerem, ellenorizze es probalkozzon ujra.',
  },
  timeout: {
    icon: Clock,
    iconColor: 'text-orange-500',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
    title: 'Idotullepes',
    message: 'A keres tul sokaig tartott. Kerem, probalkozzon ujra.',
  },
  rate_limit: {
    icon: Clock,
    iconColor: 'text-orange-500',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
    title: 'Tul sok keres',
    message: 'Tul sok kerest kuldott. Kerem, varjon egy kicsit, majd probalkozzon ujra.',
  },
  generic: {
    icon: AlertCircle,
    iconColor: 'text-red-500',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    title: 'Hiba tortent',
    message: 'Varatlan hiba tortent. Kerem, probalkozzon kesobb.',
  },
}

/**
 * Detect error type from ApiError
 */
function detectErrorType(error: ApiError | Error | string | null | undefined): ErrorType {
  if (!error) return 'generic'

  if (typeof error === 'string') return 'generic'

  if (error instanceof ApiError) {
    // Check for network error
    if (error.isNetworkError) return 'network'

    // Check status code
    switch (error.status) {
      case 0:
        return 'network'
      case 401:
        return 'unauthorized'
      case 403:
        return 'forbidden'
      case 404:
        return 'not_found'
      case 422:
        return 'validation'
      case 429:
        return 'rate_limit'
      case 500:
      case 502:
      case 503:
        return 'server'
      case 504:
        return 'timeout'
      default:
        return 'generic'
    }
  }

  // Check error message for common patterns
  const message = error.message.toLowerCase()
  if (message.includes('network') || message.includes('fetch')) return 'network'
  if (message.includes('timeout')) return 'timeout'

  return 'generic'
}

/**
 * ErrorState component for displaying various error states.
 *
 * Features:
 * - Auto-detection of error type from ApiError
 * - Hungarian error messages
 * - Customizable actions
 * - Compact mode for inline display
 * - Development-only technical details
 *
 * Usage:
 *   <ErrorState error={error} onRetry={refetch} />
 *
 *   <ErrorState
 *     type="not_found"
 *     title="Jarmu nem talalhato"
 *     message="A megadott VIN szamhoz nem tartozik jarmu."
 *     onBack={() => navigate(-1)}
 *   />
 */
export default function ErrorState({
  error,
  type,
  title,
  message,
  onRetry,
  onBack,
  onHome,
  className = '',
  compact = false,
  showDetails = import.meta.env.DEV,
  actions,
}: ErrorStateProps) {
  // Determine error type
  const errorType = type || detectErrorType(error)
  const config = ERROR_CONFIG[errorType]

  // Get display text
  const displayTitle = title || config.title
  const displayMessage = message || (error instanceof ApiError ? error.detail : config.message)

  // Get technical details
  const technicalDetails =
    error instanceof ApiError
      ? {
          code: error.code,
          status: error.status,
          field: error.field,
        }
      : error instanceof Error
        ? { message: error.message }
        : null

  const Icon = config.icon

  if (compact) {
    return (
      <div
        className={`flex items-center gap-3 p-3 rounded-lg border ${config.bgColor} ${config.borderColor} ${className}`}
        role="alert"
      >
        <Icon className={`w-5 h-5 flex-shrink-0 ${config.iconColor}`} />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900">{displayTitle}</p>
          <p className="text-sm text-gray-600 truncate">{displayMessage}</p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="flex-shrink-0 p-1.5 rounded-lg hover:bg-white/50 transition-colors"
            title="Ujraprobalas"
          >
            <RefreshCw className="w-4 h-4 text-gray-600" />
          </button>
        )}
      </div>
    )
  }

  return (
    <div
      className={`rounded-lg border ${config.bgColor} ${config.borderColor} p-6 text-center ${className}`}
      role="alert"
    >
      <div
        className={`w-12 h-12 mx-auto mb-4 rounded-full ${config.bgColor} border ${config.borderColor} flex items-center justify-center`}
      >
        <Icon className={`w-6 h-6 ${config.iconColor}`} />
      </div>

      <h3 className="text-lg font-semibold text-gray-900 mb-2">{displayTitle}</h3>

      <p className="text-gray-600 mb-4 max-w-md mx-auto">{displayMessage}</p>

      {showDetails && technicalDetails && (
        <details className="mb-4 text-left max-w-sm mx-auto">
          <summary className="cursor-pointer text-sm text-gray-500 hover:text-gray-700">
            Technikai reszletek
          </summary>
          <pre className="mt-2 p-3 bg-white rounded text-xs text-gray-600 overflow-auto">
            {JSON.stringify(technicalDetails, null, 2)}
          </pre>
        </details>
      )}

      <div className="flex flex-wrap gap-3 justify-center">
        {actions}

        {onRetry && (
          <button
            onClick={onRetry}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Ujraprobalas
          </button>
        )}

        {onBack && (
          <button
            onClick={onBack}
            className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Vissza
          </button>
        )}

        {onHome && (
          <button
            onClick={onHome}
            className="inline-flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <Home className="w-4 h-4" />
            Foodalra
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * Inline error message component (lightweight)
 */
export function InlineError({
  message,
  className = '',
}: {
  message: string
  className?: string
}) {
  return (
    <div className={`flex items-center gap-2 text-red-600 text-sm ${className}`}>
      <AlertCircle className="w-4 h-4 flex-shrink-0" />
      <span>{message}</span>
    </div>
  )
}

/**
 * Form field error component
 */
export function FieldError({
  error,
  className = '',
}: {
  error?: string | null
  className?: string
}) {
  if (!error) return null

  return (
    <p className={`mt-1 text-sm text-red-600 ${className}`} role="alert">
      {error}
    </p>
  )
}
