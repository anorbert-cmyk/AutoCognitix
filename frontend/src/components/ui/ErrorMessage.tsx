import { AlertCircle, RefreshCw, X, WifiOff, ServerOff, Clock } from 'lucide-react'
import { ApiError } from '../../services/api'

interface ErrorMessageProps {
  error: ApiError | Error | null
  onRetry?: () => void
  onDismiss?: () => void
  className?: string
  /** Show compact version */
  compact?: boolean
}

/**
 * Determine the error type and corresponding display
 */
function getErrorDisplay(error: ApiError | Error) {
  const isApiError = error instanceof ApiError

  if (isApiError && error.isNetworkError) {
    return {
      icon: WifiOff,
      title: 'Kapcsolati hiba',
      hint: 'Ellenorizze az internetkapcsolatot es probalkozzon ujra.',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200',
      iconColor: 'text-orange-500',
      textColor: 'text-orange-800',
    }
  }

  if (isApiError && error.status >= 500) {
    return {
      icon: ServerOff,
      title: 'Szerver hiba',
      hint: 'A szerver atmeneti hibaba utkozott. Kerem, probalkozzon kesobb.',
      bgColor: 'bg-red-50',
      borderColor: 'border-red-200',
      iconColor: 'text-red-500',
      textColor: 'text-red-800',
    }
  }

  if (isApiError && error.status === 429) {
    return {
      icon: Clock,
      title: 'Tul sok keres',
      hint: 'Kerem, varjon egy kicsit, majd probalkozzon ujra.',
      bgColor: 'bg-orange-50',
      borderColor: 'border-orange-200',
      iconColor: 'text-orange-500',
      textColor: 'text-orange-800',
    }
  }

  return {
    icon: AlertCircle,
    title: 'Hiba tortent',
    hint: null,
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    iconColor: 'text-red-500',
    textColor: 'text-red-800',
  }
}

/**
 * Reusable error message component with Hungarian translations.
 *
 * Features:
 * - Automatic error type detection
 * - Hungarian error messages
 * - Network/server error hints
 * - Retry and dismiss actions
 * - Compact mode for inline use
 */
export default function ErrorMessage({
  error,
  onRetry,
  onDismiss,
  className = '',
  compact = false,
}: ErrorMessageProps) {
  if (!error) return null

  const isApiError = error instanceof ApiError
  const message = isApiError ? error.detail : error.message
  const display = getErrorDisplay(error)
  const Icon = display.icon

  if (compact) {
    return (
      <div
        className={`flex items-center gap-2 text-sm ${display.textColor} ${className}`}
        role="alert"
      >
        <Icon className={`h-4 w-4 flex-shrink-0 ${display.iconColor}`} />
        <span className="truncate">{message}</span>
        {onRetry && (
          <button
            onClick={onRetry}
            className="flex-shrink-0 hover:opacity-80"
            title="Ujraprobalas"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    )
  }

  return (
    <div
      className={`rounded-lg border ${display.borderColor} ${display.bgColor} p-4 ${className}`}
      role="alert"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <Icon className={`h-5 w-5 ${display.iconColor} flex-shrink-0 mt-0.5`} />
        <div className="flex-1">
          <h3 className={`text-sm font-medium ${display.textColor}`}>
            {display.title}
          </h3>
          <p className={`mt-1 text-sm ${display.textColor} opacity-90`}>{message}</p>

          {display.hint && (
            <p className={`mt-2 text-xs ${display.textColor} opacity-75`}>
              {display.hint}
            </p>
          )}

          {onRetry && (
            <button
              onClick={onRetry}
              className={`mt-3 inline-flex items-center gap-1.5 text-sm font-medium ${display.textColor} hover:opacity-80 transition-opacity`}
            >
              <RefreshCw className="h-4 w-4" />
              Ujraprobalas
            </button>
          )}
        </div>

        {onDismiss && (
          <button
            onClick={onDismiss}
            className={`${display.iconColor} opacity-60 hover:opacity-100 transition-opacity`}
            aria-label="Bezaras"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>
    </div>
  )
}
