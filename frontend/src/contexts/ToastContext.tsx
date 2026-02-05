/**
 * Toast Notification Context
 *
 * Provides a global toast notification system for user feedback.
 * Supports success, error, warning, and info messages.
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  ReactNode,
  useEffect,
} from 'react'
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react'

// =============================================================================
// Types
// =============================================================================

export type ToastType = 'success' | 'error' | 'warning' | 'info'

export interface Toast {
  id: string
  type: ToastType
  title?: string
  message: string
  duration?: number
  dismissible?: boolean
}

interface ToastContextValue {
  toasts: Toast[]
  addToast: (toast: Omit<Toast, 'id'>) => string
  removeToast: (id: string) => void
  clearToasts: () => void
  // Convenience methods
  success: (message: string, title?: string) => string
  error: (message: string, title?: string) => string
  warning: (message: string, title?: string) => string
  info: (message: string, title?: string) => string
}

// =============================================================================
// Context
// =============================================================================

const ToastContext = createContext<ToastContextValue | null>(null)

// =============================================================================
// Toast Configuration
// =============================================================================

const TOAST_CONFIG: Record<
  ToastType,
  {
    icon: typeof CheckCircle
    iconColor: string
    bgColor: string
    borderColor: string
    progressColor: string
  }
> = {
  success: {
    icon: CheckCircle,
    iconColor: 'text-green-500',
    bgColor: 'bg-green-50',
    borderColor: 'border-green-200',
    progressColor: 'bg-green-500',
  },
  error: {
    icon: AlertCircle,
    iconColor: 'text-red-500',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
    progressColor: 'bg-red-500',
  },
  warning: {
    icon: AlertTriangle,
    iconColor: 'text-yellow-500',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
    progressColor: 'bg-yellow-500',
  },
  info: {
    icon: Info,
    iconColor: 'text-blue-500',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
    progressColor: 'bg-blue-500',
  },
}

const DEFAULT_DURATION = 5000 // 5 seconds

// =============================================================================
// Single Toast Component
// =============================================================================

interface ToastItemProps {
  toast: Toast
  onRemove: (id: string) => void
}

function ToastItem({ toast, onRemove }: ToastItemProps) {
  const config = TOAST_CONFIG[toast.type]
  const Icon = config.icon
  const duration = toast.duration ?? DEFAULT_DURATION
  const dismissible = toast.dismissible ?? true

  useEffect(() => {
    if (duration > 0) {
      const timer = setTimeout(() => {
        onRemove(toast.id)
      }, duration)
      return () => clearTimeout(timer)
    }
  }, [toast.id, duration, onRemove])

  return (
    <div
      className={`relative flex items-start gap-3 w-full max-w-sm p-4 rounded-lg border shadow-lg ${config.bgColor} ${config.borderColor} animate-slide-in-right`}
      role="alert"
      aria-live="polite"
    >
      <Icon className={`w-5 h-5 flex-shrink-0 mt-0.5 ${config.iconColor}`} />

      <div className="flex-1 min-w-0">
        {toast.title && (
          <p className="font-medium text-gray-900 mb-0.5">{toast.title}</p>
        )}
        <p className="text-sm text-gray-700">{toast.message}</p>
      </div>

      {dismissible && (
        <button
          onClick={() => onRemove(toast.id)}
          className="flex-shrink-0 p-1 rounded hover:bg-black/5 transition-colors"
          aria-label="Bezaras"
        >
          <X className="w-4 h-4 text-gray-500" />
        </button>
      )}

      {/* Progress bar */}
      {duration > 0 && (
        <div className="absolute bottom-0 left-0 right-0 h-1 overflow-hidden rounded-b-lg">
          <div
            className={`h-full ${config.progressColor} animate-toast-progress`}
            style={{
              animationDuration: `${duration}ms`,
            }}
          />
        </div>
      )}
    </div>
  )
}

// =============================================================================
// Toast Container Component
// =============================================================================

function ToastContainer({ toasts, onRemove }: { toasts: Toast[]; onRemove: (id: string) => void }) {
  if (toasts.length === 0) return null

  return (
    <div
      className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none"
      aria-label="Ertesitesek"
    >
      {toasts.map((toast) => (
        <div key={toast.id} className="pointer-events-auto">
          <ToastItem toast={toast} onRemove={onRemove} />
        </div>
      ))}
    </div>
  )
}

// =============================================================================
// Provider Component
// =============================================================================

interface ToastProviderProps {
  children: ReactNode
  maxToasts?: number
}

export function ToastProvider({ children, maxToasts = 5 }: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const generateId = useCallback(() => {
    return `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }, [])

  const addToast = useCallback(
    (toast: Omit<Toast, 'id'>): string => {
      const id = generateId()
      const newToast: Toast = {
        ...toast,
        id,
        duration: toast.duration ?? DEFAULT_DURATION,
        dismissible: toast.dismissible ?? true,
      }

      setToasts((prev) => {
        // Limit maximum toasts
        const updated = [...prev, newToast]
        if (updated.length > maxToasts) {
          return updated.slice(-maxToasts)
        }
        return updated
      })

      return id
    },
    [generateId, maxToasts]
  )

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id))
  }, [])

  const clearToasts = useCallback(() => {
    setToasts([])
  }, [])

  // Convenience methods
  const success = useCallback(
    (message: string, title?: string) => {
      return addToast({ type: 'success', message, title })
    },
    [addToast]
  )

  const error = useCallback(
    (message: string, title?: string) => {
      return addToast({ type: 'error', message, title, duration: 8000 }) // Longer duration for errors
    },
    [addToast]
  )

  const warning = useCallback(
    (message: string, title?: string) => {
      return addToast({ type: 'warning', message, title })
    },
    [addToast]
  )

  const info = useCallback(
    (message: string, title?: string) => {
      return addToast({ type: 'info', message, title })
    },
    [addToast]
  )

  const value: ToastContextValue = {
    toasts,
    addToast,
    removeToast,
    clearToasts,
    success,
    error,
    warning,
    info,
  }

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </ToastContext.Provider>
  )
}

// =============================================================================
// Hook
// =============================================================================

/**
 * Hook to access toast notifications
 *
 * Usage:
 *   const toast = useToast()
 *   toast.success('Sikeres mentes!')
 *   toast.error('Hiba tortent!', 'Hiba')
 */
export function useToast(): ToastContextValue {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}

export default ToastContext
