import { useCallback, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { ApiError } from '../services/api'

/**
 * Error handling state
 */
interface ErrorState {
  error: ApiError | Error | null
  hasError: boolean
}

/**
 * Options for useErrorHandler hook
 */
interface UseErrorHandlerOptions {
  /** Automatically show error notifications */
  showNotification?: boolean
  /** Log errors to console in development */
  logErrors?: boolean
  /** Custom error handler callback */
  onError?: (error: ApiError | Error) => void
  /** Retry callback for recoverable errors */
  onRetry?: () => void
}

/**
 * Hook for handling API errors with consistent behavior.
 *
 * Provides:
 * - Error state management
 * - Error clearing
 * - Retry functionality
 * - Notification integration (optional)
 *
 * Usage:
 *   const { error, hasError, handleError, clearError, retry } = useErrorHandler({
 *     onRetry: refetch,
 *   })
 *
 *   try {
 *     await apiCall()
 *   } catch (e) {
 *     handleError(e)
 *   }
 */
export function useErrorHandler(options: UseErrorHandlerOptions = {}) {
  const {
    showNotification = false,
    logErrors = import.meta.env.DEV,
    onError,
    onRetry,
  } = options

  const [errorState, setErrorState] = useState<ErrorState>({
    error: null,
    hasError: false,
  })

  /**
   * Handle an error
   */
  const handleError = useCallback(
    (error: unknown) => {
      // Convert to ApiError if possible
      const apiError =
        error instanceof ApiError
          ? error
          : error instanceof Error
            ? error
            : new Error(String(error))

      // Update state
      setErrorState({
        error: apiError,
        hasError: true,
      })

      // Log in development
      if (logErrors) {
        console.error('Error handled:', apiError)
      }

      // Call custom handler
      if (onError) {
        onError(apiError)
      }

      // Show notification if enabled
      if (showNotification) {
        // Integration with notification system would go here
        // For now, just log
        console.warn('Error notification:', apiError instanceof ApiError ? apiError.detail : apiError.message)
      }
    },
    [logErrors, onError, showNotification]
  )

  /**
   * Clear the current error
   */
  const clearError = useCallback(() => {
    setErrorState({
      error: null,
      hasError: false,
    })
  }, [])

  /**
   * Retry the failed operation
   */
  const retry = useCallback(() => {
    clearError()
    if (onRetry) {
      onRetry()
    }
  }, [clearError, onRetry])

  return {
    error: errorState.error,
    hasError: errorState.hasError,
    handleError,
    clearError,
    retry,
  }
}

/**
 * Options for useApiMutation hook
 */
interface UseApiMutationOptions<TData, TVariables> {
  /** Mutation function */
  mutationFn: (variables: TVariables) => Promise<TData>
  /** Success callback */
  onSuccess?: (data: TData) => void
  /** Error callback */
  onError?: (error: ApiError | Error) => void
  /** Query keys to invalidate on success */
  invalidateKeys?: string[][]
}

/**
 * Simplified mutation hook with built-in error handling.
 *
 * Provides:
 * - Loading state
 * - Error state with Hungarian messages
 * - Success state
 * - Query invalidation
 *
 * Usage:
 *   const { mutate, isLoading, error, isSuccess } = useApiMutation({
 *     mutationFn: (data) => api.post('/diagnosis', data),
 *     onSuccess: (result) => navigate(`/diagnosis/${result.id}`),
 *     invalidateKeys: [['diagnoses']],
 *   })
 */
export function useApiMutation<TData, TVariables>({
  mutationFn,
  onSuccess,
  onError,
  invalidateKeys,
}: UseApiMutationOptions<TData, TVariables>) {
  const queryClient = useQueryClient()
  const [isLoading, setIsLoading] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [data, setData] = useState<TData | null>(null)

  const reset = useCallback(() => {
    setIsLoading(false)
    setIsSuccess(false)
    setError(null)
    setData(null)
  }, [])

  const mutate = useCallback(
    async (variables: TVariables) => {
      setIsLoading(true)
      setError(null)
      setIsSuccess(false)

      try {
        const result = await mutationFn(variables)
        setData(result)
        setIsSuccess(true)

        // Invalidate related queries
        if (invalidateKeys) {
          for (const key of invalidateKeys) {
            queryClient.invalidateQueries({ queryKey: key })
          }
        }

        if (onSuccess) {
          onSuccess(result)
        }

        return result
      } catch (e) {
        const apiError =
          e instanceof ApiError
            ? e
            : e instanceof Error
              ? e
              : new Error(String(e))

        setError(apiError)

        if (onError) {
          onError(apiError)
        }

        throw apiError
      } finally {
        setIsLoading(false)
      }
    },
    [mutationFn, onSuccess, onError, invalidateKeys, queryClient]
  )

  return {
    mutate,
    mutateAsync: mutate,
    isLoading,
    isSuccess,
    isError: error !== null,
    error,
    data,
    reset,
  }
}

/**
 * Check if an error is a network error
 */
export function isNetworkError(error: unknown): boolean {
  if (error instanceof ApiError) {
    return error.isNetworkError
  }
  if (error instanceof Error) {
    const message = error.message.toLowerCase()
    return message.includes('network') || message.includes('fetch')
  }
  return false
}

/**
 * Check if an error is recoverable (worth retrying)
 */
export function isRecoverableError(error: unknown): boolean {
  if (error instanceof ApiError) {
    // Network errors and server errors are usually recoverable
    return error.isNetworkError || error.status >= 500 || error.status === 429
  }
  return isNetworkError(error)
}

/**
 * Get a user-friendly error message in Hungarian
 */
export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return error.detail
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Ismeretlen hiba tortent'
}

/**
 * Get error code if available
 */
export function getErrorCode(error: unknown): string | undefined {
  if (error instanceof ApiError) {
    return error.code
  }
  return undefined
}
