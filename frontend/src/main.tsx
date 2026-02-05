import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import { ApiError } from './services/api'
import './index.css'

/**
 * Custom retry function for TanStack Query
 * Retries on network errors and 5xx server errors
 */
function shouldRetry(failureCount: number, error: unknown): boolean {
  // Max 3 retries
  if (failureCount >= 3) return false

  // Check if error is retryable
  if (error instanceof ApiError) {
    // Don't retry client errors (4xx) except 429 (rate limit)
    if (error.status >= 400 && error.status < 500 && error.status !== 429) {
      return false
    }
    // Retry network errors, rate limits, and server errors
    return error.isNetworkError || error.status >= 500 || error.status === 429
  }

  // Retry unknown errors (might be network issues)
  return true
}

/**
 * Global error handler for queries
 */
function onQueryError(error: unknown): void {
  // Log in development
  if (import.meta.env.DEV) {
    console.error('Query error:', error)
  }

  // In production, send to error tracking service
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: shouldRetry,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
      retryDelay: 1000,
      onError: onQueryError,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
)
