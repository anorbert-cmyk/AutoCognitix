import { lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import LoadingSpinner from './components/ui/LoadingSpinner'
import ErrorBoundary from './components/ErrorBoundary'
import { AuthProvider, ProtectedRoute } from './contexts/AuthContext'
import { ToastProvider } from './contexts/ToastContext'

// Lazy load pages for code splitting
// Each page will be loaded only when navigated to
const HomePage = lazy(() => import('./pages/HomePage'))
const DiagnosisPage = lazy(() => import('./pages/DiagnosisPage'))
const ResultPage = lazy(() => import('./pages/ResultPage'))
const HistoryPage = lazy(() => import('./pages/HistoryPage'))
const DTCDetailPage = lazy(() => import('./pages/DTCDetailPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

// Auth pages
const LoginPage = lazy(() => import('./pages/LoginPage'))
const RegisterPage = lazy(() => import('./pages/RegisterPage'))
const ForgotPasswordPage = lazy(() => import('./pages/ForgotPasswordPage'))
const ResetPasswordPage = lazy(() => import('./pages/ResetPasswordPage'))

// Loading fallback component
function PageLoading() {
  return (
    <div className="flex items-center justify-center min-h-[50vh]">
      <LoadingSpinner size="lg" />
    </div>
  )
}

// Global error handler for logging
function handleGlobalError(error: Error, errorInfo: React.ErrorInfo) {
  // Log to console in development
  if (import.meta.env.DEV) {
    console.error('Global error caught:', error)
    console.error('Component stack:', errorInfo.componentStack)
  }

  // In production, you would send this to an error tracking service
  // e.g., Sentry, LogRocket, etc.
}

function App() {
  return (
    <ErrorBoundary onError={handleGlobalError}>
      <ToastProvider>
        <AuthProvider>
          <Suspense fallback={<PageLoading />}>
            <Routes>
              {/* Auth pages - no layout */}
              <Route path="/login" element={<LoginPage />} />
              <Route path="/register" element={<RegisterPage />} />
              <Route path="/forgot-password" element={<ForgotPasswordPage />} />
              <Route path="/reset-password" element={<ResetPasswordPage />} />

              {/* Main app with layout */}
              <Route path="/" element={<Layout />}>
                <Route index element={<HomePage />} />
                <Route path="diagnosis" element={<DiagnosisPage />} />
                <Route path="diagnosis/:id" element={<ResultPage />} />
                <Route
                  path="history"
                  element={
                    <ProtectedRoute>
                      <HistoryPage />
                    </ProtectedRoute>
                  }
                />
                <Route path="dtc/:code" element={<DTCDetailPage />} />
                <Route path="*" element={<NotFoundPage />} />
              </Route>
            </Routes>
          </Suspense>
        </AuthProvider>
      </ToastProvider>
    </ErrorBoundary>
  )
}

export default App
