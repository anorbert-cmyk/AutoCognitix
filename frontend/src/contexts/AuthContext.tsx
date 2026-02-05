/**
 * Authentication context for AutoCognitix.
 *
 * Provides authentication state and functions throughout the app.
 * Handles automatic token refresh and user state persistence.
 */

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  ReactNode,
} from 'react'
import {
  User,
  LoginCredentials,
  RegisterData,
  UpdateProfileData,
  ChangePasswordData,
  login as loginApi,
  logout as logoutApi,
  register as registerApi,
  getCurrentUser,
  updateProfile as updateProfileApi,
  changePassword as changePasswordApi,
  isAuthenticated as checkAuth,
  clearTokens,
} from '../services/authService'
import { ApiError } from '../services/api'

// =============================================================================
// Types
// =============================================================================

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  login: (credentials: LoginCredentials) => Promise<void>
  register: (data: RegisterData) => Promise<void>
  logout: () => Promise<void>
  updateProfile: (data: UpdateProfileData) => Promise<void>
  changePassword: (data: ChangePasswordData) => Promise<void>
  clearError: () => void
  refreshUser: () => Promise<void>
}

// =============================================================================
// Context
// =============================================================================

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// =============================================================================
// Provider
// =============================================================================

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Check authentication on mount
  useEffect(() => {
    const initAuth = async () => {
      if (checkAuth()) {
        try {
          const userData = await getCurrentUser()
          setUser(userData)
        } catch (err) {
          // Token is invalid or expired
          clearTokens()
          setUser(null)
        }
      }
      setIsLoading(false)
    }

    initAuth()
  }, [])

  const login = useCallback(async (credentials: LoginCredentials) => {
    setIsLoading(true)
    setError(null)

    try {
      await loginApi(credentials)
      const userData = await getCurrentUser()
      setUser(userData)
    } catch (err) {
      const apiError = err as ApiError
      setError(apiError.detail || 'Bejelentkezési hiba')
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  const register = useCallback(async (data: RegisterData) => {
    setIsLoading(true)
    setError(null)

    try {
      await registerApi(data)
      // Auto-login after registration
      await loginApi({ email: data.email, password: data.password })
      const userData = await getCurrentUser()
      setUser(userData)
    } catch (err) {
      const apiError = err as ApiError
      setError(apiError.detail || 'Regisztrációs hiba')
      throw err
    } finally {
      setIsLoading(false)
    }
  }, [])

  const logout = useCallback(async () => {
    setIsLoading(true)

    try {
      await logoutApi()
    } catch (err) {
      console.warn('Logout error:', err)
    } finally {
      setUser(null)
      setIsLoading(false)
    }
  }, [])

  const updateProfile = useCallback(async (data: UpdateProfileData) => {
    setError(null)

    try {
      const updatedUser = await updateProfileApi(data)
      setUser(updatedUser)
    } catch (err) {
      const apiError = err as ApiError
      setError(apiError.detail || 'Profil frissitesi hiba')
      throw err
    }
  }, [])

  const changePassword = useCallback(async (data: ChangePasswordData) => {
    setError(null)

    try {
      await changePasswordApi(data)
    } catch (err) {
      const apiError = err as ApiError
      setError(apiError.detail || 'Jelszo valtoztatas hiba')
      throw err
    }
  }, [])

  const clearError = useCallback(() => {
    setError(null)
  }, [])

  const refreshUser = useCallback(async () => {
    if (checkAuth()) {
      try {
        const userData = await getCurrentUser()
        setUser(userData)
      } catch (err) {
        clearTokens()
        setUser(null)
      }
    }
  }, [])

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    error,
    login,
    register,
    logout,
    updateProfile,
    changePassword,
    clearError,
    refreshUser,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

// =============================================================================
// Hook
// =============================================================================

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext)

  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }

  return context
}

// =============================================================================
// Protected Route Component
// =============================================================================

interface ProtectedRouteProps {
  children: ReactNode
  requiredRoles?: Array<'user' | 'mechanic' | 'admin'>
  fallback?: ReactNode
}

export function ProtectedRoute({
  children,
  requiredRoles,
  fallback,
}: ProtectedRouteProps) {
  const { isAuthenticated, isLoading, user } = useAuth()

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  if (!isAuthenticated) {
    // Redirect to login
    window.location.href = '/login'
    return null
  }

  // Check role requirements
  if (requiredRoles && user && !requiredRoles.includes(user.role)) {
    if (fallback) {
      return <>{fallback}</>
    }
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-2">
            Hozzaferes megtagadva
          </h1>
          <p className="text-gray-600">
            Nincs jogosultsaga az oldal megtekitesehez.
          </p>
        </div>
      </div>
    )
  }

  return <>{children}</>
}

// =============================================================================
// Exports
// =============================================================================

export default AuthContext
