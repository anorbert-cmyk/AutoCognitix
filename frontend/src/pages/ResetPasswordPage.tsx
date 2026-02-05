/**
 * Reset password page for AutoCognitix.
 */

import { useState, FormEvent } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { Car, Eye, EyeOff, Loader2, Check, X, CheckCircle, AlertCircle } from 'lucide-react'
import { resetPassword } from '../services/authService'
import { ApiError } from '../services/api'

// Password requirements
const PASSWORD_REQUIREMENTS = [
  { id: 'length', label: 'Legalabb 8 karakter', test: (p: string) => p.length >= 8 },
  { id: 'uppercase', label: 'Legalabb egy nagybetu', test: (p: string) => /[A-Z]/.test(p) },
  { id: 'lowercase', label: 'Legalabb egy kisbetu', test: (p: string) => /[a-z]/.test(p) },
  { id: 'number', label: 'Legalabb egy szam', test: (p: string) => /\d/.test(p) },
  { id: 'special', label: 'Legalabb egy specialis karakter', test: (p: string) => /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(p) },
]

export default function ResetPasswordPage() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token')

  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isSuccess, setIsSuccess] = useState(false)
  const [showRequirements, setShowRequirements] = useState(false)

  // No token provided
  if (!token) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <Link to="/" className="flex items-center justify-center gap-2">
            <Car className="h-10 w-10 text-primary-600" />
            <span className="text-2xl font-bold text-gray-900">AutoCognitix</span>
          </Link>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 text-center">
            <AlertCircle className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Ervenytelen link
            </h2>
            <p className="text-gray-600 mb-6">
              A jelszo visszaallitasi link ervenytelen vagy hianyos. Kerem kerjen
              uj linket.
            </p>
            <Link
              to="/forgot-password"
              className="inline-flex items-center justify-center w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
            >
              Uj link kerese
            </Link>
          </div>
        </div>
      </div>
    )
  }

  // Success state
  if (isSuccess) {
    return (
      <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <Link to="/" className="flex items-center justify-center gap-2">
            <Car className="h-10 w-10 text-primary-600" />
            <span className="text-2xl font-bold text-gray-900">AutoCognitix</span>
          </Link>
        </div>

        <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
          <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 text-center">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Jelszo megvaltoztatva
            </h2>
            <p className="text-gray-600 mb-6">
              A jelszava sikeresen megvaltozott. Most mar bejelentkezhet az uj
              jelszaval.
            </p>
            <Link
              to="/login"
              className="inline-flex items-center justify-center w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
            >
              Bejelentkezes
            </Link>
          </div>
        </div>
      </div>
    )
  }

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)

    // Validation
    if (!password) {
      setError('Kerem adja meg az uj jelszot')
      return
    }

    // Check password requirements
    const failedReqs = PASSWORD_REQUIREMENTS.filter((req) => !req.test(password))
    if (failedReqs.length > 0) {
      setError('A jelszo nem felel meg a kovetelmenyeknek')
      return
    }

    if (password !== confirmPassword) {
      setError('A jelszavak nem egyeznek')
      return
    }

    setIsLoading(true)

    try {
      await resetPassword({ token, new_password: password })
      setIsSuccess(true)
    } catch (err) {
      const apiError = err as ApiError
      setError(apiError.detail || 'Hiba tortent a jelszo visszaallitasa soran')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <Link to="/" className="flex items-center justify-center gap-2">
          <Car className="h-10 w-10 text-primary-600" />
          <span className="text-2xl font-bold text-gray-900">AutoCognitix</span>
        </Link>
        <h2 className="mt-6 text-center text-3xl font-bold text-gray-900">
          Uj jelszo beallitasa
        </h2>
        <p className="mt-2 text-center text-sm text-gray-600">
          Adjon meg egy uj jelszot a fiokjahoz.
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <form className="space-y-6" onSubmit={handleSubmit}>
            {/* Error message */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md text-sm">
                {error}
              </div>
            )}

            {/* Password field */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                Uj jelszo
              </label>
              <div className="mt-1 relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onFocus={() => setShowRequirements(true)}
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm pr-10"
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeOff className="h-5 w-5 text-gray-400" />
                  ) : (
                    <Eye className="h-5 w-5 text-gray-400" />
                  )}
                </button>
              </div>

              {/* Password requirements */}
              {showRequirements && (
                <div className="mt-2 p-3 bg-gray-50 rounded-md">
                  <p className="text-xs font-medium text-gray-700 mb-2">
                    Jelszo kovetelmenyei:
                  </p>
                  <ul className="space-y-1">
                    {PASSWORD_REQUIREMENTS.map((req) => {
                      const passed = req.test(password)
                      return (
                        <li
                          key={req.id}
                          className={`flex items-center text-xs ${
                            passed ? 'text-green-600' : 'text-gray-500'
                          }`}
                        >
                          {passed ? (
                            <Check className="h-3 w-3 mr-1" />
                          ) : (
                            <X className="h-3 w-3 mr-1" />
                          )}
                          {req.label}
                        </li>
                      )
                    })}
                  </ul>
                </div>
              )}
            </div>

            {/* Confirm password field */}
            <div>
              <label
                htmlFor="confirmPassword"
                className="block text-sm font-medium text-gray-700"
              >
                Jelszo megerositese
              </label>
              <div className="mt-1 relative">
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type={showConfirmPassword ? 'text' : 'password'}
                  autoComplete="new-password"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className={`appearance-none block w-full px-3 py-2 border rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm pr-10 ${
                    confirmPassword && password !== confirmPassword
                      ? 'border-red-300'
                      : 'border-gray-300'
                  }`}
                />
                <button
                  type="button"
                  className="absolute inset-y-0 right-0 pr-3 flex items-center"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                >
                  {showConfirmPassword ? (
                    <EyeOff className="h-5 w-5 text-gray-400" />
                  ) : (
                    <Eye className="h-5 w-5 text-gray-400" />
                  )}
                </button>
              </div>
              {confirmPassword && password !== confirmPassword && (
                <p className="mt-1 text-xs text-red-600">
                  A jelszavak nem egyeznek
                </p>
              )}
            </div>

            {/* Submit button */}
            <div>
              <button
                type="submit"
                disabled={isLoading}
                className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5 mr-2" />
                    Mentes...
                  </>
                ) : (
                  'Jelszo mentese'
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
