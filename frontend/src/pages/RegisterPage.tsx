/**
 * Registration page for AutoCognitix.
 */

import { useState, FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Car, Eye, EyeOff, Loader2, Check, X } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'

// Password requirements
const PASSWORD_REQUIREMENTS = [
  { id: 'length', label: 'Legalabb 8 karakter', test: (p: string) => p.length >= 8 },
  { id: 'uppercase', label: 'Legalabb egy nagybetu', test: (p: string) => /[A-Z]/.test(p) },
  { id: 'lowercase', label: 'Legalabb egy kisbetu', test: (p: string) => /[a-z]/.test(p) },
  { id: 'number', label: 'Legalabb egy szam', test: (p: string) => /\d/.test(p) },
  // eslint-disable-next-line no-useless-escape
  { id: 'special', label: 'Legalabb egy specialis karakter', test: (p: string) => /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(p) },
]

export default function RegisterPage() {
  const navigate = useNavigate()
  const { register, isLoading, error, clearError } = useAuth()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [role, setRole] = useState<'user' | 'mechanic'>('user')
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)
  const [showRequirements, setShowRequirements] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setLocalError(null)
    clearError()

    // Validation
    if (!email.trim()) {
      setLocalError('Kerem adja meg az email cimet')
      return
    }

    if (!password) {
      setLocalError('Kerem adja meg a jelszot')
      return
    }

    // Check password requirements
    const failedReqs = PASSWORD_REQUIREMENTS.filter((req) => !req.test(password))
    if (failedReqs.length > 0) {
      setLocalError('A jelszo nem felel meg a kovetelmenyeknek')
      return
    }

    if (password !== confirmPassword) {
      setLocalError('A jelszavak nem egyeznek')
      return
    }

    try {
      await register({
        email,
        password,
        full_name: fullName || undefined,
        role,
      })
      navigate('/')
    } catch {
      // Error is handled by AuthContext
    }
  }

  const displayError = localError || error

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        {/* Logo */}
        <Link to="/" className="flex items-center justify-center gap-2">
          <Car className="h-10 w-10 text-primary-600" />
          <span className="text-2xl font-bold text-gray-900">AutoCognitix</span>
        </Link>
        <h2 className="mt-6 text-center text-3xl font-bold text-gray-900">
          Regisztracio
        </h2>
        <p className="mt-2 text-center text-sm text-gray-600">
          Mar van fiokja?{' '}
          <Link
            to="/login"
            className="font-medium text-primary-600 hover:text-primary-500"
          >
            Jelentkezzen be
          </Link>
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <form className="space-y-6" onSubmit={handleSubmit}>
            {/* Error message */}
            {displayError && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md text-sm">
                {displayError}
              </div>
            )}

            {/* Full name field */}
            <div>
              <label
                htmlFor="fullName"
                className="block text-sm font-medium text-gray-700"
              >
                Teljes nev (opcionalis)
              </label>
              <div className="mt-1">
                <input
                  id="fullName"
                  name="fullName"
                  type="text"
                  autoComplete="name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                  placeholder="Kovacs Janos"
                />
              </div>
            </div>

            {/* Email field */}
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700"
              >
                Email cim
              </label>
              <div className="mt-1">
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                  placeholder="pelda@email.com"
                />
              </div>
            </div>

            {/* Role selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Fiok tipusa
              </label>
              <div className="mt-2 grid grid-cols-2 gap-3">
                <button
                  type="button"
                  onClick={() => setRole('user')}
                  className={`relative flex items-center justify-center px-4 py-3 border rounded-md text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 ${
                    role === 'user'
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Felhasznalo
                </button>
                <button
                  type="button"
                  onClick={() => setRole('mechanic')}
                  className={`relative flex items-center justify-center px-4 py-3 border rounded-md text-sm font-medium focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 ${
                    role === 'mechanic'
                      ? 'border-primary-500 bg-primary-50 text-primary-700'
                      : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  Szerelo
                </button>
              </div>
            </div>

            {/* Password field */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-gray-700"
              >
                Jelszo
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
                    Regisztracio...
                  </>
                ) : (
                  'Regisztracio'
                )}
              </button>
            </div>

            {/* Terms notice */}
            <p className="text-xs text-center text-gray-500">
              A regisztraciaval elfogadja a{' '}
              <a href="#" className="text-primary-600 hover:text-primary-500">
                felhasznalasi felteteleket
              </a>{' '}
              es az{' '}
              <a href="#" className="text-primary-600 hover:text-primary-500">
                adatvedelmi szabalyzatot
              </a>
              .
            </p>
          </form>
        </div>
      </div>
    </div>
  )
}
