/**
 * Forgot password page for AutoCognitix.
 */

import { useState, FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { Car, Loader2, ArrowLeft, CheckCircle } from 'lucide-react'
import { forgotPassword } from '../services/authService'
import { ApiError } from '../services/api'

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isSubmitted, setIsSubmitted] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)

    if (!email.trim()) {
      setError('Kerem adja meg az email cimet')
      return
    }

    setIsLoading(true)

    try {
      await forgotPassword({ email })
      setIsSubmitted(true)
    } catch (err) {
      const apiError = err as ApiError
      setError(apiError.detail || 'Hiba tortent a kerelem feldolgozasa soran')
    } finally {
      setIsLoading(false)
    }
  }

  if (isSubmitted) {
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
              Email elkuldve
            </h2>
            <p className="text-gray-600 mb-6">
              Ha a megadott email cim ({email}) letezik a rendszerben, elkuldtunk
              egy linket a jelszo visszaallitasahoz.
            </p>
            <p className="text-sm text-gray-500 mb-6">
              Kerem ellenorizze a leveleit, beleertve a spam mappat is.
            </p>
            <Link
              to="/login"
              className="inline-flex items-center text-primary-600 hover:text-primary-500 font-medium"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Vissza a bejelentkezeshez
            </Link>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <Link to="/" className="flex items-center justify-center gap-2">
          <Car className="h-10 w-10 text-primary-600" />
          <span className="text-2xl font-bold text-gray-900">AutoCognitix</span>
        </Link>
        <h2 className="mt-6 text-center text-3xl font-bold text-gray-900">
          Elfelejtett jelszo
        </h2>
        <p className="mt-2 text-center text-sm text-gray-600">
          Adja meg az email cimet es elkuldunk egy linket a jelszo
          visszaallitasahoz.
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
                    Kuldes...
                  </>
                ) : (
                  'Jelszo visszaallitas kerese'
                )}
              </button>
            </div>

            {/* Back to login link */}
            <div className="text-center">
              <Link
                to="/login"
                className="inline-flex items-center text-sm text-primary-600 hover:text-primary-500"
              >
                <ArrowLeft className="h-4 w-4 mr-1" />
                Vissza a bejelentkezeshez
              </Link>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
