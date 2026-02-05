/**
 * DiagnosisPage - Main diagnosis form page
 *
 * Features:
 * - VehicleSelector component for vehicle selection
 * - DTC code autocomplete input
 * - Symptom description textarea
 * - Form validation
 * - API integration for diagnosis analysis
 */

import { useState, useCallback } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { AlertCircle, FileText, Loader2 } from 'lucide-react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import { useAnalyzeDiagnosis } from '../services/hooks'
import { ErrorMessage, DTCAutocomplete } from '../components/ui'
import VehicleSelector from '../components/VehicleSelector'
import { ApiError } from '../services/api'
import { isValidDTCFormat } from '../services/dtcService'
import { useToast } from '../contexts/ToastContext'
import type { SelectedVehicle } from '../types/vehicle'

// Form validation schema
const diagnosisSchema = z.object({
  symptoms: z
    .string()
    .min(10, 'A tunetleiras legalabb 10 karakter kell legyen')
    .max(2000, 'Maximum 2000 karakter'),
})

type DiagnosisFormData = z.infer<typeof diagnosisSchema>

export default function DiagnosisPage() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const toast = useToast()

  // Get initial DTC code from URL if provided (e.g., from DTCDetailPage link)
  const initialDtcCode = searchParams.get('dtc')

  // Vehicle state (controlled by VehicleSelector)
  const [selectedVehicle, setSelectedVehicle] = useState<SelectedVehicle | null>(null)
  const [vehicleError, setVehicleError] = useState<string | null>(null)

  // DTC codes state - initialize with URL param if available
  const [dtcCodes, setDtcCodes] = useState<string[]>(
    initialDtcCode && isValidDTCFormat(initialDtcCode) ? [initialDtcCode.toUpperCase()] : []
  )
  const [dtcError, setDtcError] = useState<string | null>(null)

  // Form error state
  const [submitError, setSubmitError] = useState<ApiError | null>(null)

  // React Hook Form
  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<DiagnosisFormData>({
    resolver: zodResolver(diagnosisSchema),
    defaultValues: {
      symptoms: '',
    },
  })

  // Diagnosis mutation
  const analyzeDiagnosis = useAnalyzeDiagnosis()

  // Handle vehicle selection change
  const handleVehicleChange = useCallback((vehicle: SelectedVehicle | null) => {
    setSelectedVehicle(vehicle)
    setVehicleError(null)
  }, [])

  // Validate vehicle selection
  const validateVehicle = useCallback((): boolean => {
    if (!selectedVehicle) {
      setVehicleError('Kerem valasszon jarmut')
      return false
    }

    if (!selectedVehicle.make || selectedVehicle.make.trim() === '') {
      setVehicleError('Gyarto megadasa kotelezo')
      return false
    }

    if (!selectedVehicle.model || selectedVehicle.model.trim() === '') {
      setVehicleError('Modell megadasa kotelezo')
      return false
    }

    if (!selectedVehicle.year || selectedVehicle.year < 1900 || selectedVehicle.year > 2030) {
      setVehicleError('Ervenyes evjarat megadasa kotelezo')
      return false
    }

    setVehicleError(null)
    return true
  }, [selectedVehicle])

  // Validate DTC codes
  const validateDTCCodes = useCallback((): boolean => {
    if (dtcCodes.length === 0) {
      setDtcError('Legalabb egy DTC kod megadasa kotelezo')
      return false
    }

    const invalidCodes = dtcCodes.filter((code) => !isValidDTCFormat(code))
    if (invalidCodes.length > 0) {
      setDtcError(`Ervenytelen DTC kodok: ${invalidCodes.join(', ')}`)
      return false
    }

    setDtcError(null)
    return true
  }, [dtcCodes])

  // Form submission
  const onSubmit = async (data: DiagnosisFormData) => {
    // Validate all fields
    const isVehicleValid = validateVehicle()
    const isDTCValid = validateDTCCodes()

    if (!isVehicleValid || !isDTCValid) {
      return
    }

    setSubmitError(null)

    try {
      const result = await analyzeDiagnosis.mutateAsync({
        vehicleMake: selectedVehicle!.make,
        vehicleModel: selectedVehicle!.model,
        vehicleYear: selectedVehicle!.year,
        vin: selectedVehicle!.vin || undefined,
        dtcCodes,
        symptoms: data.symptoms,
      })
      toast.success('Diagnozis sikeresen elkeszult!', 'Sikeres elemzes')
      navigate(`/diagnosis/${result.id}`)
    } catch (err) {
      if (err instanceof ApiError) {
        setSubmitError(err)
        toast.error(err.detail, 'Diagnozis hiba')
      } else {
        setSubmitError(new ApiError('Ismeretlen hiba tortent', 500))
        toast.error('Ismeretlen hiba tortent', 'Hiba')
      }
    }
  }

  const symptomsValue = watch('symptoms')

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="mx-auto max-w-3xl px-4 sm:px-6 lg:px-8">
        {/* Page Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Uj diagnosztika
          </h1>
          <p className="text-gray-600">
            Toltse ki az alabbi mezoket a pontos diagnozishoz.
          </p>
        </div>

        {/* Submit Error */}
        {submitError && (
          <ErrorMessage
            error={submitError}
            onDismiss={() => setSubmitError(null)}
            onRetry={
              analyzeDiagnosis.isError ? () => handleSubmit(onSubmit)() : undefined
            }
            className="mb-6"
          />
        )}

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-8">
          {/* Vehicle Selector Card */}
          <div className="card">
            <div className="card-content">
              <VehicleSelector
                value={selectedVehicle || undefined}
                onChange={handleVehicleChange}
                required
                showVINInput
                error={vehicleError || undefined}
              />
            </div>
          </div>

          {/* DTC Codes Card */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-5 w-5 text-primary-600" />
                <h2 className="card-title text-lg">Hibakodok</h2>
              </div>
            </div>
            <div className="card-content">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                DTC kodok <span className="text-red-500">*</span>
              </label>
              <DTCAutocomplete
                value={dtcCodes}
                onChange={(codes) => {
                  setDtcCodes(codes)
                  setDtcError(null)
                }}
                maxCodes={20}
              />
              {dtcError && (
                <p className="mt-2 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle className="h-4 w-4" />
                  {dtcError}
                </p>
              )}
            </div>
          </div>

          {/* Symptoms Card */}
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary-600" />
                <h2 className="card-title text-lg">Tunetek leirasa</h2>
              </div>
            </div>
            <div className="card-content">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Tapasztalt tunetek <span className="text-red-500">*</span>
              </label>
              <textarea
                {...register('symptoms')}
                rows={5}
                placeholder="Irja le reszletesen a tapasztalt problemakat. Peldaul: A motor nehezen indul hidegben, egyenetlenul jar alapjaraton, megnovekedet a fogyasztas..."
                className="textarea"
              />
              <div className="mt-1 flex justify-between">
                <p className="text-sm text-gray-500">
                  Minel reszletesebb leirast ad, annal pontosabb diagnozist kaphat.
                </p>
                <span className="text-sm text-gray-400">
                  {symptomsValue?.length || 0}/2000
                </span>
              </div>
              {errors.symptoms && (
                <p className="mt-1 text-sm text-red-600 flex items-center gap-1">
                  <AlertCircle className="h-4 w-4" />
                  {errors.symptoms.message}
                </p>
              )}
            </div>
          </div>

          {/* Submit Buttons */}
          <div className="flex flex-col sm:flex-row justify-end gap-4">
            <button
              type="button"
              onClick={() => navigate('/')}
              className="btn-outline order-2 sm:order-1"
            >
              Megse
            </button>
            <button
              type="submit"
              disabled={analyzeDiagnosis.isPending}
              className="btn-primary order-1 sm:order-2"
            >
              {analyzeDiagnosis.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Elemzes folyamatban...
                </>
              ) : (
                'Diagnozis inditasa'
              )}
            </button>
          </div>
        </form>

        {/* Help Section */}
        <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h3 className="font-medium text-blue-900 mb-2">Tippek a pontos diagnozishoz</h3>
          <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
            <li>Adjon meg minden lekerdezett hibakodot a diagnosztikai muszerbol</li>
            <li>Irja le reszletesen, mikor es hogyan jelentkezik a problema</li>
            <li>Emlitsen minden eszrevett valtozast (hang, szag, vibracio stb.)</li>
            <li>Ha lehetseges, adja meg a VIN kodot a pontosabb eredmenyekert</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
