/**
 * VehicleSelector Component
 *
 * Three cascading dropdowns for Make -> Model -> Year selection
 * with VIN decode support and searchable dropdowns.
 *
 * Features:
 * - Searchable dropdowns using react-select
 * - Cascading selection (model depends on make, years depend on model)
 * - VIN input with decode functionality
 * - Loading states and error handling
 * - Keyboard navigation
 * - Mobile-friendly responsive design
 * - Hungarian labels
 */

import { useState, useEffect, useCallback, useMemo } from 'react'
import Select, { SingleValue, StylesConfig } from 'react-select'
import { Car, QrCode, Loader2, X, RefreshCw, AlertCircle } from 'lucide-react'
import { clsx } from 'clsx'
import {
  useVehicleMakes,
  useVehicleModels,
  useVINDecode,
  generateYearOptions,
  usePrefetchModels,
} from '../hooks/useVehicles'
import { validateVIN } from '../services/vehicleService'
import type {
  VehicleSelectorProps,
  VehicleOption,
  YearOption,
} from '../types/vehicle'
import type { VehicleMake, VehicleModel } from '../services/api'

// =============================================================================
// Custom Styles for react-select (typed generically)
// =============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const baseSelectStyles: StylesConfig<any, false> = {
  control: (base, state) => ({
    ...base,
    minHeight: '42px',
    borderColor: state.isFocused ? '#2563eb' : '#d1d5db',
    boxShadow: state.isFocused ? '0 0 0 2px rgba(37, 99, 235, 0.2)' : 'none',
    '&:hover': {
      borderColor: state.isFocused ? '#2563eb' : '#9ca3af',
    },
    borderRadius: '0.375rem',
    backgroundColor: state.isDisabled ? '#f3f4f6' : '#ffffff',
  }),
  placeholder: (base) => ({
    ...base,
    color: '#9ca3af',
  }),
  option: (base, state) => ({
    ...base,
    backgroundColor: state.isSelected
      ? '#2563eb'
      : state.isFocused
      ? '#eff6ff'
      : 'transparent',
    color: state.isSelected ? '#ffffff' : '#1f2937',
    cursor: 'pointer',
    '&:active': {
      backgroundColor: '#dbeafe',
    },
  }),
  menu: (base) => ({
    ...base,
    zIndex: 50,
    borderRadius: '0.375rem',
    boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  }),
  menuList: (base) => ({
    ...base,
    maxHeight: '200px',
  }),
  singleValue: (base) => ({
    ...base,
    color: '#1f2937',
  }),
  input: (base) => ({
    ...base,
    color: '#1f2937',
  }),
  noOptionsMessage: (base) => ({
    ...base,
    color: '#6b7280',
  }),
  loadingMessage: (base) => ({
    ...base,
    color: '#6b7280',
  }),
}

// =============================================================================
// VehicleSelector Component
// =============================================================================

export default function VehicleSelector({
  value,
  onChange,
  disabled = false,
  required = false,
  className,
  showVINInput = true,
  error,
}: VehicleSelectorProps) {
  // Local state for VIN input
  const [vinInput, setVinInput] = useState('')
  const [vinError, setVinError] = useState<string | null>(null)
  const [isVINDecoded, setIsVINDecoded] = useState(false)

  // Local state for selections
  const [selectedMake, setSelectedMake] = useState<VehicleOption | null>(null)
  const [selectedModel, setSelectedModel] = useState<VehicleOption | null>(null)
  const [selectedYear, setSelectedYear] = useState<YearOption | null>(null)

  // Prefetch helper
  const prefetchModels = usePrefetchModels()

  // Fetch makes
  const {
    data: makes,
    isLoading: makesLoading,
    error: makesError,
    refetch: refetchMakes,
  } = useVehicleMakes({ enabled: true })

  // Fetch models when make is selected
  const {
    data: models,
    isLoading: modelsLoading,
    error: modelsError,
    refetch: refetchModels,
  } = useVehicleModels({
    makeId: selectedMake?.value,
    enabled: !!selectedMake?.value,
  })

  // VIN decode mutation
  const vinDecode = useVINDecode({
    onSuccess: (data) => {
      // Auto-fill the form with decoded data
      const makeOption = makeOptions.find(
        (m) => m.label.toLowerCase() === data.make.toLowerCase()
      )

      if (makeOption) {
        setSelectedMake(makeOption)
        // The model will be selected after models load
      }

      // Set year
      if (data.year) {
        setSelectedYear({ value: data.year, label: data.year.toString() })
      }

      setIsVINDecoded(true)
      setVinError(null)

      // Notify parent with partial data
      onChange({
        make: data.make,
        model: data.model,
        year: data.year,
        vin: data.vin,
        engine: data.engine,
      })
    },
    onError: (err) => {
      setVinError(err.message || 'VIN dekodolasi hiba')
      setIsVINDecoded(false)
    },
  })

  // Convert makes to options
  const makeOptions: VehicleOption[] = useMemo(() => {
    if (!makes) return []
    return makes.map((make: VehicleMake) => ({
      value: make.id,
      label: make.name,
      data: make,
    }))
  }, [makes])

  // Convert models to options
  const modelOptions: VehicleOption[] = useMemo(() => {
    if (!models) return []
    return models.map((model: VehicleModel) => ({
      value: model.id,
      label: model.name,
      data: model,
    }))
  }, [models])

  // Generate year options based on selected model
  const yearOptions: YearOption[] = useMemo(() => {
    const selectedModelData = models?.find((m: VehicleModel) => m.id === selectedModel?.value)
    const years = generateYearOptions(selectedModelData)
    return years.map((year) => ({
      value: year,
      label: year.toString(),
    }))
  }, [selectedModel, models])

  // Sync external value changes
  useEffect(() => {
    if (value) {
      // Find and set make option
      const makeOpt = makeOptions.find(
        (m) => m.label.toLowerCase() === value.make.toLowerCase() || m.value === value.makeId
      )
      if (makeOpt && selectedMake?.value !== makeOpt.value) {
        setSelectedMake(makeOpt)
      }

      // Year
      if (value.year && selectedYear?.value !== value.year) {
        setSelectedYear({ value: value.year, label: value.year.toString() })
      }

      // VIN
      if (value.vin && vinInput !== value.vin) {
        setVinInput(value.vin)
      }
    }
  }, [value, makeOptions]) // eslint-disable-line react-hooks/exhaustive-deps

  // Set model after models are loaded (for VIN decode)
  useEffect(() => {
    if (value?.model && models && modelOptions.length > 0) {
      const modelOpt = modelOptions.find(
        (m) => m.label.toLowerCase() === value.model.toLowerCase() || m.value === value.modelId
      )
      if (modelOpt && selectedModel?.value !== modelOpt.value) {
        setSelectedModel(modelOpt)
      }
    }
  }, [value?.model, value?.modelId, models, modelOptions]) // eslint-disable-line react-hooks/exhaustive-deps

  // Handle make selection
  const handleMakeChange = useCallback(
    (option: SingleValue<VehicleOption>) => {
      setSelectedMake(option)
      setSelectedModel(null) // Reset model when make changes
      setIsVINDecoded(false)

      if (option) {
        // Prefetch models for this make
        prefetchModels(option.value)

        // Notify parent with partial data
        onChange({
          make: option.label,
          makeId: option.value,
          model: '',
          year: selectedYear?.value || new Date().getFullYear(),
        })
      } else {
        onChange(null)
      }
    },
    [onChange, selectedYear, prefetchModels]
  )

  // Handle model selection
  const handleModelChange = useCallback(
    (option: SingleValue<VehicleOption>) => {
      setSelectedModel(option)

      if (option && selectedMake) {
        const modelData = option.data as VehicleModel | undefined

        // Check if current year is valid for this model
        let year = selectedYear?.value
        if (modelData && year) {
          const validYears = generateYearOptions(modelData)
          if (!validYears.includes(year)) {
            // Reset to most recent valid year
            year = validYears[0]
            setSelectedYear({ value: year, label: year.toString() })
          }
        }

        onChange({
          make: selectedMake.label,
          makeId: selectedMake.value,
          model: option.label,
          modelId: option.value,
          year: year || new Date().getFullYear(),
          vin: vinInput || undefined,
        })
      }
    },
    [onChange, selectedMake, selectedYear, vinInput]
  )

  // Handle year selection
  const handleYearChange = useCallback(
    (option: SingleValue<YearOption>) => {
      setSelectedYear(option)

      if (option && selectedMake) {
        onChange({
          make: selectedMake.label,
          makeId: selectedMake.value,
          model: selectedModel?.label || '',
          modelId: selectedModel?.value,
          year: option.value,
          vin: vinInput || undefined,
        })
      }
    },
    [onChange, selectedMake, selectedModel, vinInput]
  )

  // Handle VIN input
  const handleVINChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const vin = e.target.value.toUpperCase().replace(/[^A-HJ-NPR-Z0-9]/g, '')
    setVinInput(vin)
    setVinError(null)
    setIsVINDecoded(false)
  }

  // Handle VIN decode
  const handleVINDecode = () => {
    const validationError = validateVIN(vinInput)
    if (validationError) {
      setVinError(validationError)
      return
    }
    vinDecode.mutate(vinInput)
  }

  // Handle clear/reset
  const handleClear = () => {
    setSelectedMake(null)
    setSelectedModel(null)
    setSelectedYear(null)
    setVinInput('')
    setVinError(null)
    setIsVINDecoded(false)
    onChange(null)
  }

  const hasSelection = selectedMake || selectedModel || selectedYear || vinInput

  return (
    <div className={clsx('space-y-6', className)}>
      {/* VIN Decoder Section */}
      {showVINInput && (
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
          <div className="flex items-center gap-2 mb-3">
            <QrCode className="h-5 w-5 text-primary-600" />
            <h3 className="font-medium text-gray-900">VIN dekodolas (opcionalis)</h3>
          </div>
          <p className="text-sm text-gray-600 mb-3">
            Adja meg a jarmu VIN kodjat az adatok automatikus kitoltesehez.
          </p>
          <div className="flex flex-col sm:flex-row gap-2">
            <div className="flex-1 relative">
              <input
                type="text"
                value={vinInput}
                onChange={handleVINChange}
                placeholder="pl. WVWZZZ3CZWE123456"
                maxLength={17}
                disabled={disabled}
                className={clsx(
                  'w-full px-3 py-2 border rounded-md font-mono text-sm',
                  'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
                  vinError ? 'border-red-500' : 'border-gray-300',
                  disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'
                )}
                aria-label="VIN kod"
                aria-invalid={!!vinError}
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-400">
                {vinInput.length}/17
              </span>
            </div>
            <button
              type="button"
              onClick={handleVINDecode}
              disabled={disabled || vinInput.length !== 17 || vinDecode.isPending}
              className={clsx(
                'px-4 py-2 rounded-md font-medium text-sm transition-colors',
                'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'bg-primary-600 text-white hover:bg-primary-700'
              )}
            >
              {vinDecode.isPending ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span className="hidden sm:inline">Dekodolas...</span>
                </span>
              ) : (
                'Dekodolas'
              )}
            </button>
          </div>
          {vinError && (
            <p className="mt-2 text-sm text-red-600 flex items-center gap-1">
              <AlertCircle className="h-4 w-4" />
              {vinError}
            </p>
          )}
          {isVINDecoded && !vinError && (
            <p className="mt-2 text-sm text-green-600 flex items-center gap-1">
              <Car className="h-4 w-4" />
              Jarmu adatok sikeresen betoltve!
            </p>
          )}
        </div>
      )}

      {/* Vehicle Selection Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Car className="h-5 w-5 text-primary-600" />
            <h3 className="font-medium text-gray-900">Jarmu adatok {required && '*'}</h3>
          </div>
          {hasSelection && !disabled && (
            <button
              type="button"
              onClick={handleClear}
              className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
            >
              <X className="h-4 w-4" />
              Torles
            </button>
          )}
        </div>

        {/* Error from API */}
        {(makesError || modelsError) && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-700 flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              Hiba tortent az adatok betoltesekor.
              <button
                type="button"
                onClick={() => {
                  if (makesError) refetchMakes()
                  if (modelsError) refetchModels()
                }}
                className="ml-2 text-red-800 underline hover:no-underline flex items-center gap-1"
              >
                <RefreshCw className="h-3 w-3" />
                Ujraproba
              </button>
            </p>
          </div>
        )}

        {/* Cascading Dropdowns */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Make Select */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Gyarto {required && <span className="text-red-500">*</span>}
            </label>
            <Select<VehicleOption>
              value={selectedMake}
              onChange={handleMakeChange}
              options={makeOptions}
              placeholder="Valasszon gyartot..."
              isClearable
              isSearchable
              isLoading={makesLoading}
              isDisabled={disabled}
              styles={baseSelectStyles as StylesConfig<VehicleOption, false>}
              noOptionsMessage={() => 'Nincs talalat'}
              loadingMessage={() => 'Betoltes...'}
              aria-label="Gyarto valasztas"
              classNamePrefix="vehicle-select"
            />
          </div>

          {/* Model Select */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Modell {required && <span className="text-red-500">*</span>}
            </label>
            <Select<VehicleOption>
              value={selectedModel}
              onChange={handleModelChange}
              options={modelOptions}
              placeholder={selectedMake ? 'Valasszon modellt...' : 'Eloszor valasszon gyartot'}
              isClearable
              isSearchable
              isLoading={modelsLoading}
              isDisabled={disabled || !selectedMake}
              styles={baseSelectStyles as StylesConfig<VehicleOption, false>}
              noOptionsMessage={() =>
                selectedMake ? 'Nincs modell adat ehhez a gyartohoz' : 'Eloszor valasszon gyartot'
              }
              loadingMessage={() => 'Modellek betoltese...'}
              aria-label="Modell valasztas"
              classNamePrefix="vehicle-select"
            />
            {selectedMake && !modelsLoading && modelOptions.length === 0 && (
              <p className="mt-1 text-xs text-gray-500">
                Ha nem talal modellt, irja be kezzel alabb.
              </p>
            )}
          </div>

          {/* Year Select */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Evjarat {required && <span className="text-red-500">*</span>}
            </label>
            <Select<YearOption>
              value={selectedYear}
              onChange={handleYearChange}
              options={yearOptions}
              placeholder="Valasszon evjaratot..."
              isClearable
              isSearchable
              isDisabled={disabled}
              styles={baseSelectStyles as StylesConfig<YearOption, false>}
              noOptionsMessage={() => 'Nincs talalat'}
              aria-label="Evjarat valasztas"
              classNamePrefix="vehicle-select"
            />
          </div>
        </div>

        {/* Manual Model Input (fallback when no models from API) */}
        {selectedMake && !modelsLoading && modelOptions.length === 0 && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Modell (kezzel) {required && <span className="text-red-500">*</span>}
            </label>
            <input
              type="text"
              value={selectedModel?.label || ''}
              onChange={(e) => {
                const inputValue = e.target.value
                if (inputValue) {
                  const manualOption: VehicleOption = {
                    value: inputValue.toLowerCase().replace(/\s+/g, '-'),
                    label: inputValue,
                  }
                  setSelectedModel(manualOption)
                  if (selectedMake) {
                    onChange({
                      make: selectedMake.label,
                      makeId: selectedMake.value,
                      model: inputValue,
                      year: selectedYear?.value || new Date().getFullYear(),
                      vin: vinInput || undefined,
                    })
                  }
                } else {
                  setSelectedModel(null)
                }
              }}
              placeholder="pl. Golf, Focus, Corolla"
              disabled={disabled}
              className={clsx(
                'w-full px-3 py-2 border rounded-md text-sm',
                'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
                'border-gray-300',
                disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'
              )}
            />
          </div>
        )}

        {/* Validation Error */}
        {error && (
          <p className="text-sm text-red-600 flex items-center gap-1">
            <AlertCircle className="h-4 w-4" />
            {error}
          </p>
        )}
      </div>

      {/* Summary of selected vehicle */}
      {(selectedMake || selectedModel || selectedYear) && (
        <div className="bg-blue-50 rounded-lg p-3 border border-blue-200">
          <p className="text-sm text-blue-800">
            <span className="font-medium">Kivalasztott jarmu: </span>
            {[selectedYear?.label, selectedMake?.label, selectedModel?.label]
              .filter(Boolean)
              .join(' ')}
            {vinInput && ` (VIN: ${vinInput.slice(0, 4)}...${vinInput.slice(-4)})`}
          </p>
        </div>
      )}
    </div>
  )
}
