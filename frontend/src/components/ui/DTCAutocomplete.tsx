import { useState, useRef, useEffect, useCallback } from 'react'
import { Search, X, AlertCircle } from 'lucide-react'
import { useDTCSearch } from '../../services/hooks'
import { isValidDTCFormat, getCategoryNameHu, getCategoryFromCode } from '../../services/dtcService'
import LoadingSpinner from './LoadingSpinner'

interface DTCAutocompleteProps {
  value: string[]
  onChange: (codes: string[]) => void
  maxCodes?: number
  placeholder?: string
  disabled?: boolean
}

/**
 * DTC code autocomplete with search functionality
 */
export default function DTCAutocomplete({
  value,
  onChange,
  maxCodes = 20,
  placeholder = 'pl. P0101, P0171, P0300',
  disabled = false,
}: DTCAutocompleteProps) {
  const [inputValue, setInputValue] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [highlightedIndex, setHighlightedIndex] = useState(-1)
  const inputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Debounced search query
  const [debouncedQuery, setDebouncedQuery] = useState('')
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(inputValue)
    }, 300)
    return () => clearTimeout(timer)
  }, [inputValue])

  const { data: searchResults, isLoading } = useDTCSearch(debouncedQuery, {
    limit: 10,
    enabled: debouncedQuery.length >= 2,
  })

  // Filter out already selected codes
  const filteredResults = searchResults?.filter(
    (result) => !value.includes(result.code)
  ) || []

  const addCode = useCallback((code: string) => {
    const normalizedCode = code.toUpperCase().trim()
    if (
      normalizedCode &&
      !value.includes(normalizedCode) &&
      value.length < maxCodes
    ) {
      if (isValidDTCFormat(normalizedCode)) {
        onChange([...value, normalizedCode])
        setInputValue('')
        setIsOpen(false)
        setHighlightedIndex(-1)
      }
    }
  }, [value, onChange, maxCodes])

  const removeCode = useCallback((codeToRemove: string) => {
    onChange(value.filter((code) => code !== codeToRemove))
  }, [value, onChange])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value.toUpperCase()
    setInputValue(newValue)
    setIsOpen(true)
    setHighlightedIndex(-1)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      if (highlightedIndex >= 0 && filteredResults[highlightedIndex]) {
        addCode(filteredResults[highlightedIndex].code)
      } else if (inputValue.length === 5 && isValidDTCFormat(inputValue)) {
        addCode(inputValue)
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightedIndex((prev) =>
        prev < filteredResults.length - 1 ? prev + 1 : prev
      )
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightedIndex((prev) => (prev > 0 ? prev - 1 : -1))
    } else if (e.key === 'Escape') {
      setIsOpen(false)
      setHighlightedIndex(-1)
    } else if (e.key === 'Backspace' && inputValue === '' && value.length > 0) {
      removeCode(value[value.length - 1])
    } else if (e.key === ',' || e.key === ' ') {
      e.preventDefault()
      if (inputValue.length === 5 && isValidDTCFormat(inputValue)) {
        addCode(inputValue)
      }
    }
  }

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div className="relative">
      <div
        className={`flex flex-wrap gap-2 p-2 rounded-md border bg-background min-h-[42px] ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-text'
        } ${isOpen ? 'ring-2 ring-ring ring-offset-2' : 'border-input'}`}
        onClick={() => !disabled && inputRef.current?.focus()}
      >
        {/* Selected codes */}
        {value.map((code) => {
          const category = getCategoryFromCode(code)
          return (
            <span
              key={code}
              className="inline-flex items-center gap-1 px-2 py-1 rounded bg-primary-100 text-primary-700 text-sm font-mono"
            >
              {code}
              {category && (
                <span className="text-primary-500 text-xs">
                  ({getCategoryNameHu(category)})
                </span>
              )}
              {!disabled && (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeCode(code)
                  }}
                  className="ml-1 hover:text-primary-900"
                  aria-label={`Eltavolitas: ${code}`}
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </span>
          )
        })}

        {/* Input field */}
        {value.length < maxCodes && (
          <div className="flex-1 min-w-[150px] relative">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsOpen(true)}
              disabled={disabled}
              placeholder={value.length === 0 ? placeholder : 'Tovabb...'}
              className="w-full bg-transparent outline-none text-sm font-mono"
              maxLength={5}
              aria-label="DTC kod kereses"
              aria-expanded={isOpen}
              aria-autocomplete="list"
            />
          </div>
        )}

        {/* Search icon */}
        {value.length === 0 && (
          <Search className="h-4 w-4 text-gray-400 absolute right-3 top-1/2 -translate-y-1/2" />
        )}
      </div>

      {/* Dropdown */}
      {isOpen && (debouncedQuery.length >= 2 || inputValue.length === 5) && (
        <div
          ref={dropdownRef}
          className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg max-h-60 overflow-auto"
        >
          {isLoading ? (
            <div className="p-4">
              <LoadingSpinner size="sm" text="Kereses..." />
            </div>
          ) : filteredResults.length > 0 ? (
            <ul role="listbox">
              {filteredResults.map((result, index) => (
                <li
                  key={result.code}
                  role="option"
                  aria-selected={highlightedIndex === index}
                  className={`px-4 py-2 cursor-pointer ${
                    highlightedIndex === index
                      ? 'bg-primary-50'
                      : 'hover:bg-gray-50'
                  }`}
                  onClick={() => addCode(result.code)}
                  onMouseEnter={() => setHighlightedIndex(index)}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-medium text-primary-700">
                      {result.code}
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded ${
                        result.severity === 'critical'
                          ? 'bg-red-100 text-red-700'
                          : result.severity === 'high'
                          ? 'bg-orange-100 text-orange-700'
                          : result.severity === 'medium'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-green-100 text-green-700'
                      }`}
                    >
                      {result.severity === 'critical'
                        ? 'Kritikus'
                        : result.severity === 'high'
                        ? 'Magas'
                        : result.severity === 'medium'
                        ? 'Kozepes'
                        : 'Alacsony'}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">
                    {result.description_hu || result.description_en}
                  </p>
                </li>
              ))}
            </ul>
          ) : inputValue.length === 5 && isValidDTCFormat(inputValue) ? (
            <div className="px-4 py-3 text-sm">
              <p className="text-gray-700">
                Nyomjon <kbd className="px-1 py-0.5 bg-gray-100 rounded">Enter</kbd>-t
                a <span className="font-mono font-medium">{inputValue}</span> hozzaadasahoz
              </p>
            </div>
          ) : debouncedQuery.length >= 2 ? (
            <div className="px-4 py-3 text-sm text-gray-500 flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              Nincs talalat
            </div>
          ) : null}
        </div>
      )}

      {/* Helper text */}
      <p className="mt-1 text-sm text-gray-500">
        {value.length}/{maxCodes} kod. Vesszoval vagy Enter-rel valassza el a kodokat.
      </p>
    </div>
  )
}
