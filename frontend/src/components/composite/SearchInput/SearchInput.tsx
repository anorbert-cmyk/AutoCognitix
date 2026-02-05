import { useState, useEffect, useCallback, useRef } from 'react';
import { Search, X } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SearchInputProps {
  /** Current value */
  value: string;
  /** Callback when value changes */
  onChange: (value: string) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Debounce delay in milliseconds */
  debounceMs?: number;
  /** Callback for search action (e.g., on Enter or button click) */
  onSearch?: (value: string) => void;
  /** Show clear button when there's a value */
  showClearButton?: boolean;
  /** Disabled state */
  disabled?: boolean;
  /** Additional className */
  className?: string;
}

/**
 * Search input with debounce and clear functionality.
 *
 * @example
 * ```tsx
 * <SearchInput
 *   value={searchQuery}
 *   onChange={setSearchQuery}
 *   placeholder="Keresés rendszám, hibakód..."
 *   debounceMs={300}
 *   onSearch={(query) => fetchResults(query)}
 * />
 * ```
 */
export function SearchInput({
  value,
  onChange,
  placeholder = 'Keresés...',
  debounceMs = 300,
  onSearch,
  showClearButton = true,
  disabled = false,
  className,
}: SearchInputProps) {
  const [localValue, setLocalValue] = useState(value);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync local value with external value
  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  // Debounced onChange
  const debouncedOnChange = useCallback(
    (newValue: string) => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }

      debounceTimerRef.current = setTimeout(() => {
        onChange(newValue);
      }, debounceMs);
    },
    [onChange, debounceMs]
  );

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setLocalValue(newValue);
    debouncedOnChange(newValue);
  };

  const handleClear = () => {
    setLocalValue('');
    onChange('');
    onSearch?.('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && onSearch) {
      // Cancel debounce and trigger immediately
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      onChange(localValue);
      onSearch(localValue);
    }
  };

  const hasValue = localValue.length > 0;

  return (
    <div className={cn('relative', className)}>
      {/* Search Icon */}
      <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
        <Search className="h-4 w-4 text-muted-foreground" />
      </div>

      {/* Input */}
      <input
        type="search"
        value={localValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        className={cn(
          'w-full h-10 pl-10 pr-10 text-sm',
          'bg-background border border-input rounded-lg',
          'placeholder:text-muted-foreground',
          'transition-colors duration-fast',
          'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
          'disabled:cursor-not-allowed disabled:opacity-50',
          // Hide native clear button
          '[&::-webkit-search-cancel-button]:hidden'
        )}
      />

      {/* Clear Button */}
      {showClearButton && hasValue && !disabled && (
        <button
          type="button"
          onClick={handleClear}
          className={cn(
            'absolute inset-y-0 right-0 flex items-center pr-3',
            'text-muted-foreground hover:text-foreground',
            'transition-colors duration-fast'
          )}
          aria-label="Keresés törlése"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}

export default SearchInput;
