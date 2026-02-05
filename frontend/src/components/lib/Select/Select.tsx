import {
  forwardRef,
  type SelectHTMLAttributes,
  useId,
} from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps
  extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'onChange'> {
  /** Array of options to display */
  options: SelectOption[];
  /** Currently selected value */
  value?: string;
  /** Callback when selection changes */
  onChange?: (value: string) => void;
  /** Placeholder text when no option is selected */
  placeholder?: string;
  /** Label text displayed above the select */
  label?: string;
  /** Error message displayed below the select */
  error?: string;
  /** Hint text displayed below the select */
  hint?: string;
  /** Container className */
  containerClassName?: string;
}

/**
 * Native select component with consistent styling.
 *
 * @example
 * ```tsx
 * <Select
 *   label="Gyártó"
 *   placeholder="Válassz gyártót"
 *   options={[
 *     { value: 'vw', label: 'Volkswagen' },
 *     { value: 'bmw', label: 'BMW' },
 *   ]}
 *   value={selectedMake}
 *   onChange={setSelectedMake}
 * />
 * ```
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      options,
      value,
      onChange,
      placeholder,
      label,
      error,
      hint,
      containerClassName,
      className,
      id,
      disabled,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const selectId = id || generatedId;
    const errorId = `${selectId}-error`;
    const hintId = `${selectId}-hint`;

    const hasError = Boolean(error);

    const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
      onChange?.(e.target.value);
    };

    return (
      <div className={cn('flex flex-col gap-1.5', containerClassName)}>
        {label && (
          <label
            htmlFor={selectId}
            className="text-sm font-medium text-foreground"
          >
            {label}
          </label>
        )}

        <div className="relative">
          <select
            ref={ref}
            id={selectId}
            value={value}
            onChange={handleChange}
            disabled={disabled}
            className={cn(
              // Base styles
              'flex w-full h-10 px-3 py-2 pr-10 text-sm',
              'bg-background border border-input rounded-lg',
              'appearance-none cursor-pointer',
              'transition-colors duration-fast',
              // Placeholder style (when no value selected)
              !value && 'text-muted-foreground',
              // Focus styles
              'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
              // Disabled styles
              'disabled:cursor-not-allowed disabled:opacity-50',
              // Error styles
              hasError &&
                'border-destructive focus:ring-destructive',
              className
            )}
            aria-invalid={hasError}
            aria-describedby={
              hasError ? errorId : hint ? hintId : undefined
            }
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {options.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </option>
            ))}
          </select>

          {/* Dropdown Icon */}
          <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none text-muted-foreground">
            <ChevronDown className="h-4 w-4" />
          </div>
        </div>

        {/* Error Message */}
        {hasError && (
          <p id={errorId} className="text-sm text-destructive" role="alert">
            {error}
          </p>
        )}

        {/* Hint Text */}
        {!hasError && hint && (
          <p id={hintId} className="text-sm text-muted-foreground">
            {hint}
          </p>
        )}
      </div>
    );
  }
);

Select.displayName = 'Select';

export default Select;
