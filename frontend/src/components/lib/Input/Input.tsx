import {
  forwardRef,
  type InputHTMLAttributes,
  type ReactNode,
  useId,
} from 'react';
import { cn } from '@/lib/utils';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  /** Label text displayed above the input */
  label?: string;
  /** Error message displayed below the input */
  error?: string;
  /** Hint text displayed below the input (hidden when error is shown) */
  hint?: string;
  /** Icon or element to display on the left side inside the input */
  leftIcon?: ReactNode;
  /** Icon or element to display on the right side inside the input */
  rightIcon?: ReactNode;
  /** Text or element to display as an addon on the left (outside input border) */
  leftAddon?: ReactNode;
  /** Text or element to display as an addon on the right (outside input border) */
  rightAddon?: ReactNode;
  /** Container className for the entire input group */
  containerClassName?: string;
}

/**
 * Input component with label, icons, addons, and error states.
 *
 * @example
 * ```tsx
 * <Input
 *   label="Hibakód megadása"
 *   placeholder="Pl. P0171, P0300"
 *   leftIcon={<Search />}
 * />
 *
 * <Input
 *   label="Email"
 *   type="email"
 *   error="Érvénytelen email cím"
 * />
 * ```
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      error,
      hint,
      leftIcon,
      rightIcon,
      leftAddon,
      rightAddon,
      containerClassName,
      className,
      id,
      disabled,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const inputId = id || generatedId;
    const errorId = `${inputId}-error`;
    const hintId = `${inputId}-hint`;

    const hasError = Boolean(error);

    return (
      <div className={cn('flex flex-col gap-1.5', containerClassName)}>
        {label && (
          <label
            htmlFor={inputId}
            className="text-sm font-medium text-foreground"
          >
            {label}
          </label>
        )}

        <div className="relative flex">
          {/* Left Addon */}
          {leftAddon && (
            <div className="flex items-center px-3 border border-r-0 rounded-l-lg border-input bg-muted text-muted-foreground text-sm">
              {leftAddon}
            </div>
          )}

          {/* Input Container */}
          <div className="relative flex-1">
            {/* Left Icon */}
            {leftIcon && (
              <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-muted-foreground">
                <span className="h-4 w-4">{leftIcon}</span>
              </div>
            )}

            {/* Input Element */}
            <input
              ref={ref}
              id={inputId}
              className={cn(
                // Base styles
                'flex w-full h-10 px-3 py-2 text-sm',
                'bg-background border border-input rounded-lg',
                'placeholder:text-muted-foreground',
                'transition-colors duration-fast',
                // Focus styles
                'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
                // Disabled styles
                'disabled:cursor-not-allowed disabled:opacity-50',
                // Error styles
                hasError &&
                  'border-destructive focus:ring-destructive text-destructive',
                // Icon padding
                leftIcon && 'pl-10',
                rightIcon && 'pr-10',
                // Addon border radius
                leftAddon && 'rounded-l-none',
                rightAddon && 'rounded-r-none',
                className
              )}
              disabled={disabled}
              aria-invalid={hasError}
              aria-describedby={
                hasError ? errorId : hint ? hintId : undefined
              }
              {...props}
            />

            {/* Right Icon */}
            {rightIcon && (
              <div className="absolute inset-y-0 right-0 flex items-center pr-3 text-muted-foreground">
                <span className="h-4 w-4">{rightIcon}</span>
              </div>
            )}
          </div>

          {/* Right Addon */}
          {rightAddon && (
            <div className="flex items-center px-3 border border-l-0 rounded-r-lg border-input bg-muted text-muted-foreground text-sm">
              {rightAddon}
            </div>
          )}
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

Input.displayName = 'Input';

export default Input;
