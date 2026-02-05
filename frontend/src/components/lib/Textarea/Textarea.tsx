import {
  forwardRef,
  type TextareaHTMLAttributes,
  type ReactNode,
  useId,
} from 'react';
import { cn } from '@/lib/utils';

export interface TextareaProps
  extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Label text displayed above the textarea */
  label?: string;
  /** Error message displayed below the textarea */
  error?: string;
  /** Hint text displayed below the textarea */
  hint?: string;
  /** Show character count */
  showCharacterCount?: boolean;
  /** Icon or action button to display on the right side (e.g., microphone for dictation) */
  rightAction?: ReactNode;
  /** Container className */
  containerClassName?: string;
}

/**
 * Textarea component with label, character count, and optional action button.
 *
 * @example
 * ```tsx
 * <Textarea
 *   label="Tulajdonos panaszai"
 *   placeholder="Írja le a tapasztalt problémákat..."
 *   rightAction={
 *     <button type="button" onClick={startDictation}>
 *       <Mic className="h-5 w-5" />
 *     </button>
 *   }
 *   showCharacterCount
 *   maxLength={500}
 * />
 * ```
 */
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      label,
      error,
      hint,
      showCharacterCount,
      rightAction,
      containerClassName,
      className,
      id,
      disabled,
      maxLength,
      value,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const textareaId = id || generatedId;
    const errorId = `${textareaId}-error`;
    const hintId = `${textareaId}-hint`;

    const hasError = Boolean(error);
    const currentLength = typeof value === 'string' ? value.length : 0;

    return (
      <div className={cn('flex flex-col gap-1.5', containerClassName)}>
        {/* Label and Action */}
        <div className="flex items-center justify-between">
          {label && (
            <label
              htmlFor={textareaId}
              className="text-sm font-medium text-foreground"
            >
              {label}
            </label>
          )}
          {rightAction && (
            <div className="text-muted-foreground hover:text-foreground transition-colors">
              {rightAction}
            </div>
          )}
        </div>

        {/* Textarea */}
        <textarea
          ref={ref}
          id={textareaId}
          value={value}
          maxLength={maxLength}
          disabled={disabled}
          className={cn(
            // Base styles
            'flex w-full min-h-[120px] px-3 py-2 text-sm',
            'bg-background border border-input rounded-lg',
            'placeholder:text-muted-foreground',
            'transition-colors duration-fast',
            'resize-y',
            // Focus styles
            'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
            // Disabled styles
            'disabled:cursor-not-allowed disabled:opacity-50',
            // Error styles
            hasError &&
              'border-destructive focus:ring-destructive text-destructive',
            className
          )}
          aria-invalid={hasError}
          aria-describedby={
            hasError ? errorId : hint ? hintId : undefined
          }
          {...props}
        />

        {/* Footer: Error/Hint and Character Count */}
        <div className="flex items-center justify-between">
          <div>
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

          {/* Character Count */}
          {showCharacterCount && maxLength && (
            <span
              className={cn(
                'text-xs',
                currentLength >= maxLength
                  ? 'text-destructive'
                  : 'text-muted-foreground'
              )}
            >
              {currentLength}/{maxLength}
            </span>
          )}
        </div>
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';

export default Textarea;
