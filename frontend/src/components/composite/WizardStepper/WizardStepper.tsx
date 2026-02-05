import { Check } from 'lucide-react';
import { cn } from '@/lib/utils';

export type StepStatus = 'pending' | 'active' | 'completed';

export interface WizardStep {
  /** Step title */
  title: string;
  /** Optional description */
  description?: string;
  /** Step status */
  status: StepStatus;
}

export interface WizardStepperProps {
  /** Array of steps */
  steps: WizardStep[];
  /** Current active step (0-indexed) */
  currentStep: number;
  /** Callback when step is clicked (for navigation) */
  onStepClick?: (stepIndex: number) => void;
  /** Orientation */
  orientation?: 'horizontal' | 'vertical';
  /** Additional className */
  className?: string;
}

/**
 * Wizard stepper component for multi-step forms.
 *
 * @example
 * ```tsx
 * <WizardStepper
 *   steps={[
 *     { title: 'Adatbevitel', status: 'completed' },
 *     { title: 'Elemzés', status: 'active' },
 *     { title: 'Jelentés', status: 'pending' },
 *   ]}
 *   currentStep={1}
 * />
 * ```
 */
export function WizardStepper({
  steps,
  currentStep,
  onStepClick,
  orientation = 'horizontal',
  className,
}: WizardStepperProps) {
  const isHorizontal = orientation === 'horizontal';

  return (
    <nav
      className={cn(
        'flex',
        isHorizontal ? 'flex-row items-center justify-center' : 'flex-col',
        className
      )}
      aria-label="Lépések"
    >
      {steps.map((step, index) => {
        const isActive = step.status === 'active' || index === currentStep;
        const isCompleted = step.status === 'completed' || index < currentStep;
        const isClickable = onStepClick && (isCompleted || isActive);
        const isLast = index === steps.length - 1;

        return (
          <div
            key={index}
            className={cn(
              'flex',
              isHorizontal ? 'flex-row items-center' : 'flex-col'
            )}
          >
            {/* Step Item */}
            <div
              className={cn(
                'flex items-center gap-3',
                isClickable && 'cursor-pointer',
                isHorizontal && 'flex-col sm:flex-row'
              )}
              onClick={() => isClickable && onStepClick?.(index)}
              role={isClickable ? 'button' : undefined}
              tabIndex={isClickable ? 0 : undefined}
              onKeyDown={(e) => {
                if (isClickable && (e.key === 'Enter' || e.key === ' ')) {
                  e.preventDefault();
                  onStepClick?.(index);
                }
              }}
            >
              {/* Step Circle */}
              <div
                className={cn(
                  'flex items-center justify-center w-10 h-10 rounded-full',
                  'text-sm font-medium transition-colors',
                  isCompleted
                    ? 'bg-primary-600 text-white'
                    : isActive
                    ? 'bg-primary-100 text-primary-700 border-2 border-primary-600'
                    : 'bg-muted text-muted-foreground'
                )}
              >
                {isCompleted ? (
                  <Check className="h-5 w-5" />
                ) : (
                  <span>{index + 1}</span>
                )}
              </div>

              {/* Step Text */}
              <div
                className={cn(
                  isHorizontal && 'text-center sm:text-left'
                )}
              >
                <p
                  className={cn(
                    'text-sm font-medium',
                    isActive || isCompleted
                      ? 'text-foreground'
                      : 'text-muted-foreground'
                  )}
                >
                  {step.title}
                </p>
                {step.description && (
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {step.description}
                  </p>
                )}
              </div>
            </div>

            {/* Connector Line */}
            {!isLast && (
              <div
                className={cn(
                  'transition-colors',
                  isHorizontal
                    ? 'hidden sm:block w-12 md:w-20 lg:w-32 h-0.5 mx-4'
                    : 'w-0.5 h-8 ml-5 my-2',
                  index < currentStep ? 'bg-primary-600' : 'bg-border'
                )}
              />
            )}
          </div>
        );
      })}
    </nav>
  );
}

export default WizardStepper;
