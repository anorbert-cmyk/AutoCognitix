import { useEffect } from 'react';
import { Check, Loader2, Clock } from 'lucide-react';
import { cn } from '@/lib/utils';

export type AnalysisStepStatus = 'pending' | 'in_progress' | 'completed' | 'error';

export interface AnalysisStep {
  id: string;
  title: string;
  status: AnalysisStepStatus;
  description?: string;
}

export interface AnalysisProgressProps {
  /** Array of analysis steps */
  steps: AnalysisStep[];
  /** Currently active step index */
  currentStep: number;
  /** Callback when analysis completes */
  onComplete?: () => void;
  /** Callback when error occurs */
  onError?: (error: string) => void;
  /** Additional className */
  className?: string;
}

const defaultSteps: AnalysisStep[] = [
  { id: 'vehicle', title: 'Jármű adatok ellenőrzése', status: 'pending' },
  { id: 'dtc', title: 'DTC kód azonosítása', status: 'pending' },
  { id: 'symptoms', title: 'Tünet elemzés', status: 'pending' },
  { id: 'ranking', title: 'Hibalehetőségek rangsorolása', status: 'pending' },
  { id: 'solutions', title: 'Javítási javaslatok generálása', status: 'pending' },
];

const statusIcons = {
  pending: Clock,
  in_progress: Loader2,
  completed: Check,
  error: Clock,
};

const statusStyles = {
  pending: 'text-muted-foreground bg-muted',
  in_progress: 'text-primary-600 bg-primary-100',
  completed: 'text-status-success bg-status-success-light',
  error: 'text-status-error bg-status-error-light',
};

/**
 * Analysis progress component showing step completion.
 *
 * @example
 * ```tsx
 * <AnalysisProgress
 *   steps={analysisSteps}
 *   currentStep={currentStepIndex}
 *   onComplete={() => navigateToResults()}
 * />
 * ```
 */
export function AnalysisProgress({
  steps = defaultSteps,
  currentStep,
  onComplete,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onError: _onError,
  className,
}: AnalysisProgressProps) {
  // Calculate progress percentage
  const completedSteps = steps.filter((s) => s.status === 'completed').length;
  const progressPercent = Math.round((completedSteps / steps.length) * 100);

  // Check for completion
  useEffect(() => {
    if (completedSteps === steps.length && onComplete) {
      onComplete();
    }
  }, [completedSteps, steps.length, onComplete]);

  return (
    <div className={cn('space-y-6', className)}>
      {/* Progress Bar */}
      <div className="bg-card border border-border rounded-lg p-6">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-foreground">
            Elemzés folyamatban...
          </h3>
          <span className="text-sm font-medium text-primary-600">
            {progressPercent}%
          </span>
        </div>
        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary-600 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Steps List */}
      <div className="bg-card border border-border rounded-lg p-6">
        <ul className="space-y-4">
          {steps.map((step, index) => {
            const Icon = statusIcons[step.status];
            const isActive = index === currentStep;

            return (
              <li
                key={step.id}
                className={cn(
                  'flex items-center gap-4 p-4 rounded-lg transition-colors',
                  isActive && 'bg-primary-50/50',
                  step.status === 'completed' && 'opacity-70'
                )}
              >
                {/* Status Icon */}
                <div
                  className={cn(
                    'flex items-center justify-center w-10 h-10 rounded-full',
                    statusStyles[step.status]
                  )}
                >
                  <Icon
                    className={cn(
                      'h-5 w-5',
                      step.status === 'in_progress' && 'animate-spin'
                    )}
                  />
                </div>

                {/* Step Info */}
                <div className="flex-1">
                  <p
                    className={cn(
                      'font-medium',
                      step.status === 'completed'
                        ? 'text-muted-foreground'
                        : step.status === 'in_progress'
                        ? 'text-primary-700'
                        : 'text-foreground'
                    )}
                  >
                    {step.title}
                  </p>
                  {step.description && (
                    <p className="text-sm text-muted-foreground mt-0.5">
                      {step.description}
                    </p>
                  )}
                </div>

                {/* Status Text */}
                <span
                  className={cn(
                    'text-sm font-medium',
                    step.status === 'completed' && 'text-status-success',
                    step.status === 'in_progress' && 'text-primary-600',
                    step.status === 'error' && 'text-status-error',
                    step.status === 'pending' && 'text-muted-foreground'
                  )}
                >
                  {step.status === 'completed' && 'Kész'}
                  {step.status === 'in_progress' && 'Folyamatban...'}
                  {step.status === 'error' && 'Hiba'}
                  {step.status === 'pending' && 'Várakozik'}
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      {/* Estimated Time */}
      <p className="text-center text-sm text-muted-foreground">
        Becsült idő: ~{Math.max(1, steps.length - completedSteps) * 5} másodperc
      </p>
    </div>
  );
}

export default AnalysisProgress;
