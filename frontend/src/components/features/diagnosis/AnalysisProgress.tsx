/**
 * AnalysisProgress - Exact match to provided HTML design
 * Shows the AI analysis in progress with step indicators
 * Primary: #137fec, Background: #f6f7f8
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Check,
  Loader2,
  Clock,
  X,
  Car,
  AlertTriangle,
} from 'lucide-react';

export type AnalysisStepStatus = 'pending' | 'in_progress' | 'completed' | 'error';

export interface AnalysisStep {
  id: string;
  title: string;
  status: AnalysisStepStatus;
  description?: string;
}

export interface AnalysisProgressProps {
  /** Array of analysis steps */
  steps?: AnalysisStep[];
  /** Currently active step index (for external control) */
  currentStep?: number;
  /** Diagnosis ID to navigate to when complete */
  diagnosisId?: string;
  /** Cancel handler */
  onCancel?: () => void;
  /** Callback when analysis completes */
  onComplete?: () => void;
  /** Callback when error occurs */
  onError?: (error: string) => void;
  /** Vehicle info for display */
  vehicleInfo?: {
    make: string;
    model: string;
    year: number;
    dtcCode: string;
  };
  /** Additional className */
  className?: string;
}

const defaultSteps: AnalysisStep[] = [
  { id: 'obd', title: 'OBD kód értelmezése', status: 'completed' },
  { id: 'model', title: 'Specifikus modell hibák keresése', status: 'completed' },
  { id: 'causes', title: 'Hiba lehetséges okainak elemzése', status: 'in_progress' },
  { id: 'docs', title: 'Műszaki dokumentáció áttekintése', status: 'pending' },
  { id: 'plan', title: 'Javasolt javítási terv összeállítása', status: 'pending' },
];

// Legutóbbi elemzések kártyák az alsó szekcióhoz
const recentAnalyses = [
  {
    id: '1',
    vehicleMake: 'Volkswagen',
    vehicleModel: 'Golf',
    dtcCode: 'P0420',
    description: 'Katalizátor rendszer hatásfoka küszöbérték alatt',
  },
  {
    id: '2',
    vehicleMake: 'Ford',
    vehicleModel: 'Focus',
    dtcCode: 'P0171',
    description: 'Rendszer túl szegény (1. bank)',
  },
];

/**
 * Full-page Analysis Progress component showing step completion.
 * Matches the provided HTML design with #137fec primary color.
 *
 * @example
 * ```tsx
 * <AnalysisProgress
 *   diagnosisId="123"
 *   onCancel={() => navigate('/diagnosis')}
 *   vehicleInfo={{ make: 'Toyota', model: 'Camry', year: 2019, dtcCode: 'P0300' }}
 * />
 * ```
 */
export function AnalysisProgress({
  steps: initialSteps,
  diagnosisId,
  onCancel,
  onComplete,
  vehicleInfo,
}: AnalysisProgressProps) {
  const navigate = useNavigate();
  const [steps, setSteps] = useState<AnalysisStep[]>(initialSteps || defaultSteps);
  const [estimatedTime, setEstimatedTime] = useState(45);

  // Calculate progress
  const completedSteps = steps.filter((s) => s.status === 'completed').length;
  const progress = Math.round((completedSteps / steps.length) * 100);

  // Simulate progress
  useEffect(() => {
    const interval = setInterval(() => {
      setSteps((currentSteps) => {
        const inProgressIndex = currentSteps.findIndex(
          (s) => s.status === 'in_progress'
        );

        if (inProgressIndex === -1) return currentSteps;

        // 30% chance to complete current step each tick
        if (Math.random() > 0.7) {
          const newSteps = [...currentSteps];
          newSteps[inProgressIndex].status = 'completed';

          // Start next step if exists
          if (inProgressIndex < newSteps.length - 1) {
            newSteps[inProgressIndex + 1].status = 'in_progress';
          }

          // If all done, trigger complete
          const newCompletedCount = newSteps.filter(
            (s) => s.status === 'completed'
          ).length;

          if (newCompletedCount === newSteps.length) {
            if (onComplete) {
              setTimeout(onComplete, 500);
            } else if (diagnosisId) {
              setTimeout(() => navigate(`/diagnosis/${diagnosisId}`), 500);
            }
          }

          return newSteps;
        }

        return currentSteps;
      });

      // Decrease estimated time
      setEstimatedTime((t) => Math.max(0, t - 5));
    }, 1500);

    return () => clearInterval(interval);
  }, [diagnosisId, navigate, onComplete]);

  const handleCancel = () => {
    if (onCancel) {
      onCancel();
    } else {
      navigate('/diagnosis');
    }
  };

  return (
    <div className="min-h-screen bg-[#f6f7f8] flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-[#137fec] rounded-lg flex items-center justify-center">
                <Car className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-xl text-gray-900">
                MechanicAI <span className="text-[#137fec]">Pro</span>
              </span>
            </div>

            {/* Wizard Steps */}
            <div className="hidden md:flex items-center gap-4">
              {/* Step 1 - Completed */}
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-green-500 text-white flex items-center justify-center">
                  <Check className="w-5 h-5" />
                </div>
                <span className="text-sm font-medium text-gray-500">
                  Adatbevitel
                </span>
              </div>
              <div className="w-12 h-0.5 bg-green-500" />

              {/* Step 2 - Active */}
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-[#137fec] text-white flex items-center justify-center font-semibold">
                  2
                </div>
                <span className="text-sm font-semibold text-gray-900">
                  Elemzés
                </span>
              </div>
              <div className="w-12 h-0.5 bg-gray-300" />

              {/* Step 3 - Pending */}
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gray-200 text-gray-500 flex items-center justify-center font-semibold">
                  3
                </div>
                <span className="text-sm font-medium text-gray-400">
                  Jelentés
                </span>
              </div>
            </div>

            {/* Cancel button */}
            <button
              onClick={handleCancel}
              className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <X className="w-5 h-5" />
              <span className="hidden sm:inline font-medium">Mégse</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-12">
        {/* Title Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-[#137fec]/10 mb-6">
            <Loader2 className="w-8 h-8 text-[#137fec] animate-spin" />
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-3">
            Elemzés folyamatban...
          </h1>
          <p className="text-lg text-gray-600 max-w-lg mx-auto">
            AI rendszerünk elemzi a megadott adatokat és szakértői javaslatokat készít.
          </p>
        </div>

        {/* Vehicle Info Card (if provided) */}
        {vehicleInfo && (
          <div className="bg-white rounded-xl border border-gray-200 p-4 mb-8 flex items-center gap-4 shadow-sm">
            <div className="w-12 h-12 rounded-lg bg-gray-100 flex items-center justify-center">
              <Car className="w-6 h-6 text-gray-600" />
            </div>
            <div>
              <p className="font-semibold text-gray-900">
                {vehicleInfo.year} {vehicleInfo.make} {vehicleInfo.model}
              </p>
              <p className="text-sm text-gray-500 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500" />
                {vehicleInfo.dtcCode}
              </p>
            </div>
          </div>
        )}

        {/* Progress Section */}
        <div className="bg-white rounded-xl border border-gray-200 p-8 mb-8 shadow-sm">
          {/* Progress Bar */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-600">
                Elemzés állapota
              </span>
              <span className="text-sm font-bold text-[#137fec]">
                {completedSteps}/{steps.length} lépés kész
              </span>
            </div>
            <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-[#137fec] rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>

          {/* Steps List */}
          <div className="space-y-4">
            {steps.map((step) => (
              <div
                key={step.id}
                className={`flex items-center gap-4 p-4 rounded-lg transition-colors ${
                  step.status === 'in_progress'
                    ? 'bg-[#137fec]/5 border border-[#137fec]/20'
                    : step.status === 'completed'
                    ? 'bg-green-50 border border-green-100'
                    : 'bg-gray-50 border border-gray-100'
                }`}
              >
                {/* Status Icon */}
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    step.status === 'completed'
                      ? 'bg-green-500 text-white'
                      : step.status === 'in_progress'
                      ? 'bg-[#137fec] text-white'
                      : 'bg-gray-300 text-gray-500'
                  }`}
                >
                  {step.status === 'completed' ? (
                    <Check className="w-5 h-5" />
                  ) : step.status === 'in_progress' ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Clock className="w-4 h-4" />
                  )}
                </div>

                {/* Label */}
                <span
                  className={`font-medium flex-1 ${
                    step.status === 'completed'
                      ? 'text-green-700'
                      : step.status === 'in_progress'
                      ? 'text-[#137fec]'
                      : 'text-gray-500'
                  }`}
                >
                  {step.title}
                </span>

                {/* Status badge for in-progress */}
                {step.status === 'in_progress' && (
                  <span className="text-xs font-bold text-[#137fec] bg-[#137fec]/10 px-2 py-1 rounded">
                    Folyamatban
                  </span>
                )}

                {step.status === 'completed' && (
                  <span className="text-xs font-bold text-green-600 bg-green-50 px-2 py-1 rounded">
                    Kész
                  </span>
                )}
              </div>
            ))}
          </div>

          {/* Estimated Time */}
          <div className="mt-8 pt-6 border-t border-gray-200 flex items-center justify-center gap-2 text-gray-500">
            <Clock className="w-5 h-5" />
            <span className="font-medium">
              Becsült hátralévő idő:{' '}
              <span className="text-gray-900 font-bold">~{estimatedTime} mp</span>
            </span>
          </div>
        </div>

        {/* Cancel Button */}
        <div className="text-center">
          <button
            onClick={handleCancel}
            className="px-8 py-3 border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors"
          >
            Elemzés megszakítása
          </button>
        </div>
      </main>

      {/* Bottom Section - Recent Analyses */}
      <div className="bg-white border-t border-gray-200 py-8">
        <div className="max-w-4xl mx-auto px-4">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">
            Legutóbbi elemzések
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recentAnalyses.map((analysis) => (
              <div
                key={analysis.id}
                className="p-4 rounded-lg border border-gray-200 bg-gray-50 hover:border-[#137fec] transition-colors cursor-pointer"
                onClick={() => navigate(`/diagnosis/${analysis.id}`)}
              >
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 rounded-lg bg-white border border-gray-200 flex items-center justify-center flex-shrink-0">
                    <Car className="w-5 h-5 text-gray-600" />
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900">
                      {analysis.vehicleMake} {analysis.vehicleModel}
                    </p>
                    <p className="text-sm text-gray-500">
                      <span className="font-mono text-[#137fec] font-semibold">
                        {analysis.dtcCode}
                      </span>{' '}
                      - {analysis.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalysisProgress;
