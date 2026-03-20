/**
 * AnalysisProgress - Exact match to provided HTML design
 * Shows the AI analysis in progress with step indicators.
 *
 * Supports two modes:
 * 1. **Streaming mode** (default): Consumes real SSE events from the backend
 *    via `streamDiagnosis()` and maps them to progress steps.
 * 2. **Mock mode** (`streamingEnabled=false`): Simulates progress locally
 *    for demos / fallback when streaming is unavailable.
 *
 * Primary: #137fec, Background: #f6f7f8
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Check,
  Loader2,
  Clock,
  X,
  Car,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import type {
  StreamingEventType,
  StreamingCallbacks,
  DiagnosisStreamRequest,
} from '@/types/streaming';
import type { DiagnosisRequest, DiagnosisResponse } from '@/services/api';

// ---------------------------------------------------------------------------
// streamDiagnosis – the actual function is being created by another agent in
// diagnosisService.ts.  We import the entire module and look up the function
// at call-time so the component works whether or not it exists yet.
// ---------------------------------------------------------------------------
import * as diagnosisServiceModule from '@/services/diagnosisService';

type StreamDiagnosisFn = (
  params: DiagnosisStreamRequest,
  callbacks: StreamingCallbacks,
) => { abort: () => void };

function getStreamDiagnosisFn(): StreamDiagnosisFn | undefined {
  const fn = (diagnosisServiceModule as Record<string, unknown>)['streamDiagnosis'];
  return typeof fn === 'function' ? (fn as StreamDiagnosisFn) : undefined;
}

// =============================================================================
// Types
// =============================================================================

export type AnalysisStepStatus = 'pending' | 'in_progress' | 'completed' | 'error';

export interface AnalysisStep {
  id: string;
  title: string;
  status: AnalysisStepStatus;
  description?: string;
}

export interface AnalysisProgressProps {
  /** Array of analysis steps (only used in mock mode) */
  steps?: AnalysisStep[];
  /** Currently active step index (for external control) */
  currentStep?: number;
  /** Diagnosis ID to navigate to when complete */
  diagnosisId?: string;
  /** Cancel handler */
  onCancel?: () => void;
  /** Callback when analysis completes (receives DiagnosisResponse in streaming mode) */
  onComplete?: (result?: DiagnosisResponse) => void;
  /** Callback when error occurs */
  onError?: (error: string) => void;
  /** Vehicle info for display */
  vehicleInfo?: {
    make: string;
    model: string;
    year: number;
    dtcCode: string;
  };
  /** Enable SSE streaming (default: true) */
  streamingEnabled?: boolean;
  /** Diagnosis request in API format – required when streamingEnabled is true */
  diagnosisRequest?: DiagnosisRequest;
  /** Additional className */
  className?: string;
}

// =============================================================================
// Constants
// =============================================================================

/** Default steps shown in mock mode */
const defaultSteps: AnalysisStep[] = [
  { id: 'obd', title: 'OBD kod ertelmezese', status: 'completed' },
  { id: 'model', title: 'Specifikus modell hibak keresese', status: 'completed' },
  { id: 'causes', title: 'Hiba lehetseges okainak elemzese', status: 'in_progress' },
  { id: 'docs', title: 'Muszaki dokumentacio attekintese', status: 'pending' },
  { id: 'plan', title: 'Javasolt javitasi terv osszeallitasa', status: 'pending' },
];

/** Streaming step definitions keyed by SSE event type */
const STREAMING_STEPS: { eventType: StreamingEventType; id: string; title: string }[] = [
  { eventType: 'start', id: 'start', title: 'Adatok feldolgozasa' },
  { eventType: 'context', id: 'context', title: 'Kontextus lekerdezes' },
  { eventType: 'analysis', id: 'analysis', title: 'AI elemzes' },
  { eventType: 'cause', id: 'cause', title: 'Okok azonositasa' },
  { eventType: 'repair', id: 'repair', title: 'Javitasi terv' },
  { eventType: 'complete', id: 'complete', title: 'Befejezes' },
];

/** Map event type to step index for quick lookup */
const EVENT_TO_STEP_INDEX: Record<string, number> = {};
STREAMING_STEPS.forEach((s, i) => {
  EVENT_TO_STEP_INDEX[s.eventType] = i;
});

/** Recent analyses cards for the bottom section */
const recentAnalyses = [
  {
    id: '1',
    vehicleMake: 'Volkswagen',
    vehicleModel: 'Golf',
    dtcCode: 'P0420',
    description: 'Katalizator rendszer hatasfoka kuszobertek alatt',
  },
  {
    id: '2',
    vehicleMake: 'Ford',
    vehicleModel: 'Focus',
    dtcCode: 'P0171',
    description: 'Rendszer tul szegeny (1. bank)',
  },
];

// =============================================================================
// Helpers
// =============================================================================

function buildStreamingSteps(activeIndex: number, errorIndex?: number): AnalysisStep[] {
  return STREAMING_STEPS.map((s, i) => {
    let status: AnalysisStepStatus = 'pending';
    if (errorIndex !== undefined && i === errorIndex) {
      status = 'error';
    } else if (i < activeIndex) {
      status = 'completed';
    } else if (i === activeIndex) {
      status = 'in_progress';
    }
    return { id: s.id, title: s.title, status };
  });
}

// =============================================================================
// Component
// =============================================================================

/**
 * Full-page Analysis Progress component showing step completion.
 * Matches the provided HTML design with #137fec primary color.
 *
 * @example
 * ```tsx
 * // Streaming mode (default)
 * <AnalysisProgress
 *   diagnosisRequest={formData}
 *   onCancel={() => navigate('/diagnosis')}
 *   onComplete={(result) => navigate(`/diagnosis/${result?.id}`)}
 *   vehicleInfo={{ make: 'Toyota', model: 'Camry', year: 2019, dtcCode: 'P0300' }}
 * />
 *
 * // Mock / fallback mode
 * <AnalysisProgress
 *   streamingEnabled={false}
 *   diagnosisId="123"
 *   onCancel={() => navigate('/diagnosis')}
 * />
 * ```
 */
export function AnalysisProgress({
  steps: initialSteps,
  diagnosisId,
  onCancel,
  onComplete,
  onError,
  vehicleInfo,
  streamingEnabled = true,
  diagnosisRequest,
}: AnalysisProgressProps) {
  const navigate = useNavigate();

  // ---- Shared UI state ----
  const [steps, setSteps] = useState<AnalysisStep[]>(
    initialSteps || (streamingEnabled ? buildStreamingSteps(-1) : defaultSteps)
  );
  const [estimatedTime, setEstimatedTime] = useState(45);
  const [streamingProgress, setStreamingProgress] = useState(0);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isRetrying, setIsRetrying] = useState(false);

  // Abort controller ref for cleanup
  const abortRef = useRef<{ abort: () => void } | null>(null);
  const mountedRef = useRef(true);

  // Track whether streaming has been started to avoid double-invocation
  const streamStartedRef = useRef(false);

  // ---- Derived state ----
  const completedSteps = steps.filter((s) => s.status === 'completed').length;
  const progress = streamingEnabled
    ? Math.round(streamingProgress * 100)
    : Math.round((completedSteps / steps.length) * 100);

  // =========================================================================
  // Streaming mode logic
  // =========================================================================

  const startStreaming = useCallback(() => {
    const streamFn = getStreamDiagnosisFn();
    if (!diagnosisRequest || !streamFn) return;

    // Reset state for (re)start
    setErrorMessage(null);
    setWarnings([]);
    setStreamingProgress(0);
    setSteps(buildStreamingSteps(0));

    const handle = streamFn(
      {
        ...diagnosisRequest,
        include_context: true,
        include_progress: true,
      },
      {
        onStart: () => {
          if (!mountedRef.current) return;
          setSteps(buildStreamingSteps(0));
          setStreamingProgress(0.05);
        },
        onContext: () => {
          if (!mountedRef.current) return;
          setSteps(buildStreamingSteps(1));
          setStreamingProgress(0.2);
        },
        onAnalysis: () => {
          if (!mountedRef.current) return;
          setSteps(buildStreamingSteps(2));
          setStreamingProgress(0.45);
        },
        onCause: () => {
          if (!mountedRef.current) return;
          setSteps(buildStreamingSteps(3));
          setStreamingProgress(0.68);
        },
        onRepair: () => {
          if (!mountedRef.current) return;
          setSteps(buildStreamingSteps(4));
          setStreamingProgress(0.82);
        },
        onWarning: (data) => {
          if (!mountedRef.current) return;
          const msg = (data?.message as string) || 'Ismeretlen figyelmeztetes';
          setWarnings((prev) => [...prev, msg]);
        },
        onComplete: (data) => {
          if (!mountedRef.current) return;
          setSteps(buildStreamingSteps(STREAMING_STEPS.length));
          setStreamingProgress(1);
          // Give a brief moment to show 100% before navigating
          setTimeout(() => {
            if (!mountedRef.current) return;
            if (onComplete) {
              onComplete(data as unknown as DiagnosisResponse);
            } else if (diagnosisId) {
              navigate(`/diagnosis/${diagnosisId}`);
            }
          }, 600);
        },
        onError: (err) => {
          if (!mountedRef.current) return;
          const msg = err?.message || 'Ismeretlen hiba tortent az elemzes soran';
          setErrorMessage(msg);
          // Mark current in-progress step as error
          setSteps((prev) =>
            prev.map((s) => (s.status === 'in_progress' ? { ...s, status: 'error' as const } : s))
          );
          if (onError) {
            onError(msg);
          }
        },
        onProgress: (prog, stepName) => {
          if (!mountedRef.current) return;
          setStreamingProgress(prog);
          // Update estimated time based on progress
          if (prog > 0) {
            setEstimatedTime(Math.max(0, Math.round((1 - prog) * 45)));
          }
          // If stepName provided, update the active step index
          if (stepName) {
            const idx = EVENT_TO_STEP_INDEX[stepName];
            if (idx !== undefined) {
              setSteps(buildStreamingSteps(idx));
            }
          }
        },
      }
    );

    abortRef.current = handle;
  }, [diagnosisRequest, diagnosisId, navigate, onComplete, onError]);

  // Start streaming on mount (once)
  useEffect(() => {
    if (!streamingEnabled) return;
    if (streamStartedRef.current) return;

    // If streamDiagnosis is not available, fall back to mock
    if (!getStreamDiagnosisFn()) {
      console.warn(
        'AnalysisProgress: streamDiagnosis not available, falling back to mock mode'
      );
      return;
    }

    if (!diagnosisRequest) {
      console.warn(
        'AnalysisProgress: streamingEnabled but no diagnosisRequest provided'
      );
      return;
    }

    streamStartedRef.current = true;
    startStreaming();

    // Cleanup on unmount
    return () => {
      mountedRef.current = false;
      if (abortRef.current) {
        abortRef.current.abort();
        abortRef.current = null;
      }
    };
  }, [streamingEnabled, diagnosisRequest, startStreaming]);

  // =========================================================================
  // Mock mode logic (fallback)
  // =========================================================================

  const isMockMode = !streamingEnabled || !getStreamDiagnosisFn() || !diagnosisRequest;

  useEffect(() => {
    if (!isMockMode) return;

    const interval = setInterval(() => {
      setSteps((currentSteps) => {
        const inProgressIndex = currentSteps.findIndex(
          (s) => s.status === 'in_progress'
        );

        if (inProgressIndex === -1) return currentSteps;

        // 30% chance to complete current step each tick
        if (Math.random() > 0.7) {
          const newSteps = [...currentSteps];
          newSteps[inProgressIndex] = { ...newSteps[inProgressIndex], status: 'completed' };

          // Start next step if exists
          if (inProgressIndex < newSteps.length - 1) {
            newSteps[inProgressIndex + 1] = {
              ...newSteps[inProgressIndex + 1],
              status: 'in_progress',
            };
          }

          // If all done, trigger complete
          const newCompletedCount = newSteps.filter(
            (s) => s.status === 'completed'
          ).length;

          if (newCompletedCount === newSteps.length) {
            if (onComplete) {
              setTimeout(() => onComplete(), 500);
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
  }, [isMockMode, diagnosisId, navigate, onComplete]);

  // =========================================================================
  // Handlers
  // =========================================================================

  const handleCancel = () => {
    // Abort streaming if active
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    if (onCancel) {
      onCancel();
    } else {
      navigate('/diagnosis');
    }
  };

  const handleRetry = () => {
    if (!diagnosisRequest || !getStreamDiagnosisFn()) return;
    setIsRetrying(true);
    streamStartedRef.current = false;

    // Abort previous if any
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }

    mountedRef.current = true;
    streamStartedRef.current = true;
    setIsRetrying(false);
    startStreaming();
  };

  // =========================================================================
  // Render
  // =========================================================================

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
                  Elemzes
                </span>
              </div>
              <div className="w-12 h-0.5 bg-gray-300" />

              {/* Step 3 - Pending */}
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gray-200 text-gray-500 flex items-center justify-center font-semibold">
                  3
                </div>
                <span className="text-sm font-medium text-gray-400">
                  Jelentes
                </span>
              </div>
            </div>

            {/* Cancel button */}
            <button
              onClick={handleCancel}
              className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-900 transition-colors"
            >
              <X className="w-5 h-5" />
              <span className="hidden sm:inline font-medium">Megse</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-12">
        {/* Title Section */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-[#137fec]/10 mb-6">
            {errorMessage ? (
              <AlertTriangle className="w-8 h-8 text-red-500" />
            ) : (
              <Loader2 className="w-8 h-8 text-[#137fec] animate-spin" />
            )}
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-3">
            {errorMessage ? 'Hiba tortent az elemzes soran' : 'Elemzes folyamatban...'}
          </h1>
          <p className="text-lg text-gray-600 max-w-lg mx-auto">
            {errorMessage
              ? errorMessage
              : 'AI rendszerunk elemzi a megadott adatokat es szakertoi javaslatokat keszit.'}
          </p>
        </div>

        {/* Warnings */}
        {warnings.length > 0 && (
          <div className="mb-8 space-y-2">
            {warnings.map((w, i) => (
              <div
                key={i}
                className="flex items-start gap-3 p-4 rounded-lg bg-amber-50 border border-amber-200"
              >
                <AlertTriangle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                <p className="text-sm text-amber-800 font-medium">{w}</p>
              </div>
            ))}
          </div>
        )}

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
                Elemzes allapota
              </span>
              <span className="text-sm font-bold text-[#137fec]">
                {completedSteps}/{steps.length} lepes kesz
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
                  step.status === 'error'
                    ? 'bg-red-50 border border-red-200'
                    : step.status === 'in_progress'
                    ? 'bg-[#137fec]/5 border border-[#137fec]/20'
                    : step.status === 'completed'
                    ? 'bg-green-50 border border-green-100'
                    : 'bg-gray-50 border border-gray-100'
                }`}
              >
                {/* Status Icon */}
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    step.status === 'error'
                      ? 'bg-red-500 text-white'
                      : step.status === 'completed'
                      ? 'bg-green-500 text-white'
                      : step.status === 'in_progress'
                      ? 'bg-[#137fec] text-white'
                      : 'bg-gray-300 text-gray-500'
                  }`}
                >
                  {step.status === 'error' ? (
                    <X className="w-5 h-5" />
                  ) : step.status === 'completed' ? (
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
                    step.status === 'error'
                      ? 'text-red-700'
                      : step.status === 'completed'
                      ? 'text-green-700'
                      : step.status === 'in_progress'
                      ? 'text-[#137fec]'
                      : 'text-gray-500'
                  }`}
                >
                  {step.title}
                </span>

                {/* Status badge */}
                {step.status === 'error' && (
                  <span className="text-xs font-bold text-red-600 bg-red-100 px-2 py-1 rounded">
                    Hiba
                  </span>
                )}

                {step.status === 'in_progress' && (
                  <span className="text-xs font-bold text-[#137fec] bg-[#137fec]/10 px-2 py-1 rounded">
                    Folyamatban
                  </span>
                )}

                {step.status === 'completed' && (
                  <span className="text-xs font-bold text-green-600 bg-green-50 px-2 py-1 rounded">
                    Kesz
                  </span>
                )}
              </div>
            ))}
          </div>

          {/* Estimated Time */}
          {!errorMessage && (
            <div className="mt-8 pt-6 border-t border-gray-200 flex items-center justify-center gap-2 text-gray-500">
              <Clock className="w-5 h-5" />
              <span className="font-medium">
                Becsult hatralevo ido:{' '}
                <span className="text-gray-900 font-bold">~{estimatedTime} mp</span>
              </span>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="text-center flex items-center justify-center gap-4">
          {errorMessage && !isMockMode && (
            <button
              onClick={handleRetry}
              disabled={isRetrying}
              className="px-8 py-3 bg-[#137fec] text-white font-semibold rounded-lg hover:bg-[#0f6fd4] transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <RefreshCw className={`w-5 h-5 ${isRetrying ? 'animate-spin' : ''}`} />
              Ujraprobalas
            </button>
          )}
          <button
            onClick={handleCancel}
            className="px-8 py-3 border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors"
          >
            Elemzes megszakitasa
          </button>
        </div>
      </main>

      {/* Bottom Section - Recent Analyses */}
      <div className="bg-white border-t border-gray-200 py-8">
        <div className="max-w-4xl mx-auto px-4">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">
            Legutobbi elemzesek
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
