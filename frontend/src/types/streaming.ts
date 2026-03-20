/**
 * Streaming Diagnosis Types
 *
 * TypeScript types for Server-Sent Events (SSE) streaming diagnosis.
 * Mirrors backend schemas: StreamingEventType, StreamingEvent, DiagnosisStreamRequest.
 */

// =============================================================================
// Event Types
// =============================================================================

export type StreamingEventType =
  | 'start'
  | 'context'
  | 'analysis'
  | 'cause'
  | 'repair'
  | 'warning'
  | 'complete'
  | 'error';

/**
 * A single SSE event from the streaming diagnosis endpoint.
 * Matches the backend StreamingEvent schema and _format_sse_event output.
 */
export interface StreamingEvent {
  event_type: StreamingEventType;
  data: Record<string, unknown>;
  diagnosis_id: string;
  timestamp: string;
  progress: number; // 0.0 - 1.0 (backend uses float, not percentage)
}

// =============================================================================
// Callback Interfaces
// =============================================================================

/**
 * Callbacks for handling streaming diagnosis events.
 * Each callback corresponds to a StreamingEventType.
 */
export interface StreamingCallbacks {
  onStart?: (data: Record<string, unknown>) => void;
  onContext?: (data: Record<string, unknown>) => void;
  onAnalysis?: (data: Record<string, unknown>) => void;
  onCause?: (data: Record<string, unknown>) => void;
  onRepair?: (data: Record<string, unknown>) => void;
  onWarning?: (data: Record<string, unknown>) => void;
  onComplete?: (data: Record<string, unknown>) => void;
  onError?: (error: Error) => void;
  onProgress?: (progress: number, stepName?: string) => void;
}

// =============================================================================
// Request Types
// =============================================================================

/**
 * Request body for the streaming diagnosis endpoint.
 * Mirrors backend DiagnosisStreamRequest schema.
 */
export interface DiagnosisStreamRequest {
  vehicle_make: string;
  vehicle_model: string;
  vehicle_year: number;
  vehicle_engine?: string;
  vin?: string;
  dtc_codes: string[];
  symptoms: string;
  additional_context?: string;
  include_context?: boolean;
  include_progress?: boolean;
}
