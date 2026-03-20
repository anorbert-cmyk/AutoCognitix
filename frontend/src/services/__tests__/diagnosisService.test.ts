import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { StreamingCallbacks } from '../../types/streaming';

// =============================================================================
// Test Helpers
// =============================================================================

/**
 * Create a mock ReadableStream that emits SSE-formatted events.
 * Each event is encoded as "data: {json}\n\n" matching the backend format.
 */
function createMockSSEStream(
  events: Array<{
    event_type: string;
    data: Record<string, unknown>;
    progress: number;
    diagnosis_id?: string;
    timestamp?: string;
  }>
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const event of events) {
        const payload = {
          event_type: event.event_type,
          data: event.data,
          progress: event.progress,
          diagnosis_id: event.diagnosis_id || 'test-diag-123',
          timestamp: event.timestamp || '2026-03-20T10:00:00Z',
        };
        const line = `data: ${JSON.stringify(payload)}\n\n`;
        controller.enqueue(encoder.encode(line));
      }
      controller.close();
    },
  });
}

/**
 * Create a mock ReadableStream that emits events with a delay (for abort testing).
 */
function createSlowMockSSEStream(
  events: Array<{
    event_type: string;
    data: Record<string, unknown>;
    progress: number;
  }>,
  delayMs = 100
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    async start(controller) {
      for (const event of events) {
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        const payload = {
          event_type: event.event_type,
          data: event.data,
          progress: event.progress,
          diagnosis_id: 'test-diag-123',
          timestamp: '2026-03-20T10:00:00Z',
        };
        const line = `data: ${JSON.stringify(payload)}\n\n`;
        controller.enqueue(encoder.encode(line));
      }
      controller.close();
    },
  });
}

/** Valid diagnosis form data for tests */
const validFormData = {
  vehicleMake: 'Volkswagen',
  vehicleModel: 'Golf',
  vehicleYear: 2018,
  vehicleEngine: '1.4 TSI',
  dtcCodes: ['P0300'],
  symptoms: 'Motor razkodik es egeszkimaradas tapasztalhato',
};

// =============================================================================
// Tests
// =============================================================================

describe('diagnosisService', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    localStorage.setItem('access_token', 'test-token');
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  describe('streamDiagnosis', () => {
    it('should call fetch with correct URL and POST method', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([]),
      });
      global.fetch = mockFetch;

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, {});

      // Allow the async stream promise to execute
      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalledOnce();
      });

      const [url, options] = mockFetch.mock.calls[0];
      expect(url).toContain('/api/v1/diagnosis/analyze/stream');
      expect(options.method).toBe('POST');
    });

    it('should include auth token in request headers', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([]),
      });
      global.fetch = mockFetch;

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, {});

      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalledOnce();
      });

      const [, options] = mockFetch.mock.calls[0];
      expect(options.headers['Authorization']).toBe('Bearer test-token');
      expect(options.headers['Content-Type']).toBe('application/json');
      expect(options.headers['Accept']).toBe('text/event-stream');
    });

    it('should send correct request body with transformed field names', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([]),
      });
      global.fetch = mockFetch;

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, {});

      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalledOnce();
      });

      const [, options] = mockFetch.mock.calls[0];
      const body = JSON.parse(options.body);
      expect(body.vehicle_make).toBe('Volkswagen');
      expect(body.vehicle_model).toBe('Golf');
      expect(body.vehicle_year).toBe(2018);
      expect(body.vehicle_engine).toBe('1.4 TSI');
      expect(body.dtc_codes).toEqual(['P0300']);
      expect(body.symptoms).toBe('Motor razkodik es egeszkimaradas tapasztalhato');
      expect(body.include_context).toBe(true);
      expect(body.include_progress).toBe(true);
    });

    it('should call onStart when start event is received', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([
          {
            event_type: 'start',
            data: { message: 'Diagnosztika indítása' },
            progress: 0,
          },
        ]),
      });
      global.fetch = mockFetch;

      const onStart = vi.fn();
      const callbacks: StreamingCallbacks = { onStart };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onStart).toHaveBeenCalledOnce();
      });

      expect(onStart).toHaveBeenCalledWith({ message: 'Diagnosztika indítása' });
    });

    it('should call onProgress with progress value and stage name', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([
          {
            event_type: 'context',
            data: { stage: 'Kontextus gyűjtés' },
            progress: 0.25,
          },
          {
            event_type: 'analysis',
            data: { stage: 'Elemzés' },
            progress: 0.5,
          },
        ]),
      });
      global.fetch = mockFetch;

      const onProgress = vi.fn();
      const callbacks: StreamingCallbacks = { onProgress };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onProgress).toHaveBeenCalledTimes(2);
      });

      expect(onProgress).toHaveBeenNthCalledWith(1, 0.25, 'Kontextus gyűjtés');
      expect(onProgress).toHaveBeenNthCalledWith(2, 0.5, 'Elemzés');
    });

    it('should call onComplete when complete event is received', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([
          {
            event_type: 'start',
            data: { message: 'Started' },
            progress: 0,
          },
          {
            event_type: 'complete',
            data: { diagnosis_id: 'final-123', confidence: 0.85 },
            progress: 1.0,
          },
        ]),
      });
      global.fetch = mockFetch;

      const onComplete = vi.fn();
      const callbacks: StreamingCallbacks = { onComplete };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onComplete).toHaveBeenCalledOnce();
      });

      expect(onComplete).toHaveBeenCalledWith({
        diagnosis_id: 'final-123',
        confidence: 0.85,
      });
    });

    it('should call onError when error event is received in SSE stream', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([
          {
            event_type: 'error',
            data: { message: 'LLM service unavailable' },
            progress: 0.3,
          },
        ]),
      });
      global.fetch = mockFetch;

      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledOnce();
      });

      const error = onError.mock.calls[0][0];
      expect(error.message).toBe('LLM service unavailable');
    });

    it('should dispatch events to correct typed callbacks', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([
          { event_type: 'start', data: { step: 'init' }, progress: 0 },
          { event_type: 'context', data: { dtcInfo: 'P0300' }, progress: 0.2 },
          { event_type: 'cause', data: { cause: 'Ignition coil' }, progress: 0.5 },
          { event_type: 'repair', data: { action: 'Replace coil' }, progress: 0.7 },
          { event_type: 'warning', data: { warning: 'Check engine' }, progress: 0.8 },
          { event_type: 'complete', data: { done: true }, progress: 1.0 },
        ]),
      });
      global.fetch = mockFetch;

      const onStart = vi.fn();
      const onContext = vi.fn();
      const onCause = vi.fn();
      const onRepair = vi.fn();
      const onWarning = vi.fn();
      const onComplete = vi.fn();

      const callbacks: StreamingCallbacks = {
        onStart,
        onContext,
        onCause,
        onRepair,
        onWarning,
        onComplete,
      };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onComplete).toHaveBeenCalledOnce();
      });

      expect(onStart).toHaveBeenCalledWith({ step: 'init' });
      expect(onContext).toHaveBeenCalledWith({ dtcInfo: 'P0300' });
      expect(onCause).toHaveBeenCalledWith({ cause: 'Ignition coil' });
      expect(onRepair).toHaveBeenCalledWith({ action: 'Replace coil' });
      expect(onWarning).toHaveBeenCalledWith({ warning: 'Check engine' });
      expect(onComplete).toHaveBeenCalledWith({ done: true });
    });

    it('should return an AbortController', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([]),
      });
      global.fetch = mockFetch;

      const { streamDiagnosis } = await import('../diagnosisService');
      const controller = streamDiagnosis(validFormData, {});

      expect(controller).toBeInstanceOf(AbortController);
    });

    it('should pass abort signal to fetch', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([]),
      });
      global.fetch = mockFetch;

      const { streamDiagnosis } = await import('../diagnosisService');
      const controller = streamDiagnosis(validFormData, {});

      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalledOnce();
      });

      const [, options] = mockFetch.mock.calls[0];
      expect(options.signal).toBe(controller.signal);
    });

    it('should handle network errors gracefully', async () => {
      const mockFetch = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'));
      global.fetch = mockFetch;

      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledOnce();
      });

      const error = onError.mock.calls[0][0];
      expect(error.message).toContain('Halozati hiba');
    });

    it('should handle HTTP error responses', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: vi.fn().mockResolvedValue({ detail: 'Internal server error' }),
      });
      global.fetch = mockFetch;

      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledOnce();
      });

      const error = onError.mock.calls[0][0];
      expect(error.message).toBe('Internal server error');
      expect(error.status).toBe(500);
    });

    it('should handle HTTP error responses with non-JSON body', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: false,
        status: 502,
        json: vi.fn().mockRejectedValue(new SyntaxError('Unexpected token')),
      });
      global.fetch = mockFetch;

      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledOnce();
      });

      const error = onError.mock.calls[0][0];
      expect(error.status).toBe(502);
      expect(error.message).toContain('502');
    });

    it('should handle missing response body (no streaming support)', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: null,
      });
      global.fetch = mockFetch;

      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledOnce();
      });

      const error = onError.mock.calls[0][0];
      expect(error.message).toContain('streaminget');
    });

    it('should not call onError when abort is triggered', async () => {
      const abortError = new DOMException('The operation was aborted', 'AbortError');
      const mockFetch = vi.fn().mockRejectedValue(abortError);
      global.fetch = mockFetch;

      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      const controller = streamDiagnosis(validFormData, callbacks);
      controller.abort();

      // Give time for the async code to run
      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(onError).not.toHaveBeenCalled();
    });

    it('should call onError for validation failures without calling fetch', async () => {
      const mockFetch = vi.fn();
      global.fetch = mockFetch;

      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onError };

      const invalidData = {
        vehicleMake: '',
        vehicleModel: '',
        vehicleYear: 2018,
        dtcCodes: ['P0300'],
        symptoms: 'Motor razkodik es egeszkimaradas tapasztalhato',
      };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(invalidData, callbacks);

      // Validation errors are dispatched via queueMicrotask
      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledOnce();
      });

      expect(mockFetch).not.toHaveBeenCalled();
      const error = onError.mock.calls[0][0];
      expect(error.message).toContain('Gyarto megadasa kotelezo');
    });

    it('should handle malformed SSE data gracefully (skip bad JSON)', async () => {
      // Create a stream with a malformed JSON line manually
      const encoder = new TextEncoder();
      const mockStream = new ReadableStream<Uint8Array>({
        start(controller) {
          // Valid event
          const validEvent = {
            event_type: 'start',
            data: { message: 'ok' },
            progress: 0,
            diagnosis_id: 'test-123',
            timestamp: '2026-03-20T10:00:00Z',
          };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(validEvent)}\n\n`));
          // Malformed JSON
          controller.enqueue(encoder.encode('data: {not valid json\n\n'));
          // Another valid event
          const completeEvent = {
            event_type: 'complete',
            data: { done: true },
            progress: 1.0,
            diagnosis_id: 'test-123',
            timestamp: '2026-03-20T10:00:00Z',
          };
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(completeEvent)}\n\n`));
          controller.close();
        },
      });

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: mockStream,
      });
      global.fetch = mockFetch;

      const onStart = vi.fn();
      const onComplete = vi.fn();
      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onStart, onComplete, onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onComplete).toHaveBeenCalledOnce();
      });

      // Start and complete should fire; malformed line should be skipped silently
      expect(onStart).toHaveBeenCalledOnce();
      expect(onError).not.toHaveBeenCalled();
    });

    it('should not include Authorization header when no token is stored', async () => {
      localStorage.removeItem('access_token');

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([]),
      });
      global.fetch = mockFetch;

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, {});

      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalledOnce();
      });

      const [, options] = mockFetch.mock.calls[0];
      expect(options.headers['Authorization']).toBeUndefined();
    });

    it('should uppercase DTC codes in the request body', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([]),
      });
      global.fetch = mockFetch;

      const dataWithLowerCase = {
        ...validFormData,
        dtcCodes: ['p0300', 'p0301'],
      };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(dataWithLowerCase, {});

      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalledOnce();
      });

      const [, options] = mockFetch.mock.calls[0];
      const body = JSON.parse(options.body);
      expect(body.dtc_codes).toEqual(['P0300', 'P0301']);
    });

    it('should stop processing events after an error event', async () => {
      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: createMockSSEStream([
          { event_type: 'start', data: { step: 'init' }, progress: 0 },
          { event_type: 'error', data: { message: 'Failed' }, progress: 0.3 },
          { event_type: 'complete', data: { done: true }, progress: 1.0 },
        ]),
      });
      global.fetch = mockFetch;

      const onStart = vi.fn();
      const onComplete = vi.fn();
      const onError = vi.fn();
      const callbacks: StreamingCallbacks = { onStart, onComplete, onError };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledOnce();
      });

      // Give extra time to ensure complete is NOT called
      await new Promise((resolve) => setTimeout(resolve, 50));

      expect(onStart).toHaveBeenCalledOnce();
      expect(onComplete).not.toHaveBeenCalled();
    });

    it('should handle SSE events with event: prefix lines', async () => {
      // The backend sends "event: {type}\ndata: {json}\n\n" format
      const encoder = new TextEncoder();
      const payload = {
        event_type: 'start',
        data: { message: 'hello' },
        progress: 0,
        diagnosis_id: 'test-123',
        timestamp: '2026-03-20T10:00:00Z',
      };
      const sseText = `event: start\ndata: ${JSON.stringify(payload)}\n\n`;

      const mockStream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoder.encode(sseText));
          controller.close();
        },
      });

      const mockFetch = vi.fn().mockResolvedValue({
        ok: true,
        body: mockStream,
      });
      global.fetch = mockFetch;

      const onStart = vi.fn();
      const callbacks: StreamingCallbacks = { onStart };

      const { streamDiagnosis } = await import('../diagnosisService');
      streamDiagnosis(validFormData, callbacks);

      await vi.waitFor(() => {
        expect(onStart).toHaveBeenCalledOnce();
      });

      expect(onStart).toHaveBeenCalledWith({ message: 'hello' });
    });
  });
});
