import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { StreamingCallbacks } from '../../types/streaming';

// =============================================================================
// Mock the api default export for non-streaming service function tests
// =============================================================================

vi.mock('../api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api')>();
  return {
    ...actual,
    default: {
      get: vi.fn(),
      post: vi.fn(),
      put: vi.fn(),
      delete: vi.fn(),
    },
  };
});

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

    it('should use cookie-based auth with credentials include', async () => {
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
      expect(options.credentials).toBe('include');
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

      // onProgress now receives event_type (not data.stage) for step index lookup
      expect(onProgress).toHaveBeenNthCalledWith(1, 0.25, 'context');
      expect(onProgress).toHaveBeenNthCalledWith(2, 0.5, 'analysis');
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

    it('should use credentials include regardless of stored tokens', async () => {
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
      expect(options.credentials).toBe('include');
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

// =============================================================================
// Non-streaming service functions — api default export is mocked above
// =============================================================================

describe('diagnosisService (api-mocked)', () => {
  let mockApi: { get: ReturnType<typeof vi.fn>; post: ReturnType<typeof vi.fn>; delete: ReturnType<typeof vi.fn> };

  beforeEach(async () => {
    vi.clearAllMocks();
    const apiModule = await import('../api');
    mockApi = apiModule.default as unknown as typeof mockApi;
  });

  describe('analyzeDiagnosis', () => {
    const validData = {
      vehicleMake: 'Volkswagen',
      vehicleModel: 'Golf',
      vehicleYear: 2018,
      vehicleEngine: '1.4 TSI',
      dtcCodes: ['P0300'],
      symptoms: 'Motor razkodik es egeszkimaradas tapasztalhato',
    };

    it('should call api.post with transformed snake_case body', async () => {
      const mockResponse = { data: { id: 'diag-1', confidence_score: 0.9 } };
      mockApi.post = vi.fn().mockResolvedValue(mockResponse);

      const { analyzeDiagnosis } = await import('../diagnosisService');
      const result = await analyzeDiagnosis(validData);

      expect(mockApi.post).toHaveBeenCalledOnce();
      const [endpoint, body] = (mockApi.post as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(endpoint).toBe('/diagnosis/analyze');
      expect(body.vehicle_make).toBe('Volkswagen');
      expect(body.vehicle_model).toBe('Golf');
      expect(body.dtc_codes).toEqual(['P0300']);
      expect(result).toEqual(mockResponse.data);
    });

    it('should throw ApiError when validation fails (missing make)', async () => {
      const { analyzeDiagnosis, ApiError: _ApiError } = await import('../diagnosisService');
      const { ApiError } = await import('../api');

      await expect(
        analyzeDiagnosis({ ...validData, vehicleMake: '' }),
      ).rejects.toBeInstanceOf(ApiError);
    });

    it('should uppercase DTC codes before sending', async () => {
      const mockResponse = { data: { id: 'diag-1' } };
      mockApi.post = vi.fn().mockResolvedValue(mockResponse);

      const { analyzeDiagnosis } = await import('../diagnosisService');
      await analyzeDiagnosis({ ...validData, dtcCodes: ['p0300', 'p0301'] });

      const [, body] = (mockApi.post as ReturnType<typeof vi.fn>).mock.calls[0];
      expect(body.dtc_codes).toEqual(['P0300', 'P0301']);
    });
  });

  describe('getDiagnosisById', () => {
    it('should call api.get with the correct endpoint', async () => {
      const mockResponse = { data: { id: 'abc-123', vehicle_make: 'BMW' } };
      mockApi.get = vi.fn().mockResolvedValue(mockResponse);

      const { getDiagnosisById } = await import('../diagnosisService');
      const result = await getDiagnosisById('abc-123');

      expect(mockApi.get).toHaveBeenCalledWith('/diagnosis/abc-123');
      expect(result).toEqual(mockResponse.data);
    });

    it('should throw ApiError when id is empty', async () => {
      const { getDiagnosisById } = await import('../diagnosisService');
      const { ApiError } = await import('../api');

      await expect(getDiagnosisById('')).rejects.toBeInstanceOf(ApiError);
    });
  });

  describe('deleteDiagnosis', () => {
    it('should call api.delete with the correct endpoint', async () => {
      const mockResponse = { data: { success: true, message: 'Deleted' } };
      mockApi.delete = vi.fn().mockResolvedValue(mockResponse);

      const { deleteDiagnosis } = await import('../diagnosisService');
      const result = await deleteDiagnosis('diag-999');

      expect(mockApi.delete).toHaveBeenCalledWith('/diagnosis/diag-999');
      expect(result.success).toBe(true);
    });

    it('should throw ApiError when id is empty', async () => {
      const { deleteDiagnosis } = await import('../diagnosisService');
      const { ApiError } = await import('../api');

      await expect(deleteDiagnosis('')).rejects.toBeInstanceOf(ApiError);
    });
  });

  describe('validateDiagnosisRequest', () => {
    it('should return no errors for valid data', async () => {
      const { validateDiagnosisRequest } = await import('../diagnosisService');
      const errors = validateDiagnosisRequest({
        vehicleMake: 'Ford',
        vehicleModel: 'Focus',
        vehicleYear: 2020,
        dtcCodes: ['P0100'],
        symptoms: 'Engine stutter at idle speed',
      });
      expect(errors).toHaveLength(0);
    });

    it('should return error when vehicleMake is empty', async () => {
      const { validateDiagnosisRequest } = await import('../diagnosisService');
      const errors = validateDiagnosisRequest({
        vehicleMake: '',
        vehicleModel: 'Focus',
        vehicleYear: 2020,
        dtcCodes: ['P0100'],
        symptoms: 'Engine stutter at idle speed',
      });
      expect(errors.some((e) => e.includes('Gyarto'))).toBe(true);
    });

    it('should return error when vehicleModel is empty', async () => {
      const { validateDiagnosisRequest } = await import('../diagnosisService');
      const errors = validateDiagnosisRequest({
        vehicleMake: 'Ford',
        vehicleModel: '',
        vehicleYear: 2020,
        dtcCodes: ['P0100'],
        symptoms: 'Engine stutter at idle speed',
      });
      expect(errors.some((e) => e.includes('Modell'))).toBe(true);
    });

    it('should return error for invalid year', async () => {
      const { validateDiagnosisRequest } = await import('../diagnosisService');
      const errors = validateDiagnosisRequest({
        vehicleMake: 'Ford',
        vehicleModel: 'Focus',
        vehicleYear: 1800,
        dtcCodes: ['P0100'],
        symptoms: 'Engine stutter at idle speed',
      });
      expect(errors.some((e) => e.includes('evjarat'))).toBe(true);
    });

    it('should return error when symptoms are too short', async () => {
      const { validateDiagnosisRequest } = await import('../diagnosisService');
      const errors = validateDiagnosisRequest({
        vehicleMake: 'Ford',
        vehicleModel: 'Focus',
        vehicleYear: 2020,
        dtcCodes: ['P0100'],
        symptoms: 'Short',
      });
      expect(errors.some((e) => e.includes('tunetleiras'))).toBe(true);
    });

    it('should return error when no DTC codes provided', async () => {
      const { validateDiagnosisRequest } = await import('../diagnosisService');
      const errors = validateDiagnosisRequest({
        vehicleMake: 'Ford',
        vehicleModel: 'Focus',
        vehicleYear: 2020,
        dtcCodes: [],
        symptoms: 'Engine stutter at idle speed',
      });
      expect(errors.some((e) => e.includes('DTC'))).toBe(true);
    });
  });

  describe('formatConfidenceScore', () => {
    it('should format 0.85 as 85%', async () => {
      const { formatConfidenceScore } = await import('../diagnosisService');
      expect(formatConfidenceScore(0.85)).toBe('85%');
    });

    it('should format 0 as 0%', async () => {
      const { formatConfidenceScore } = await import('../diagnosisService');
      expect(formatConfidenceScore(0)).toBe('0%');
    });

    it('should format 1 as 100%', async () => {
      const { formatConfidenceScore } = await import('../diagnosisService');
      expect(formatConfidenceScore(1)).toBe('100%');
    });
  });

  describe('getConfidenceLevelHu', () => {
    it('should return Nagyon magas for score >= 0.8', async () => {
      const { getConfidenceLevelHu } = await import('../diagnosisService');
      expect(getConfidenceLevelHu(0.9)).toBe('Nagyon magas');
    });

    it('should return Alacsony for score between 0.2 and 0.4', async () => {
      const { getConfidenceLevelHu } = await import('../diagnosisService');
      expect(getConfidenceLevelHu(0.3)).toBe('Alacsony');
    });
  });

  describe('formatTime', () => {
    it('should format minutes under an hour', async () => {
      const { formatTime } = await import('../diagnosisService');
      expect(formatTime(45)).toBe('45 perc');
    });

    it('should format exactly 60 minutes as 1 hour', async () => {
      const { formatTime } = await import('../diagnosisService');
      expect(formatTime(60)).toBe('1 ora');
    });

    it('should format 90 minutes as 1 ora 30 perc', async () => {
      const { formatTime } = await import('../diagnosisService');
      expect(formatTime(90)).toBe('1 ora 30 perc');
    });

    it('should return Nincs becsles for undefined', async () => {
      const { formatTime } = await import('../diagnosisService');
      expect(formatTime(undefined)).toBe('Nincs becslés');
    });
  });
});
