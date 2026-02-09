import { describe, it, expect } from 'vitest';
import { ApiError } from '../api';

describe('ApiError', () => {
  it('should create an ApiError with default values', () => {
    const error = new ApiError('test error');
    expect(error.message).toBe('test error');
    expect(error.status).toBe(500);
    expect(error.detail).toBe('test error');
    expect(error.isNetworkError).toBe(false);
    expect(error.name).toBe('ApiError');
  });

  it('should create an ApiError with custom status', () => {
    const error = new ApiError('not found', 404, 'Resource not found');
    expect(error.status).toBe(404);
    expect(error.detail).toBe('Resource not found');
  });

  it('should create a network error', () => {
    const error = new ApiError(
      'Network error',
      0,
      'Network error',
      'NETWORK_ERROR',
      undefined,
      true,
    );
    expect(error.isNetworkError).toBe(true);
    expect(error.code).toBe('NETWORK_ERROR');
  });

  it('should create ApiError from AxiosError without response (network error)', () => {
    const axiosError = {
      response: undefined,
      message: 'Network Error',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.isNetworkError).toBe(true);
    expect(apiError.status).toBe(0);
    expect(apiError.code).toBe('NETWORK_ERROR');
  });

  it('should create ApiError from AxiosError with 401 response', () => {
    const axiosError = {
      response: {
        status: 401,
        data: { detail: 'Token expired' },
      },
      message: 'Unauthorized',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(401);
    expect(apiError.message).toBe('Bejelentkezés szükséges');
  });

  it('should create ApiError from AxiosError with 403 response', () => {
    const axiosError = {
      response: {
        status: 403,
        data: { detail: 'Forbidden' },
      },
      message: 'Forbidden',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(403);
    expect(apiError.message).toBe('Nincs jogosultság');
  });

  it('should create ApiError from AxiosError with 422 response', () => {
    const axiosError = {
      response: {
        status: 422,
        data: { detail: 'Invalid email format' },
      },
      message: 'Unprocessable Entity',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(422);
    expect(apiError.message).toBe('Invalid email format');
  });

  it('should create ApiError from AxiosError with 429 response', () => {
    const axiosError = {
      response: {
        status: 429,
        data: {},
      },
      message: 'Too Many Requests',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(429);
    expect(apiError.message).toBe('Túl sok kérés - kérjük várjon');
  });

  it('should create ApiError from AxiosError with 500 response', () => {
    const axiosError = {
      response: {
        status: 500,
        data: {},
      },
      message: 'Internal Server Error',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(500);
    expect(apiError.message).toBe('Szerver hiba - kérjük próbálja újra később');
  });
});

describe('createLoadingState', () => {
  it('should create initial loading state', async () => {
    const { createLoadingState } = await import('../api');
    const state = createLoadingState<string>();
    expect(state.data).toBeNull();
    expect(state.isLoading).toBe(false);
    expect(state.error).toBeNull();
  });
});
