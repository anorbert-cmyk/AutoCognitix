import { describe, it, expect, vi, beforeEach } from 'vitest';
import axios from 'axios';

vi.mock('axios', () => {
  const mockAxiosInstance = {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    patch: vi.fn(),
    delete: vi.fn(),
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
    defaults: { headers: { common: {} } },
  };
  return {
    default: {
      create: vi.fn(() => mockAxiosInstance),
      isAxiosError: vi.fn((err: unknown) => (err as { isAxiosError?: boolean })?.isAxiosError === true),
    },
    AxiosError: class AxiosError extends Error {
      isAxiosError = true;
    },
  };
});

describe('ApiError', () => {
  it('should create an ApiError with default values', () => {
    const { ApiError } = require('../api');
    const error = new ApiError('test error');
    expect(error.message).toBe('test error');
    expect(error.status).toBe(500);
    expect(error.detail).toBe('test error');
    expect(error.isNetworkError).toBe(false);
    expect(error.name).toBe('ApiError');
  });

  it('should create an ApiError with custom status', () => {
    const { ApiError } = require('../api');
    const error = new ApiError('not found', 404, 'Resource not found');
    expect(error.status).toBe(404);
    expect(error.detail).toBe('Resource not found');
  });

  it('should create a network error', () => {
    const { ApiError } = require('../api');
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
    const { ApiError } = require('../api');
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
    const { ApiError } = require('../api');
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
    const { ApiError } = require('../api');
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
    const { ApiError } = require('../api');
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
    const { ApiError } = require('../api');
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
    const { ApiError } = require('../api');
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

  it('should create ApiError from AxiosError with 400 response using detail', () => {
    const { ApiError } = require('../api');
    const axiosError = {
      response: {
        status: 400,
        data: { detail: 'Bad request data' },
      },
      message: 'Bad Request',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(400);
    expect(apiError.message).toBe('Bad request data');
  });

  it('should create ApiError from AxiosError with 404 response using detail', () => {
    const { ApiError } = require('../api');
    const axiosError = {
      response: {
        status: 404,
        data: { detail: 'Diagnosis not found' },
      },
      message: 'Not Found',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(404);
    expect(apiError.message).toBe('Diagnosis not found');
  });

  it('should create ApiError from AxiosError with 502 response', () => {
    const { ApiError } = require('../api');
    const axiosError = {
      response: {
        status: 502,
        data: { detail: 'Service unavailable' },
      },
      message: 'Bad Gateway',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(502);
    expect(apiError.message).toBe('Service unavailable');
  });

  it('should create ApiError from AxiosError with unknown status using detail', () => {
    const { ApiError } = require('../api');
    const axiosError = {
      response: {
        status: 418,
        data: { detail: "I'm a teapot" },
      },
      message: "I'm a teapot",
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.status).toBe(418);
    expect(apiError.message).toBe("I'm a teapot");
  });

  it('should carry field information from AxiosError response', () => {
    const { ApiError } = require('../api');
    const axiosError = {
      response: {
        status: 422,
        data: { detail: 'Invalid field', code: 'VALIDATION_ERROR', field: 'email' },
      },
      message: 'Validation Error',
      isAxiosError: true,
      config: {},
      toJSON: () => ({}),
    } as any;

    const apiError = ApiError.fromAxiosError(axiosError);
    expect(apiError.code).toBe('VALIDATION_ERROR');
    expect(apiError.field).toBe('email');
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

  it('should be generic over data type', async () => {
    const { createLoadingState } = await import('../api');
    const numState = createLoadingState<number>();
    const objState = createLoadingState<{ id: string }>();
    expect(numState.data).toBeNull();
    expect(objState.data).toBeNull();
  });
});

describe('setCsrfToken and getCsrfToken', () => {
  beforeEach(() => {
    // Reset CSRF token before each test
    const { setCsrfToken } = require('../api');
    setCsrfToken(null);
  });

  it('should set and get CSRF token', () => {
    const { setCsrfToken, getCsrfToken } = require('../api');
    setCsrfToken('test-csrf-token');
    expect(getCsrfToken()).toBe('test-csrf-token');
  });

  it('should return null when no CSRF token is set', () => {
    const { getCsrfToken } = require('../api');
    expect(getCsrfToken()).toBeNull();
  });

  it('should clear CSRF token when set to null', () => {
    const { setCsrfToken, getCsrfToken } = require('../api');
    setCsrfToken('some-token');
    expect(getCsrfToken()).toBe('some-token');
    setCsrfToken(null);
    expect(getCsrfToken()).toBeNull();
  });

  it('should overwrite previous CSRF token', () => {
    const { setCsrfToken, getCsrfToken } = require('../api');
    setCsrfToken('first-token');
    setCsrfToken('second-token');
    expect(getCsrfToken()).toBe('second-token');
  });
});

describe('axios mock - api instance creation', () => {
  it('should create axios instance via axios.create (mocked)', () => {
    // Verify that the mocked axios.create was called during module load
    expect(axios.create).toHaveBeenCalled();
  });
});
