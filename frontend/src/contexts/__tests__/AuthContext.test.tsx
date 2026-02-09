import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AuthProvider, useAuth } from '../AuthContext';

// Mock authService
vi.mock('../../services/authService', () => ({
  login: vi.fn(),
  logout: vi.fn(),
  register: vi.fn(),
  getCurrentUser: vi.fn(),
  updateProfile: vi.fn(),
  changePassword: vi.fn(),
  isAuthenticated: vi.fn(),
  clearTokens: vi.fn(),
}));

// Mock api
vi.mock('../../services/api', () => ({
  ApiError: class ApiError extends Error {
    detail: string;
    constructor(message: string) {
      super(message);
      this.detail = message;
    }
  },
  default: {
    interceptors: {
      request: { use: vi.fn() },
      response: { use: vi.fn() },
    },
  },
}));

import * as authService from '../../services/authService';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <BrowserRouter>
        <QueryClientProvider client={queryClient}>
          <AuthProvider>{children}</AuthProvider>
        </QueryClientProvider>
      </BrowserRouter>
    );
  };
}

describe('AuthContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    vi.mocked(authService.isAuthenticated).mockReturnValue(false);
  });

  it('should have no user when not authenticated', async () => {
    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });

  it('should load existing user on mount if token exists', async () => {
    const mockUser = {
      id: '1',
      email: 'test@example.com',
      full_name: 'Test User',
      is_active: true,
      role: 'user' as const,
    };

    vi.mocked(authService.isAuthenticated).mockReturnValue(true);
    vi.mocked(authService.getCurrentUser).mockResolvedValue(mockUser);

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.user).toEqual(mockUser);
    expect(result.current.isAuthenticated).toBe(true);
  });

  it('should clear user if token is invalid on mount', async () => {
    vi.mocked(authService.isAuthenticated).mockReturnValue(true);
    vi.mocked(authService.getCurrentUser).mockRejectedValue(new Error('Invalid token'));

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.user).toBeNull();
    expect(authService.clearTokens).toHaveBeenCalled();
  });

  it('should login successfully', async () => {
    const mockUser = {
      id: '1',
      email: 'test@example.com',
      is_active: true,
      role: 'user' as const,
    };

    vi.mocked(authService.login).mockResolvedValue({
      access_token: 'token',
      refresh_token: 'refresh',
      token_type: 'bearer',
    });
    vi.mocked(authService.getCurrentUser).mockResolvedValue(mockUser);

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    await act(async () => {
      await result.current.login({ email: 'test@example.com', password: 'pass' });
    });

    expect(authService.login).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'pass',
    });
    expect(result.current.user).toEqual(mockUser);
    expect(result.current.isAuthenticated).toBe(true);
  });

  it('should call authService.login on login attempt', async () => {
    vi.mocked(authService.login).mockRejectedValue(new Error('fail'));

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    try {
      await act(async () => {
        await result.current.login({ email: 'test@example.com', password: 'wrong' });
      });
    } catch {
      // Expected
    }

    expect(authService.login).toHaveBeenCalledWith({
      email: 'test@example.com',
      password: 'wrong',
    });
  });

  it('should logout successfully', async () => {
    const mockUser = {
      id: '1',
      email: 'test@example.com',
      is_active: true,
      role: 'user' as const,
    };

    vi.mocked(authService.isAuthenticated).mockReturnValue(true);
    vi.mocked(authService.getCurrentUser).mockResolvedValue(mockUser);
    vi.mocked(authService.logout).mockResolvedValue(undefined);

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.user).toEqual(mockUser);
    });

    await act(async () => {
      await result.current.logout();
    });

    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });

  it('should have clearError function available', async () => {
    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // clearError should be callable and not throw
    expect(result.current.clearError).toBeDefined();
    act(() => {
      result.current.clearError();
    });
    expect(result.current.error).toBeNull();
  });

  it('should throw if useAuth is used outside AuthProvider', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useAuth());
    }).toThrow('useAuth must be used within an AuthProvider');

    consoleError.mockRestore();
  });
});
