/**
 * Authentication service for AutoCognitix.
 *
 * Handles user registration, login, logout, token refresh,
 * and password management.
 */

import api, { ApiError, setCsrfToken, getCsrfToken } from './api'

// =============================================================================
// Types
// =============================================================================

export interface User {
  id: string
  email: string
  full_name?: string
  is_active: boolean
  role: 'user' | 'mechanic' | 'admin'
  created_at?: string
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterData {
  email: string
  password: string
  full_name?: string
  role?: 'user' | 'mechanic'
}

export interface AuthTokens {
  access_token: string
  refresh_token: string
  token_type: string
  csrf_token?: string
}

export interface UpdateProfileData {
  full_name?: string
  email?: string
}

export interface ChangePasswordData {
  current_password: string
  new_password: string
}

export interface ForgotPasswordData {
  email: string
}

export interface ResetPasswordData {
  token: string
  new_password: string
}

// =============================================================================
// Auth State (cookie-based - no localStorage)
// =============================================================================

// Track authentication state in memory. Tokens are stored in httpOnly cookies
// (not accessible to JavaScript). The CSRF token is stored in memory via api.ts.

let authenticated = false

export function getAccessToken(): string | null {
  // Access token is in httpOnly cookie, not accessible to JS.
  // Return null - callers should rely on isAuthenticated() instead.
  return null
}

export function getRefreshToken(): string | null {
  // Refresh token is in httpOnly cookie, not accessible to JS.
  return null
}

export function setTokens(tokens: AuthTokens): void {
  // Tokens are set as httpOnly cookies by the backend.
  // We only store the CSRF token in memory.
  if (tokens.csrf_token) {
    setCsrfToken(tokens.csrf_token)
  }
  authenticated = true
}

export function clearTokens(): void {
  // Cookies are cleared by the backend on logout.
  // We only clear the in-memory CSRF token.
  setCsrfToken(null)
  authenticated = false
}

export function isAuthenticated(): boolean {
  // Check if we have a CSRF token (set on login) as a proxy for auth state
  return authenticated || !!getCsrfToken()
}

// =============================================================================
// API Functions
// =============================================================================

/**
 * Register a new user account.
 */
export async function register(data: RegisterData): Promise<User> {
  const response = await api.post<User>('/auth/register', data)
  return response.data
}

/**
 * Login with email and password.
 * Tokens are set as httpOnly cookies by the backend.
 * CSRF token is stored in memory for state-changing requests.
 */
export async function login(credentials: LoginCredentials): Promise<AuthTokens> {
  // OAuth2 password flow requires form data
  const formData = new URLSearchParams()
  formData.append('username', credentials.email)
  formData.append('password', credentials.password)

  const response = await api.post<AuthTokens>('/auth/login', formData, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  })

  // Store CSRF token in memory (tokens are in httpOnly cookies)
  setTokens(response.data)
  return response.data
}

/**
 * Logout the current user.
 * Server invalidates tokens and clears httpOnly cookies.
 */
export async function logout(): Promise<void> {
  try {
    // Cookies (access + refresh tokens) are sent automatically
    await api.post('/auth/logout')
  } catch (error) {
    // Continue with local logout even if server request fails
    console.warn('Server logout failed, proceeding with local logout')
  }

  clearTokens()
}

/**
 * Refresh the access token using the refresh token cookie.
 */
export async function refreshTokens(): Promise<AuthTokens> {
  // Refresh token is sent automatically via httpOnly cookie
  const response = await api.post<AuthTokens>('/auth/refresh')

  // Update CSRF token in memory
  setTokens(response.data)
  return response.data
}

/**
 * Get the current authenticated user's profile.
 */
export async function getCurrentUser(): Promise<User> {
  const response = await api.get<User>('/auth/me')
  return response.data
}

/**
 * Update the current user's profile.
 */
export async function updateProfile(data: UpdateProfileData): Promise<User> {
  const response = await api.put<User>('/auth/me', data)
  return response.data
}

/**
 * Change the current user's password.
 */
export async function changePassword(data: ChangePasswordData): Promise<void> {
  await api.put('/auth/me/password', data)
}

/**
 * Request a password reset email.
 * Always returns success for security (doesn't reveal if email exists).
 */
export async function forgotPassword(data: ForgotPasswordData): Promise<void> {
  await api.post('/auth/forgot-password', data)
}

/**
 * Reset password using the token from email.
 */
export async function resetPassword(data: ResetPasswordData): Promise<void> {
  await api.post('/auth/reset-password', data)
}

// =============================================================================
// Exports
// =============================================================================

export const authService = {
  register,
  login,
  logout,
  refreshTokens,
  getCurrentUser,
  updateProfile,
  changePassword,
  forgotPassword,
  resetPassword,
  getAccessToken,
  getRefreshToken,
  setTokens,
  clearTokens,
  isAuthenticated,
}

export default authService
