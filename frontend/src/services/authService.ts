/**
 * Authentication service for AutoCognitix.
 *
 * Handles user registration, login, logout, token refresh,
 * and password management.
 */

import api, { ApiError } from './api'

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
// Token Storage
// =============================================================================

const ACCESS_TOKEN_KEY = 'access_token'
const REFRESH_TOKEN_KEY = 'refresh_token'

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_TOKEN_KEY)
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_TOKEN_KEY)
}

export function setTokens(tokens: AuthTokens): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, tokens.access_token)
  localStorage.setItem(REFRESH_TOKEN_KEY, tokens.refresh_token)
}

export function clearTokens(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY)
  localStorage.removeItem(REFRESH_TOKEN_KEY)
}

export function isAuthenticated(): boolean {
  return !!getAccessToken()
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
 * Returns tokens and stores them in localStorage.
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

  setTokens(response.data)
  return response.data
}

/**
 * Logout the current user.
 * Invalidates tokens on server and clears local storage.
 */
export async function logout(): Promise<void> {
  const refreshToken = getRefreshToken()

  try {
    await api.post('/auth/logout', {
      refresh_token: refreshToken,
    })
  } catch (error) {
    // Continue with local logout even if server request fails
    console.warn('Server logout failed, proceeding with local logout')
  }

  clearTokens()
}

/**
 * Refresh the access token using the refresh token.
 */
export async function refreshTokens(): Promise<AuthTokens> {
  const refreshToken = getRefreshToken()

  if (!refreshToken) {
    throw new ApiError('Nincs refresh token', 401, 'Nincs refresh token')
  }

  const response = await api.post<AuthTokens>('/auth/refresh', {
    refresh_token: refreshToken,
  })

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
