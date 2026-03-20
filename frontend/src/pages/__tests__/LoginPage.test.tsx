import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '../../test/test-utils';
import userEvent from '@testing-library/user-event';
import LoginPage from '../LoginPage';

// Mock AuthContext - dynamic mock values
const mockLogin = vi.fn();
const mockClearError = vi.fn();
let mockIsLoading = false;
let mockError: string | null = null;

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    login: mockLogin,
    isLoading: mockIsLoading,
    error: mockError,
    clearError: mockClearError,
  }),
}));

// Mock react-router-dom navigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useLocation: () => ({ state: null, pathname: '/login' }),
  };
});

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsLoading = false;
    mockError = null;
  });

  it('should render login form with email and password fields', () => {
    render(<LoginPage />);
    expect(screen.getByLabelText('Email cim')).toBeInTheDocument();
    expect(screen.getByLabelText('Jelszo')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /bejelentkezes/i })).toBeInTheDocument();
  });

  it('should render AutoCognitix branding', () => {
    render(<LoginPage />);
    expect(screen.getByText('AutoCognitix')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /bejelentkezes/i })).toBeInTheDocument();
  });

  it('should have a link to the register page', () => {
    render(<LoginPage />);
    const registerLink = screen.getByText('Regisztraljon most');
    expect(registerLink).toBeInTheDocument();
    expect(registerLink.closest('a')).toHaveAttribute('href', '/register');
  });

  it('should have a link to forgot password', () => {
    render(<LoginPage />);
    const forgotLink = screen.getByText('Elfelejtett jelszo?');
    expect(forgotLink).toBeInTheDocument();
    expect(forgotLink.closest('a')).toHaveAttribute('href', '/forgot-password');
  });

  it('should have required attribute on email and password fields', () => {
    render(<LoginPage />);
    expect(screen.getByLabelText('Email cim')).toHaveAttribute('required');
    expect(screen.getByLabelText('Jelszo')).toHaveAttribute('required');
  });

  it('should have correct input types', () => {
    render(<LoginPage />);
    expect(screen.getByLabelText('Email cim')).toHaveAttribute('type', 'email');
    expect(screen.getByLabelText('Jelszo')).toHaveAttribute('type', 'password');
  });

  it('should show validation error when email is empty on submit', async () => {
    render(<LoginPage />);

    // Use fireEvent.submit to bypass browser-level required validation
    const form = screen.getByRole('button', { name: /bejelentkezes/i }).closest('form')!;
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText('Kerem adja meg az email cimet')).toBeInTheDocument();
    });
    expect(mockLogin).not.toHaveBeenCalled();
  });

  it('should show validation error when password is empty on submit', async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');

    // Use fireEvent.submit to bypass browser-level required validation
    const form = screen.getByRole('button', { name: /bejelentkezes/i }).closest('form')!;
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText('Kerem adja meg a jelszot')).toBeInTheDocument();
    });
    expect(mockLogin).not.toHaveBeenCalled();
  });

  it('should show error message on failed login from AuthContext', () => {
    mockError = 'Hibas email vagy jelszo';
    render(<LoginPage />);

    expect(screen.getByText('Hibas email vagy jelszo')).toBeInTheDocument();
  });

  it('should call login with correct credentials on valid submit', async () => {
    mockLogin.mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), 'password123');
    await user.click(screen.getByRole('button', { name: /bejelentkezes/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
    });
  });

  it('should navigate to home after successful login', async () => {
    mockLogin.mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), 'password123');
    await user.click(screen.getByRole('button', { name: /bejelentkezes/i }));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/', { replace: true });
    });
  });

  it('should not navigate when login fails', async () => {
    mockLogin.mockRejectedValue(new Error('Login failed'));
    const user = userEvent.setup();
    render(<LoginPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), 'wrongpassword');
    await user.click(screen.getByRole('button', { name: /bejelentkezes/i }));

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalled();
    });
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('should toggle password visibility', async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    const passwordInput = screen.getByLabelText('Jelszo');
    expect(passwordInput).toHaveAttribute('type', 'password');

    const toggleButton = passwordInput.parentElement?.querySelector('button');
    expect(toggleButton).toBeTruthy();
    await user.click(toggleButton!);

    expect(passwordInput).toHaveAttribute('type', 'text');
  });

  it('should update email input value on type', async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    const emailInput = screen.getByLabelText('Email cim');
    await user.type(emailInput, 'hello@test.com');
    expect(emailInput).toHaveValue('hello@test.com');
  });

  it('should disable submit button when loading', () => {
    mockIsLoading = true;
    render(<LoginPage />);

    const submitButton = screen.getByRole('button', { name: /bejelentkezes/i });
    expect(submitButton).toBeDisabled();
  });

  it('should clear error on form submit', async () => {
    const user = userEvent.setup();
    mockLogin.mockResolvedValue(undefined);
    render(<LoginPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), 'password123');
    await user.click(screen.getByRole('button', { name: /bejelentkezes/i }));

    await waitFor(() => {
      expect(mockClearError).toHaveBeenCalled();
    });
  });

  it('should not call login without user interaction', () => {
    render(<LoginPage />);
    expect(mockLogin).not.toHaveBeenCalled();
  });
});
