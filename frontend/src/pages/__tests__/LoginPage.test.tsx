import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../../test/test-utils';
import userEvent from '@testing-library/user-event';
import LoginPage from '../LoginPage';

// Mock AuthContext
const mockLogin = vi.fn();
const mockClearError = vi.fn();

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    login: mockLogin,
    isLoading: false,
    error: null,
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
    // "Bejelentkezes" appears both in heading and submit button
    expect(screen.getByRole('heading', { name: /bejelentkezes/i })).toBeInTheDocument();
  });

  it('should render registration and forgot password links', () => {
    render(<LoginPage />);
    expect(screen.getByText('Regisztraljon most')).toBeInTheDocument();
    expect(screen.getByText('Elfelejtett jelszo?')).toBeInTheDocument();
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

  it('should call login with credentials on valid submit', async () => {
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

  it('should toggle password visibility', async () => {
    const user = userEvent.setup();
    render(<LoginPage />);

    const passwordInput = screen.getByLabelText('Jelszo');
    expect(passwordInput).toHaveAttribute('type', 'password');

    // Click the toggle button (eye icon)
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

  it('should not call login without user interaction', () => {
    render(<LoginPage />);
    expect(mockLogin).not.toHaveBeenCalled();
  });
});
