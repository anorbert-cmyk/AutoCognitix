import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '../../test/test-utils';
import userEvent from '@testing-library/user-event';
import RegisterPage from '../RegisterPage';

// Mock AuthContext - dynamic mock values
const mockRegister = vi.fn();
const mockClearError = vi.fn();
let mockIsLoading = false;
let mockError: string | null = null;

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    register: mockRegister,
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
  };
});

// Valid password that meets all requirements
const VALID_PASSWORD = 'Test1234!';

describe('RegisterPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsLoading = false;
    mockError = null;
  });

  // =========================================================================
  // Rendering
  // =========================================================================

  it('should render registration form with all fields', () => {
    render(<RegisterPage />);

    expect(screen.getByLabelText(/teljes nev/i)).toBeInTheDocument();
    expect(screen.getByLabelText('Email cim')).toBeInTheDocument();
    expect(screen.getByLabelText('Jelszo')).toBeInTheDocument();
    expect(screen.getByLabelText(/jelszo megerositese/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /regisztracio$/i })).toBeInTheDocument();
  });

  it('should render AutoCognitix branding', () => {
    render(<RegisterPage />);
    expect(screen.getByText('AutoCognitix')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /regisztracio/i })).toBeInTheDocument();
  });

  it('should have a link to the login page', () => {
    render(<RegisterPage />);
    const loginLink = screen.getByText('Jelentkezzen be');
    expect(loginLink).toBeInTheDocument();
    expect(loginLink.closest('a')).toHaveAttribute('href', '/login');
  });

  it('should render role selection buttons', () => {
    render(<RegisterPage />);
    expect(screen.getByText('Felhasznalo')).toBeInTheDocument();
    expect(screen.getByText('Szerelo')).toBeInTheDocument();
  });

  it('should have required attribute on email, password, and confirm password', () => {
    render(<RegisterPage />);
    expect(screen.getByLabelText('Email cim')).toHaveAttribute('required');
    expect(screen.getByLabelText('Jelszo')).toHaveAttribute('required');
    expect(screen.getByLabelText(/jelszo megerositese/i)).toHaveAttribute('required');
  });

  // =========================================================================
  // Validation - Empty fields
  // =========================================================================

  it('should show validation error when email is empty on submit', async () => {
    render(<RegisterPage />);

    // Use fireEvent.submit to bypass browser-level required validation
    const form = screen.getByRole('button', { name: /regisztracio$/i }).closest('form')!;
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText('Kerem adja meg az email cimet')).toBeInTheDocument();
    });
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it('should show validation error when password is empty on submit', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');

    // Use fireEvent.submit to bypass browser-level required validation
    const form = screen.getByRole('button', { name: /regisztracio$/i }).closest('form')!;
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText('Kerem adja meg a jelszot')).toBeInTheDocument();
    });
    expect(mockRegister).not.toHaveBeenCalled();
  });

  // =========================================================================
  // Password strength / requirements indicator
  // =========================================================================

  it('should show password requirements when password field is focused', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    // Requirements should not be visible initially
    expect(screen.queryByText('Jelszo kovetelmenyei:')).not.toBeInTheDocument();

    // Focus the password field
    await user.click(screen.getByLabelText('Jelszo'));

    expect(screen.getByText('Jelszo kovetelmenyei:')).toBeInTheDocument();
    expect(screen.getByText('Legalabb 8 karakter')).toBeInTheDocument();
    expect(screen.getByText('Legalabb egy nagybetu')).toBeInTheDocument();
    expect(screen.getByText('Legalabb egy kisbetu')).toBeInTheDocument();
    expect(screen.getByText('Legalabb egy szam')).toBeInTheDocument();
    expect(screen.getByText('Legalabb egy specialis karakter')).toBeInTheDocument();
  });

  it('should show error when password does not meet requirements', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), 'weak');
    await user.type(screen.getByLabelText(/jelszo megerositese/i), 'weak');

    // Use fireEvent.submit to bypass browser-level required validation
    const form = screen.getByRole('button', { name: /regisztracio$/i }).closest('form')!;
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText('A jelszo nem felel meg a kovetelmenyeknek')).toBeInTheDocument();
    });
    expect(mockRegister).not.toHaveBeenCalled();
  });

  // =========================================================================
  // Password confirmation matching
  // =========================================================================

  it('should show mismatch message when passwords do not match while typing', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), 'DifferentPass1!');

    expect(screen.getByText('A jelszavak nem egyeznek')).toBeInTheDocument();
  });

  it('should show validation error on submit when passwords do not match', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), 'Different1!');

    // Use fireEvent.submit to bypass browser-level required validation
    const form = screen.getByRole('button', { name: /regisztracio$/i }).closest('form')!;
    fireEvent.submit(form);

    // Both the inline warning and the form-level error show "A jelszavak nem egyeznek"
    await waitFor(() => {
      const mismatchMessages = screen.getAllByText('A jelszavak nem egyeznek');
      expect(mismatchMessages.length).toBeGreaterThanOrEqual(1);
    });
    expect(mockRegister).not.toHaveBeenCalled();
  });

  it('should not show mismatch message when passwords match', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), VALID_PASSWORD);

    // The inline mismatch warning (below the confirm field) should not be present
    const allMismatchTexts = screen.queryAllByText('A jelszavak nem egyeznek');
    expect(allMismatchTexts).toHaveLength(0);
  });

  // =========================================================================
  // Successful registration
  // =========================================================================

  it('should call register with form data on valid submit', async () => {
    mockRegister.mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText(/teljes nev/i), 'Kovacs Janos');
    await user.type(screen.getByLabelText('Email cim'), 'kovacs@example.com');
    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), VALID_PASSWORD);
    await user.click(screen.getByRole('button', { name: /regisztracio$/i }));

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith({
        email: 'kovacs@example.com',
        password: VALID_PASSWORD,
        full_name: 'Kovacs Janos',
        role: 'user',
      });
    });
  });

  it('should navigate to home after successful registration', async () => {
    mockRegister.mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), VALID_PASSWORD);
    await user.click(screen.getByRole('button', { name: /regisztracio$/i }));

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/');
    });
  });

  it('should register with mechanic role when selected', async () => {
    mockRegister.mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'mechanic@example.com');
    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), VALID_PASSWORD);
    await user.click(screen.getByText('Szerelo'));
    await user.click(screen.getByRole('button', { name: /regisztracio$/i }));

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        expect.objectContaining({ role: 'mechanic' }),
      );
    });
  });

  it('should send undefined for full_name when left empty', async () => {
    mockRegister.mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), VALID_PASSWORD);
    await user.click(screen.getByRole('button', { name: /regisztracio$/i }));

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalledWith(
        expect.objectContaining({ full_name: undefined }),
      );
    });
  });

  // =========================================================================
  // Error display
  // =========================================================================

  it('should show error message from AuthContext', () => {
    mockError = 'Ez az email cim mar foglalt';
    render(<RegisterPage />);

    expect(screen.getByText('Ez az email cim mar foglalt')).toBeInTheDocument();
  });

  it('should not navigate when registration fails', async () => {
    mockRegister.mockRejectedValue(new Error('Registration failed'));
    const user = userEvent.setup();
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), VALID_PASSWORD);
    await user.click(screen.getByRole('button', { name: /regisztracio$/i }));

    await waitFor(() => {
      expect(mockRegister).toHaveBeenCalled();
    });
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  // =========================================================================
  // UI interactions
  // =========================================================================

  it('should toggle password visibility', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    const passwordInput = screen.getByLabelText('Jelszo');
    expect(passwordInput).toHaveAttribute('type', 'password');

    const toggleButton = passwordInput.parentElement?.querySelector('button');
    expect(toggleButton).toBeTruthy();
    await user.click(toggleButton!);

    expect(passwordInput).toHaveAttribute('type', 'text');
  });

  it('should toggle confirm password visibility', async () => {
    const user = userEvent.setup();
    render(<RegisterPage />);

    const confirmInput = screen.getByLabelText(/jelszo megerositese/i);
    expect(confirmInput).toHaveAttribute('type', 'password');

    const toggleButton = confirmInput.parentElement?.querySelector('button');
    expect(toggleButton).toBeTruthy();
    await user.click(toggleButton!);

    expect(confirmInput).toHaveAttribute('type', 'text');
  });

  it('should disable submit button when loading', () => {
    mockIsLoading = true;
    render(<RegisterPage />);

    const submitButton = screen.getByRole('button', { name: /regisztracio/i });
    expect(submitButton).toBeDisabled();
  });

  it('should clear error on form submit', async () => {
    const user = userEvent.setup();
    mockRegister.mockResolvedValue(undefined);
    render(<RegisterPage />);

    await user.type(screen.getByLabelText('Email cim'), 'test@example.com');
    await user.type(screen.getByLabelText('Jelszo'), VALID_PASSWORD);
    await user.type(screen.getByLabelText(/jelszo megerositese/i), VALID_PASSWORD);
    await user.click(screen.getByRole('button', { name: /regisztracio$/i }));

    await waitFor(() => {
      expect(mockClearError).toHaveBeenCalled();
    });
  });

  it('should not call register without user interaction', () => {
    render(<RegisterPage />);
    expect(mockRegister).not.toHaveBeenCalled();
  });
});
