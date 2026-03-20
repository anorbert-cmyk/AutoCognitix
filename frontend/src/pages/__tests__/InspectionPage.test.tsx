import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../test/test-utils';
import InspectionPage from '../InspectionPage';

// =============================================================================
// Mocks
// =============================================================================

const mockMutateAsync = vi.fn();

vi.mock('../../contexts/ToastContext', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
    info: vi.fn(),
    toasts: [],
    addToast: vi.fn(),
    removeToast: vi.fn(),
    clearToasts: vi.fn(),
  }),
}));

vi.mock('../../services/hooks/useInspection', () => ({
  useEvaluateInspection: () => ({
    mutateAsync: mockMutateAsync,
    isPending: false,
  }),
}));

vi.mock('../../components/features/inspection/RiskGauge', () => ({
  default: () => <div data-testid="risk-gauge">RiskGauge</div>,
}));

vi.mock('../../components/features/inspection/InspectionCategoryCard', () => ({
  default: () => <div data-testid="inspection-category-card">Card</div>,
}));

// =============================================================================
// Tests
// =============================================================================

describe('InspectionPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  it('should render the page title', () => {
    render(<InspectionPage />);
    expect(
      screen.getByText('Muszaki vizsga kockazat elemzes'),
    ).toBeInTheDocument();
  });

  it('should render the subtitle badge', () => {
    render(<InspectionPage />);
    expect(
      screen.getByText('Muszaki Vizsga Kockazat'),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Form Fields
  // ---------------------------------------------------------------------------

  it('should render all required form fields', () => {
    render(<InspectionPage />);

    // Vehicle fields
    expect(screen.getByLabelText('Gyarto')).toBeInTheDocument();
    expect(screen.getByLabelText('Modell')).toBeInTheDocument();
    expect(screen.getByLabelText('Evjarat')).toBeInTheDocument();

    // DTC codes
    expect(screen.getByLabelText('DTC hibakodok')).toBeInTheDocument();

    // Mileage
    expect(screen.getByLabelText(/Kilometerorallas/)).toBeInTheDocument();

    // Symptoms (optional)
    expect(screen.getByLabelText(/Eszlelt tunetek/)).toBeInTheDocument();
  });

  it('should render the manufacturer dropdown with placeholder', () => {
    render(<InspectionPage />);
    expect(screen.getByText('Valasszon gyartot')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Submit Button
  // ---------------------------------------------------------------------------

  it('should render the submit button with Hungarian text', () => {
    render(<InspectionPage />);
    expect(
      screen.getByText('Vizsga kockazat elemzese'),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Section Headers
  // ---------------------------------------------------------------------------

  it('should render vehicle data section header', () => {
    render(<InspectionPage />);
    expect(screen.getByText('Jarmu adatok')).toBeInTheDocument();
  });

  it('should render DTC section header', () => {
    render(<InspectionPage />);
    expect(screen.getByText('Hibakodok es allapot')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Footer
  // ---------------------------------------------------------------------------

  it('should render the footer with current year', () => {
    render(<InspectionPage />);
    const currentYear = new Date().getFullYear();
    expect(
      screen.getByText(new RegExp(`${currentYear} MechanicAI`)),
    ).toBeInTheDocument();
  });
});
