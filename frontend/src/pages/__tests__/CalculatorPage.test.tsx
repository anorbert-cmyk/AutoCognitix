import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../test/test-utils';
import CalculatorPage from '../CalculatorPage';

// =============================================================================
// Mocks
// =============================================================================

vi.mock('../../services/hooks/useCalculator', () => ({
  useEvaluateCalculator: () => ({
    mutate: vi.fn(),
    isPending: false,
    isError: false,
    error: null,
    data: null,
  }),
}));

vi.mock('../../components/layouts', () => ({
  PageContainer: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="page-container">{children}</div>
  ),
}));

vi.mock('../../components/lib', () => ({
  Button: ({
    children,
    ...props
  }: React.ButtonHTMLAttributes<HTMLButtonElement> & { children: React.ReactNode }) => (
    <button {...props}>{children}</button>
  ),
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input {...props} />
  ),
  Badge: ({ children }: { children: React.ReactNode }) => (
    <span>{children}</span>
  ),
}));

vi.mock('../../components/features/calculator/RecommendationCard', () => ({
  RecommendationCard: () => <div data-testid="recommendation-card" />,
}));

vi.mock('../../components/features/calculator/ValueComparison', () => ({
  ValueComparison: () => <div data-testid="value-comparison" />,
}));

// =============================================================================
// Tests
// =============================================================================

describe('CalculatorPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  it('should render the page title', () => {
    render(<CalculatorPage />);
    expect(
      screen.getByText('Megeri megjavitani?'),
    ).toBeInTheDocument();
  });

  it('should render the description text', () => {
    render(<CalculatorPage />);
    expect(
      screen.getByText(/Szamolja ki, hogy megeri-e megjavitani/),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Form Fields
  // ---------------------------------------------------------------------------

  it('should render the vehicle form fields', () => {
    render(<CalculatorPage />);

    expect(screen.getByLabelText(/Gyarto/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Modell/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Evjarat/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Kilometerora allas/)).toBeInTheDocument();
  });

  it('should render the fuel type selector', () => {
    render(<CalculatorPage />);
    expect(screen.getByText('Valasszon uzemanyagot')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Condition Radio Buttons
  // ---------------------------------------------------------------------------

  it('should render condition radio buttons with Hungarian labels', () => {
    render(<CalculatorPage />);

    expect(screen.getByText('Kivalo')).toBeInTheDocument();
    expect(screen.getByText('Jo')).toBeInTheDocument();
    expect(screen.getByText('Elfogadhato')).toBeInTheDocument();
    expect(screen.getByText('Rossz')).toBeInTheDocument();
  });

  it('should render condition descriptions', () => {
    render(<CalculatorPage />);

    expect(
      screen.getByText('Szinte uj allapot, minimalis kopas'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Normalis hasznalati nyomok, megbizhato'),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Submit Button
  // ---------------------------------------------------------------------------

  it('should render the submit button with Hungarian text', () => {
    render(<CalculatorPage />);
    expect(
      screen.getByText('Kalkuacio inditasa'),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Section Header
  // ---------------------------------------------------------------------------

  it('should render the vehicle data section header', () => {
    render(<CalculatorPage />);
    expect(screen.getByText('Jarmu adatok')).toBeInTheDocument();
  });

  it('should render the general condition label', () => {
    render(<CalculatorPage />);
    expect(screen.getByText(/Altalanos allapot/)).toBeInTheDocument();
  });
});
