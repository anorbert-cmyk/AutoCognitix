import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../../test/test-utils';
import userEvent from '@testing-library/user-event';
import DiagnosisPage from '../DiagnosisPage';

// =============================================================================
// Mocks
// =============================================================================

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const mockMutateAsync = vi.fn();
const mockToastError = vi.fn();
const mockToastInfo = vi.fn();

vi.mock('../../contexts/ToastContext', () => ({
  useToast: () => ({
    success: vi.fn(),
    error: mockToastError,
    warning: vi.fn(),
    info: mockToastInfo,
    toasts: [],
    addToast: vi.fn(),
    removeToast: vi.fn(),
    clearToasts: vi.fn(),
  }),
}));

vi.mock('../../services/hooks', () => ({
  useAnalyzeDiagnosis: () => ({
    mutateAsync: mockMutateAsync,
    isPending: false,
  }),
  useDiagnosisHistory: () => ({
    data: {
      items: [
        {
          id: 'diag-1',
          vehicle_make: 'BMW',
          vehicle_model: 'X5',
          vehicle_year: 2020,
          dtc_codes: ['P0171'],
          confidence_score: 0.85,
          created_at: '2026-03-01T10:00:00Z',
        },
        {
          id: 'diag-2',
          vehicle_make: 'Audi',
          vehicle_model: 'A4',
          vehicle_year: 2019,
          dtc_codes: ['P0300'],
          confidence_score: 0.92,
          created_at: '2026-02-28T14:00:00Z',
        },
      ],
      total: 2,
      skip: 0,
      limit: 2,
    },
    isLoading: false,
    error: null,
  }),
}));

vi.mock('../../components/features/diagnosis/AnalysisProgress', () => ({
  AnalysisProgress: ({
    onCancel,
    onComplete,
    vehicleInfo,
  }: {
    diagnosisId?: string;
    onCancel?: () => void;
    onComplete?: (result?: { id?: string }) => void;
    vehicleInfo?: { make: string; model: string; year: number; dtcCode: string };
  }) => (
    <div data-testid="analysis-progress">
      <span>Elemzes folyamatban...</span>
      {vehicleInfo && (
        <span data-testid="vehicle-info">
          {vehicleInfo.make} {vehicleInfo.model} {vehicleInfo.year} {vehicleInfo.dtcCode}
        </span>
      )}
      <button onClick={onCancel} data-testid="cancel-analysis">
        Megse
      </button>
      <button onClick={() => onComplete?.({ id: 'stream-result-123' })} data-testid="complete-analysis">
        Kesz
      </button>
    </div>
  ),
}));

// =============================================================================
// Tests
// =============================================================================

describe('DiagnosisPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  it('should render the initial input form', () => {
    render(<DiagnosisPage />);
    expect(
      screen.getByText('Új diagnosztikai folyamat'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('AI Diagnosztikai Eszköz'),
    ).toBeInTheDocument();
  });

  it('should render all required input fields', () => {
    render(<DiagnosisPage />);

    // DTC code field
    expect(screen.getByLabelText(/Elsődleges hibakód/i)).toBeInTheDocument();

    // Vehicle fields
    expect(screen.getByLabelText('Gyártó')).toBeInTheDocument();
    expect(screen.getByLabelText('Modell')).toBeInTheDocument();
    expect(screen.getByLabelText('Évjárat')).toBeInTheDocument();

    // Text areas
    expect(screen.getByLabelText(/Tulajdonos panaszai/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Szerelői jegyzetek/i)).toBeInTheDocument();
  });

  it('should render the DTC code placeholder with example', () => {
    render(<DiagnosisPage />);
    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    expect(dtcInput).toBeInTheDocument();
  });

  it('should render the submit button with Hungarian text', () => {
    render(<DiagnosisPage />);
    expect(
      screen.getByText('AI Megoldás Generálása'),
    ).toBeInTheDocument();
  });

  it('should render the draft save button', () => {
    render(<DiagnosisPage />);
    expect(
      screen.getByText('Piszkozat mentése'),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Wizard Steps Display
  // ---------------------------------------------------------------------------

  it('should display wizard step indicators', () => {
    render(<DiagnosisPage />);
    expect(screen.getByText('Adatbevitel')).toBeInTheDocument();
    expect(screen.getByText('Elemzés')).toBeInTheDocument();
    expect(screen.getByText('Jelentés')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Manufacturer Select
  // ---------------------------------------------------------------------------

  it('should render manufacturer dropdown with options', () => {
    render(<DiagnosisPage />);
    const select = screen.getByLabelText('Gyártó') as HTMLSelectElement;
    expect(select).toBeInTheDocument();

    // Check default placeholder option
    expect(screen.getByText('Válasszon gyártót')).toBeInTheDocument();

    // Check some manufacturers are in the list
    expect(screen.getByText('Toyota')).toBeInTheDocument();
    expect(screen.getByText('Volkswagen')).toBeInTheDocument();
    expect(screen.getByText('BMW')).toBeInTheDocument();
    expect(screen.getByText('Škoda')).toBeInTheDocument();
  });

  it('should update manufacturer on selection', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    const select = screen.getByLabelText('Gyártó') as HTMLSelectElement;
    await user.selectOptions(select, 'Volkswagen');
    expect(select.value).toBe('Volkswagen');
  });

  // ---------------------------------------------------------------------------
  // Year Field
  // ---------------------------------------------------------------------------

  it('should have year field with correct min/max attributes', () => {
    render(<DiagnosisPage />);
    const yearInput = screen.getByLabelText('Évjárat') as HTMLInputElement;
    expect(yearInput).toHaveAttribute('min', '1990');
    const currentYear = new Date().getFullYear();
    expect(yearInput).toHaveAttribute('max', String(currentYear + 1));
  });

  it('should allow typing a year value', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    const yearInput = screen.getByLabelText('Évjárat') as HTMLInputElement;
    await user.type(yearInput, '2020');
    expect(yearInput.value).toBe('2020');
  });

  // ---------------------------------------------------------------------------
  // DTC Code Input
  // ---------------------------------------------------------------------------

  it('should convert DTC code to uppercase', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    const dtcInput = screen.getByPlaceholderText('pl. P0300') as HTMLInputElement;
    await user.type(dtcInput, 'p0300');
    expect(dtcInput.value).toBe('P0300');
  });

  it('should accept valid DTC code formats (P/C/B/U + 4 digits)', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    const dtcInput = screen.getByPlaceholderText('pl. P0300') as HTMLInputElement;

    // Test P-code (powertrain)
    await user.clear(dtcInput);
    await user.type(dtcInput, 'P0420');
    expect(dtcInput.value).toBe('P0420');

    // Test C-code (chassis)
    await user.clear(dtcInput);
    await user.type(dtcInput, 'c1234');
    expect(dtcInput.value).toBe('C1234');

    // Test B-code (body)
    await user.clear(dtcInput);
    await user.type(dtcInput, 'b0100');
    expect(dtcInput.value).toBe('B0100');

    // Test U-code (network)
    await user.clear(dtcInput);
    await user.type(dtcInput, 'u0401');
    expect(dtcInput.value).toBe('U0401');
  });

  // ---------------------------------------------------------------------------
  // Form Validation
  // ---------------------------------------------------------------------------

  it('should show error toast when submitting without DTC code', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    expect(mockToastError).toHaveBeenCalledWith('Hibakód megadása kötelező');
    expect(mockMutateAsync).not.toHaveBeenCalled();
  });

  it('should show error toast when DTC code is only whitespace', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    await user.type(dtcInput, '   ');

    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    expect(mockToastError).toHaveBeenCalledWith('Hibakód megadása kötelező');
  });

  // ---------------------------------------------------------------------------
  // Form Submission - Success
  // ---------------------------------------------------------------------------

  it('should submit form with valid data and transition to analysis step', async () => {
    const user = userEvent.setup();

    render(<DiagnosisPage />);

    // Fill in form fields
    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    await user.type(dtcInput, 'P0300');

    const makeSelect = screen.getByLabelText('Gyártó');
    await user.selectOptions(makeSelect, 'Volkswagen');

    const modelInput = screen.getByLabelText('Modell');
    await user.type(modelInput, 'Golf');

    const yearInput = screen.getByLabelText('Évjárat');
    await user.type(yearInput, '2018');

    const complaintsInput = screen.getByLabelText(/Tulajdonos panaszai/i);
    await user.type(complaintsInput, 'Egyenetlen alapjárat');

    const notesInput = screen.getByLabelText(/Szerelői jegyzetek/i);
    await user.type(notesInput, 'Gyújtógyertyák ellenőrizve');

    // Submit the form
    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    // Should transition to analysis view (streaming mode: form submit transitions to analysis step)
    await waitFor(() => {
      expect(screen.getByTestId('analysis-progress')).toBeInTheDocument();
    });
  });

  it('should use defaults when optional fields are empty', async () => {
    const user = userEvent.setup();

    render(<DiagnosisPage />);

    // Only fill the required DTC code
    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    await user.type(dtcInput, 'P0171');

    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    // Should transition to analysis view
    await waitFor(() => {
      expect(screen.getByTestId('analysis-progress')).toBeInTheDocument();
    });
  });

  // ---------------------------------------------------------------------------
  // Form Submission - Error Handling
  // ---------------------------------------------------------------------------

  it('should transition to analysis step even when streaming will handle errors', async () => {
    const user = userEvent.setup();

    render(<DiagnosisPage />);

    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    await user.type(dtcInput, 'P0300');

    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    // In streaming mode, submit transitions to analysis step
    // Errors are handled by AnalysisProgress + fallback handler
    await waitFor(() => {
      expect(screen.getByTestId('analysis-progress')).toBeInTheDocument();
    });
  });

  // ---------------------------------------------------------------------------
  // Analysis Step - Navigation & Cancel
  // ---------------------------------------------------------------------------

  it('should navigate to result page when analysis completes', async () => {
    const user = userEvent.setup();

    render(<DiagnosisPage />);

    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    await user.type(dtcInput, 'P0300');

    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    // Wait for analysis step
    await waitFor(() => {
      expect(screen.getByTestId('analysis-progress')).toBeInTheDocument();
    });

    // Click the complete button (from our mocked AnalysisProgress)
    const completeButton = screen.getByTestId('complete-analysis');
    await user.click(completeButton);

    // In streaming mode, navigation is triggered by onComplete callback
    expect(mockNavigate).toHaveBeenCalled();
  });

  it('should return to input step when analysis is cancelled', async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockResolvedValueOnce({ id: 'result-cancel' });

    render(<DiagnosisPage />);

    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    await user.type(dtcInput, 'P0300');

    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByTestId('analysis-progress')).toBeInTheDocument();
    });

    // Cancel the analysis
    const cancelButton = screen.getByTestId('cancel-analysis');
    await user.click(cancelButton);

    // Should return to input form
    expect(screen.getByText('Új diagnosztikai folyamat')).toBeInTheDocument();
  });

  it('should pass vehicle info to AnalysisProgress when make is provided', async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockResolvedValueOnce({ id: 'result-vehicle' });

    render(<DiagnosisPage />);

    const dtcInput = screen.getByPlaceholderText('pl. P0300');
    await user.type(dtcInput, 'P0300');

    const makeSelect = screen.getByLabelText('Gyártó');
    await user.selectOptions(makeSelect, 'BMW');

    const modelInput = screen.getByLabelText('Modell');
    await user.type(modelInput, 'X3');

    const yearInput = screen.getByLabelText('Évjárat');
    await user.type(yearInput, '2021');

    const submitButton = screen.getByText('AI Megoldás Generálása');
    await user.click(submitButton);

    await waitFor(() => {
      expect(screen.getByTestId('vehicle-info')).toHaveTextContent(
        'BMW X3 2021 P0300',
      );
    });
  });

  // ---------------------------------------------------------------------------
  // Recent History Cards
  // ---------------------------------------------------------------------------

  it('should render recent history cards with data from API', () => {
    render(<DiagnosisPage />);

    // First history card - uses "Legutóbbi:" prefix with vehicle data
    expect(screen.getByText(/Legutóbbi:.*BMW.*X5/)).toBeInTheDocument();
    expect(screen.getByText('P0171')).toBeInTheDocument();

    // Second history card
    expect(screen.getByText(/Legutóbbi:.*Audi.*A4/)).toBeInTheDocument();
    expect(screen.getByText('P0300')).toBeInTheDocument();
  });

  it('should render report view buttons on history cards', () => {
    render(<DiagnosisPage />);
    const viewButtons = screen.getAllByText('Jelentés megtekintése');
    expect(viewButtons).toHaveLength(2);
  });

  it('should navigate to diagnosis detail when clicking a history card', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    // Find the first history card by looking for the "Legutóbbi: BMW X5" heading
    const bmwText = screen.getByText(/Legutóbbi:.*BMW.*X5/);
    const card = bmwText.closest('[class*="cursor-pointer"]');
    expect(card).not.toBeNull();

    await user.click(card!);
    expect(mockNavigate).toHaveBeenCalledWith('/diagnosis/diag-1');
  });

  // ---------------------------------------------------------------------------
  // Help Card
  // ---------------------------------------------------------------------------

  it('should render the help card', () => {
    render(<DiagnosisPage />);
    expect(
      screen.getByText('Segítségre van szüksége egy kóddal?'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('DTC adatbázis böngészése'),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Speech Recognition (Diktálás)
  // ---------------------------------------------------------------------------

  it('should render the dictation button', () => {
    render(<DiagnosisPage />);
    expect(screen.getByText('Diktálás')).toBeInTheDocument();
  });

  it('should show error toast when speech recognition is not supported', async () => {
    const user = userEvent.setup();
    render(<DiagnosisPage />);

    const dictationButton = screen.getByText('Diktálás');
    await user.click(dictationButton);

    expect(mockToastError).toHaveBeenCalledWith(
      'A böngésző nem támogatja a beszédfelismerést',
    );
  });

  it('should start speech recognition when supported', async () => {
    const user = userEvent.setup();
    const mockStart = vi.fn();
    const MockSpeechRecognition = vi.fn().mockImplementation(() => ({
      lang: '',
      continuous: false,
      interimResults: false,
      onresult: null,
      onerror: null,
      start: mockStart,
    }));

    // Add webkitSpeechRecognition to window
    Object.defineProperty(window, 'webkitSpeechRecognition', {
      value: MockSpeechRecognition,
      writable: true,
      configurable: true,
    });

    render(<DiagnosisPage />);

    const dictationButton = screen.getByText('Diktálás');
    await user.click(dictationButton);

    expect(mockStart).toHaveBeenCalled();
    expect(mockToastInfo).toHaveBeenCalledWith('Beszéljen most...');

    // Cleanup
    // @ts-expect-error cleaning up test property
    delete window.webkitSpeechRecognition;
  });

  // ---------------------------------------------------------------------------
  // Footer
  // ---------------------------------------------------------------------------

  it('should render the footer with current year', () => {
    render(<DiagnosisPage />);
    const currentYear = new Date().getFullYear();
    expect(
      screen.getByText(new RegExp(`${currentYear} MechanicAI`)),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Description Text
  // ---------------------------------------------------------------------------

  it('should render the description paragraph', () => {
    render(<DiagnosisPage />);
    expect(
      screen.getByText(/Adja meg a jármű adatait és az OBD hibakódokat/),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Auto-save indicator
  // ---------------------------------------------------------------------------

  it('should display the auto-save indicator', () => {
    render(<DiagnosisPage />);
    expect(
      screen.getByText('Automatikusan mentve 2 perce'),
    ).toBeInTheDocument();
  });
});
