import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../test/test-utils';
import HistoryPage from '../HistoryPage';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

const mockUseDiagnosisHistory = vi.fn();
const mockUseDiagnosisStats = vi.fn();
const mockDelete = vi.fn();

vi.mock('@/services/hooks', () => ({
  useDiagnosisHistory: (...args: unknown[]) => mockUseDiagnosisHistory(...args),
  useDiagnosisStats: (...args: unknown[]) => mockUseDiagnosisStats(...args),
  useDeleteDiagnosis: () => ({ mutateAsync: mockDelete, isPending: false }),
}));

// test-utils has no ToastProvider, so the real useToast would throw. Mock it
// per-file, mirroring the page's relative import path (../contexts/ToastContext).
vi.mock('../../contexts/ToastContext', () => ({
  useToast: () => ({ success: vi.fn(), error: vi.fn() }),
}));

const apiHistoryFixture = {
  items: [
    {
      id: 'api-1',
      vehicle_make: 'Volkswagen',
      vehicle_model: 'Golf',
      vehicle_year: 2020,
      vehicle_vin: 'WVWZZZ1KZAW000001',
      dtc_codes: ['P0301'],
      symptoms_text: 'Rángatás gyorsításkor és alapjáraton.',
      confidence_score: 0.91,
      created_at: '2026-01-15',
    },
    {
      id: 'api-2',
      vehicle_make: 'Skoda',
      vehicle_model: 'Octavia',
      vehicle_year: 2019,
      vehicle_vin: null,
      dtc_codes: ['P0420'],
      symptoms_text: 'Katalizátor hatásfok a küszöb alatt.',
      confidence_score: 0.78,
      created_at: '2026-01-10',
    },
  ],
  total: 2,
  skip: 0,
  limit: 10,
  has_more: false,
};

const historyResult = (data: unknown) => ({
  data,
  isLoading: false,
  error: null,
  refetch: vi.fn(),
});

describe('HistoryPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDelete.mockResolvedValue(undefined);
    mockUseDiagnosisHistory.mockReturnValue(historyResult(undefined));
    mockUseDiagnosisStats.mockReturnValue({
      data: undefined,
      isLoading: false,
    });
  });

  it('should render the page title', () => {
    render(<HistoryPage />);
    expect(screen.getByText('Műhely Előzmények')).toBeInTheDocument();
  });

  it('should render the page description', () => {
    render(<HistoryPage />);
    expect(
      screen.getByText(
        'Diagnosztikai rekordok ellenőrzése és áttekintése MI-alapú pontossággal.',
      ),
    ).toBeInTheDocument();
  });

  it('should show empty CTA when no history exists', () => {
    render(<HistoryPage />);
    expect(screen.getByText('Még nincs diagnosztikai előzmény.')).toBeInTheDocument();
    expect(screen.getByText('Új diagnózis indítása')).toBeInTheDocument();
  });

  it('should display loading state when data is loading', () => {
    mockUseDiagnosisHistory.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    });
    render(<HistoryPage />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('should render error state with retry button on fetch failure', () => {
    const refetch = vi.fn();
    mockUseDiagnosisHistory.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('boom'),
      refetch,
    });
    render(<HistoryPage />);
    expect(
      screen.getByText('Hiba történt az adatok betöltésekor. Próbálja újra!'),
    ).toBeInTheDocument();
    const retry = screen.getByText('Újrapróbálás');
    fireEvent.click(retry);
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  it('should render history items from API when available', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    expect(screen.getByText('Volkswagen Golf')).toBeInTheDocument();
    expect(screen.getByText('Skoda Octavia')).toBeInTheDocument();
    expect(screen.getByText('P0301')).toBeInTheDocument();
    expect(screen.getByText('P0420')).toBeInTheDocument();
  });

  it('should render the real symptoms_text as the main symptom', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    expect(screen.getByText('Rángatás gyorsításkor és alapjáraton.')).toBeInTheDocument();
    expect(screen.getByText('Katalizátor hatásfok a küszöb alatt.')).toBeInTheDocument();
  });

  it('should render the VIN column and fall back to em-dash when VIN is null', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    expect(screen.getByText('WVWZZZ1KZAW000001')).toBeInTheDocument();
    // api-2 has vehicle_vin: null → its row falls back to an em-dash
    const skodaRow = screen.getByText('Skoda Octavia').closest('tr');
    expect(skodaRow).toHaveTextContent('—');
  });

  it('should not render any fabricated status/plate/symptom placeholders', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    expect(screen.queryByText('Javítva')).not.toBeInTheDocument();
    expect(screen.queryByText('Folyamatban')).not.toBeInTheDocument();
    expect(screen.queryByText('Függőben')).not.toBeInTheDocument();
    expect(screen.queryByText('N/A')).not.toBeInTheDocument();
    expect(screen.queryByText('Nincs megadva')).not.toBeInTheDocument();
  });

  it('does not filter until "Szűrők alkalmazása" is clicked', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    fireEvent.change(screen.getByLabelText('Gyártó szűrő'), { target: { value: 'Skoda' } });
    // No client-side filtering: both rows remain rendered.
    expect(screen.getByText('Volkswagen Golf')).toBeInTheDocument();
    expect(screen.getByText('Skoda Octavia')).toBeInTheDocument();
    // The query hook was never invoked with the (unapplied) draft make.
    const calledWithMake = mockUseDiagnosisHistory.mock.calls.some(
      (c) => (c[0] as { vehicleMake?: string })?.vehicleMake === 'Skoda',
    );
    expect(calledWithMake).toBe(false);
  });

  it('sends server-side params when filters are applied', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    fireEvent.change(screen.getByLabelText('Gyártó szűrő'), { target: { value: 'Skoda' } });
    fireEvent.click(screen.getByText('Szűrők alkalmazása'));
    const calls = mockUseDiagnosisHistory.mock.calls;
    const lastParams = calls[calls.length - 1][0];
    expect(lastParams).toEqual(expect.objectContaining({ vehicleMake: 'Skoda', skip: 0 }));
  });

  it('applies dateTo as an inclusive end-of-day timestamp', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    fireEvent.change(screen.getByLabelText('Záró dátum'), { target: { value: '2026-01-31' } });
    fireEvent.click(screen.getByText('Szűrők alkalmazása'));
    const calls = mockUseDiagnosisHistory.mock.calls;
    const lastParams = calls[calls.length - 1][0];
    expect(lastParams).toEqual(
      expect.objectContaining({ dateTo: '2026-01-31T23:59:59', skip: 0 }),
    );
  });

  it('shows the filtered-empty message when active filters match nothing', () => {
    mockUseDiagnosisHistory.mockReturnValue(
      historyResult({ items: [], total: 0, skip: 0, limit: 10, has_more: false }),
    );
    render(<HistoryPage />);
    fireEvent.change(screen.getByLabelText('Gyártó szűrő'), { target: { value: 'NincsIlyen' } });
    fireEvent.click(screen.getByText('Szűrők alkalmazása'));
    expect(screen.getByText('Nincs a szűrőknek megfelelő találat.')).toBeInTheDocument();
  });

  it('disables the Next button when has_more is false', () => {
    mockUseDiagnosisHistory.mockReturnValue(
      historyResult({ ...apiHistoryFixture, has_more: false }),
    );
    render(<HistoryPage />);
    expect(screen.getByLabelText('Következő oldal')).toBeDisabled();
  });

  it('enables the Next button when has_more is true', () => {
    mockUseDiagnosisHistory.mockReturnValue(
      historyResult({ ...apiHistoryFixture, has_more: true }),
    );
    render(<HistoryPage />);
    expect(screen.getByLabelText('Következő oldal')).not.toBeDisabled();
  });

  it('deletes a diagnosis after confirmation', async () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    fireEvent.click(screen.getByLabelText('Törlés: Volkswagen Golf'));
    expect(confirmSpy).toHaveBeenCalled();
    expect(mockDelete).toHaveBeenCalledWith('api-1');
    await Promise.resolve();
    confirmSpy.mockRestore();
  });

  it('does not delete when confirmation is cancelled', () => {
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    fireEvent.click(screen.getByLabelText('Törlés: Volkswagen Golf'));
    expect(confirmSpy).toHaveBeenCalled();
    expect(mockDelete).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it('should navigate to diagnosis detail when row is clicked', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    const row = screen.getByText('Volkswagen Golf').closest('tr');
    expect(row).toBeTruthy();
    fireEvent.click(row!);
    expect(mockNavigate).toHaveBeenCalledWith('/diagnosis/api-1');
  });

  it('should navigate via "Részletek" button click', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    const detailButtons = screen.getAllByText('Részletek');
    fireEvent.click(detailButtons[0]);
    expect(mockNavigate).toHaveBeenCalledWith('/diagnosis/api-1');
  });

  it('should render stats placeholder when API stats not available', () => {
    render(<HistoryPage />);
    expect(screen.getByText('Összes diagnosztika')).toBeInTheDocument();
    expect(screen.getByText('Átlagos AI konfidencia')).toBeInTheDocument();
    // No stats → both cards show em-dash
    expect(screen.getAllByText('—').length).toBeGreaterThanOrEqual(2);
  });

  it('should render stats from API when available', () => {
    mockUseDiagnosisStats.mockReturnValue({
      data: { total_diagnoses: 42, avg_confidence: 0.873 },
      isLoading: false,
    });
    render(<HistoryPage />);
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('87.3%')).toBeInTheDocument();
  });

  it('should display record count in pagination footer', () => {
    mockUseDiagnosisHistory.mockReturnValue(historyResult(apiHistoryFixture));
    render(<HistoryPage />);
    expect(screen.getByText(/Megjelenítve:/)).toBeInTheDocument();
  });
});
