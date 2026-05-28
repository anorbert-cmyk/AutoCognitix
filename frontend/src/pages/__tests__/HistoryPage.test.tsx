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

vi.mock('@/services/hooks', () => ({
  useDiagnosisHistory: (...args: unknown[]) => mockUseDiagnosisHistory(...args),
  useDiagnosisStats: (...args: unknown[]) => mockUseDiagnosisStats(...args),
}));

const apiHistoryFixture = {
  items: [
    {
      id: 'api-1',
      vehicle_make: 'Volkswagen',
      vehicle_model: 'Golf',
      vehicle_year: 2020,
      dtc_codes: ['P0301'],
      created_at: '2026-01-15',
    },
    {
      id: 'api-2',
      vehicle_make: 'Skoda',
      vehicle_model: 'Octavia',
      vehicle_year: 2019,
      dtc_codes: ['P0420'],
      created_at: '2026-01-10',
    },
  ],
  total: 2,
};

describe('HistoryPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseDiagnosisHistory.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
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
    expect(screen.getByText('Betöltés...')).toBeInTheDocument();
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
    mockUseDiagnosisHistory.mockReturnValue({
      data: apiHistoryFixture,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<HistoryPage />);
    expect(screen.getByText('Volkswagen Golf')).toBeInTheDocument();
    expect(screen.getByText('Skoda Octavia')).toBeInTheDocument();
    expect(screen.getByText('P0301')).toBeInTheDocument();
    expect(screen.getByText('P0420')).toBeInTheDocument();
  });

  it('should filter API items based on search query', () => {
    mockUseDiagnosisHistory.mockReturnValue({
      data: apiHistoryFixture,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<HistoryPage />);
    const searchInput = screen.getByPlaceholderText(
      'Keresés rendszám, alvázszám vagy tünet alapján...',
    );
    fireEvent.change(searchInput, { target: { value: 'Skoda' } });
    expect(screen.getByText('Skoda Octavia')).toBeInTheDocument();
    expect(screen.queryByText('Volkswagen Golf')).not.toBeInTheDocument();
  });

  it('should show no-match empty state when search yields no results', () => {
    mockUseDiagnosisHistory.mockReturnValue({
      data: apiHistoryFixture,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<HistoryPage />);
    const searchInput = screen.getByPlaceholderText(
      'Keresés rendszám, alvázszám vagy tünet alapján...',
    );
    fireEvent.change(searchInput, { target: { value: 'nonexistent-query-xyz' } });
    expect(
      screen.getByText('A keresés nem hozott eredményt. Próbáljon más szűrőfeltételeket!'),
    ).toBeInTheDocument();
  });

  it('should navigate to diagnosis detail when row is clicked', () => {
    mockUseDiagnosisHistory.mockReturnValue({
      data: apiHistoryFixture,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<HistoryPage />);
    const row = screen.getByText('Volkswagen Golf').closest('tr');
    expect(row).toBeTruthy();
    fireEvent.click(row!);
    expect(mockNavigate).toHaveBeenCalledWith('/diagnosis/api-1');
  });

  it('should navigate via "Részletek" button click', () => {
    mockUseDiagnosisHistory.mockReturnValue({
      data: apiHistoryFixture,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
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
    mockUseDiagnosisHistory.mockReturnValue({
      data: apiHistoryFixture,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    });
    render(<HistoryPage />);
    expect(screen.getByText(/Megjelenítve:/)).toBeInTheDocument();
  });
});
