import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../test/test-utils';
import HistoryPage from '../HistoryPage';

// Mock the hooks
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

describe('HistoryPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default: API returns no data, so mock/fallback data is used
    mockUseDiagnosisHistory.mockReturnValue({
      data: undefined,
      isLoading: false,
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

  it('should show empty state when no history matches search', () => {
    render(<HistoryPage />);
    const searchInput = screen.getByPlaceholderText(
      'Keresés rendszám, alvázszám vagy tünet alapján...',
    );
    fireEvent.change(searchInput, { target: { value: 'nonexistent-query-xyz' } });
    expect(screen.getByText('Nincs találat')).toBeInTheDocument();
  });

  it('should render history items with vehicle info when data is available (mock fallback)', () => {
    render(<HistoryPage />);
    // Mock fallback data includes these vehicles
    expect(screen.getByText('Toyota Camry')).toBeInTheDocument();
    expect(screen.getByText('Ford F-150')).toBeInTheDocument();
    expect(screen.getByText('Honda Civic')).toBeInTheDocument();
    expect(screen.getByText('Tesla Model 3')).toBeInTheDocument();
    expect(screen.getByText('BMW X5')).toBeInTheDocument();
  });

  it('should show DTC codes for each history item', () => {
    render(<HistoryPage />);
    expect(screen.getByText('P0300')).toBeInTheDocument();
    expect(screen.getByText('P0420')).toBeInTheDocument();
    expect(screen.getByText('C0021')).toBeInTheDocument();
    expect(screen.getByText('B1234')).toBeInTheDocument();
    expect(screen.getByText('P0171')).toBeInTheDocument();
  });

  it('should show license plates for each item', () => {
    render(<HistoryPage />);
    expect(screen.getByText('ABC-1234')).toBeInTheDocument();
    expect(screen.getByText('XYZ-9876')).toBeInTheDocument();
  });

  it('should show status badges for each item', () => {
    render(<HistoryPage />);
    // 3 items have 'fixed' status, 1 'in_progress', 1 'pending'
    const fixedBadges = screen.getAllByText('Javítva');
    expect(fixedBadges).toHaveLength(3);
    expect(screen.getByText('Folyamatban')).toBeInTheDocument();
    expect(screen.getByText('Függőben')).toBeInTheDocument();
  });

  it('should navigate to diagnosis detail when row is clicked', () => {
    render(<HistoryPage />);
    const row = screen.getByText('Toyota Camry').closest('tr');
    expect(row).toBeTruthy();
    fireEvent.click(row!);
    expect(mockNavigate).toHaveBeenCalledWith('/diagnosis/1');
  });

  it('should navigate to diagnosis detail when "Részletek" button is clicked', () => {
    render(<HistoryPage />);
    const detailButtons = screen.getAllByText('Részletek');
    fireEvent.click(detailButtons[0]);
    expect(mockNavigate).toHaveBeenCalledWith('/diagnosis/1');
  });

  it('should display loading state when data is loading', () => {
    mockUseDiagnosisHistory.mockReturnValue({
      data: undefined,
      isLoading: true,
    });
    render(<HistoryPage />);
    expect(screen.getByText('Betöltés...')).toBeInTheDocument();
  });

  it('should render stats cards with mock data', () => {
    render(<HistoryPage />);
    expect(screen.getByText('Megoldási arány')).toBeInTheDocument();
    expect(screen.getByText('94.2%')).toBeInTheDocument();
    expect(screen.getByText('Összes diagnosztika')).toBeInTheDocument();
    expect(screen.getByText('1,248')).toBeInTheDocument();
    expect(screen.getByText('MI pontosság')).toBeInTheDocument();
    expect(screen.getByText('98.8%')).toBeInTheDocument();
  });

  it('should render history items from API when available', () => {
    mockUseDiagnosisHistory.mockReturnValue({
      data: {
        items: [
          {
            id: 'api-1',
            vehicle_make: 'Volkswagen',
            vehicle_model: 'Golf',
            vehicle_year: 2020,
            dtc_codes: ['P0301'],
            created_at: '2026-01-15',
          },
        ],
        total: 1,
      },
      isLoading: false,
    });
    render(<HistoryPage />);
    expect(screen.getByText('Volkswagen Golf')).toBeInTheDocument();
    expect(screen.getByText('P0301')).toBeInTheDocument();
  });

  it('should filter history items based on search query', () => {
    render(<HistoryPage />);
    const searchInput = screen.getByPlaceholderText(
      'Keresés rendszám, alvázszám vagy tünet alapján...',
    );
    fireEvent.change(searchInput, { target: { value: 'Toyota' } });
    expect(screen.getByText('Toyota Camry')).toBeInTheDocument();
    expect(screen.queryByText('Ford F-150')).not.toBeInTheDocument();
  });

  it('should display record count in pagination', () => {
    render(<HistoryPage />);
    expect(screen.getByText(/Megjelenítve:/)).toBeInTheDocument();
  });
});
