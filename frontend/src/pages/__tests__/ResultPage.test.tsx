import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../test/test-utils';
import ResultPage from '../ResultPage';
import { DiagnosisResponse } from '../../services/api';

// ─── Mocks ──────────────────────────────────────────────────────────────────

// Mock react-router-dom (partially, preserve Link/BrowserRouter from test-utils)
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return {
    ...actual,
    useParams: vi.fn(() => ({ id: 'test-diagnosis-id' })),
    useNavigate: () => mockNavigate,
  };
});

// Mock useDiagnosisDetail hook
const mockUseDiagnosisDetail = vi.fn();
vi.mock('../../services/hooks', async () => {
  const actual = await vi.importActual<typeof import('../../services/hooks')>('../../services/hooks');
  return {
    ...actual,
    useDiagnosisDetail: (...args: unknown[]) => mockUseDiagnosisDetail(...args),
  };
});

// Mock ToastContext
const mockToastInfo = vi.fn();
vi.mock('../../contexts/ToastContext', () => ({
  useToast: () => ({
    info: mockToastInfo,
    success: vi.fn(),
    error: vi.fn(),
    warning: vi.fn(),
  }),
}));

// Mock child components that are complex / use Material Symbols
vi.mock('../../components/ui/MaterialIcon', () => ({
  MaterialIcon: ({ name, className }: { name: string; className?: string }) => (
    <span data-testid={`material-icon-${name}`} className={className}>{name}</span>
  ),
}));

vi.mock('../../components/features/diagnosis/DiagnosticConfidence', () => ({
  DiagnosticConfidence: ({ percentage }: { percentage: number }) => (
    <div data-testid="diagnostic-confidence">{percentage}%</div>
  ),
}));

vi.mock('../../components/features/diagnosis/RepairStep', () => ({
  RepairStep: ({ number, title, description }: { number: number; title: string; description: string }) => (
    <div data-testid={`repair-step-${number}`}>
      <span>{title}</span>
      <span>{description}</span>
    </div>
  ),
}));

// ─── Test Data ───────────────────────────────────────────────────────────────

function createMockResult(overrides?: Partial<DiagnosisResponse>): DiagnosisResponse {
  return {
    id: 'diag-abc-4829',
    vehicle_make: 'Volkswagen',
    vehicle_model: 'Golf',
    vehicle_year: 2018,
    dtc_codes: ['P0300', 'P0301', 'P0304'],
    symptoms: 'A motor rázkódik gyorsításkor, különösen hideg indítás után.',
    probable_causes: [
      {
        title: 'Több hengeres égéskimaradás',
        description: 'A gyújtótekercsek vagy gyújtógyertyák elhasználódása okozza.',
        confidence: 0.85,
        related_dtc_codes: ['P0300', 'P0301'],
        components: ['ignition_coil', 'spark_plug'],
      },
      {
        title: 'Üzemanyag rendszer hiba',
        description: 'Eltömődött befecskendezők.',
        confidence: 0.6,
        related_dtc_codes: ['P0300'],
        components: ['fuel_injector'],
      },
    ],
    recommended_repairs: [
      {
        title: 'Gyújtógyertya csere',
        description: 'Cserélje ki az összes gyújtógyertyát.',
        estimated_cost_min: 8000,
        estimated_cost_max: 15000,
        estimated_cost_currency: 'HUF',
        difficulty: 'beginner',
        parts_needed: ['Gyújtógyertya NGK BKUR6ET-10'],
        estimated_time_minutes: 45,
        tools_needed: [
          { name: 'Gyújtógyertya kulcs', icon_hint: 'build' },
          { name: 'Nyomatékkulcs', icon_hint: 'precision_manufacturing' },
        ],
        expert_tips: ['Mindig cserélje az összes gyertyát egyszerre.'],
        root_cause_explanation: 'Elhasználódott elektródák miatt gyenge szikra.',
      },
      {
        title: 'Gyújtótekercs ellenőrzés',
        description: 'Ellenőrizze a gyújtótekercsek ellenállását multiméterrel.',
        estimated_cost_currency: 'HUF',
        difficulty: 'intermediate',
        parts_needed: ['Gyújtótekercs'],
        tools_needed: [],
        expert_tips: [],
      },
    ],
    confidence_score: 0.87,
    sources: [
      { type: 'database', title: 'NHTSA Complaints', relevance_score: 0.9 },
    ],
    created_at: '2026-03-15T10:30:00Z',
    parts_with_prices: [
      {
        id: 'part-1',
        name: 'Gyújtógyertya',
        name_en: 'Spark Plug',
        category: 'Gyújtás',
        price_range_min: 2500,
        price_range_max: 5500,
        labor_hours: 0.5,
        currency: 'HUF',
      },
      {
        id: 'part-2',
        name: 'Gyújtótekercs',
        name_en: 'Ignition Coil',
        category: 'Gyújtás',
        price_range_min: 12000,
        price_range_max: 28000,
        labor_hours: 0.75,
        currency: 'HUF',
      },
    ],
    total_cost_estimate: {
      parts_min: 14500,
      parts_max: 33500,
      labor_min: 8000,
      labor_max: 15000,
      total_min: 22500,
      total_max: 48500,
      currency: 'HUF',
      estimated_hours: 1.25,
      difficulty: 'medium',
      disclaimer: 'Az árak tájékoztató jellegűek, a tényleges költség eltérhet.',
    },
    root_cause_analysis: 'A főtengely pozíció szenzor adatai alapján a 3. henger szikra időzítése eltér a normálistól.',
    ...overrides,
  };
}

function setupHookReturn(overrides: {
  data?: DiagnosisResponse | null;
  isLoading?: boolean;
  error?: Error | null;
}) {
  const refetch = vi.fn();
  mockUseDiagnosisDetail.mockReturnValue({
    data: overrides.data ?? null,
    isLoading: overrides.isLoading ?? false,
    error: overrides.error ?? null,
    refetch,
  });
  return { refetch };
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('ResultPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Default: mock window.print
    vi.spyOn(window, 'print').mockImplementation(() => {});
  });

  // ── Loading state ──────────────────────────────────────────────────────

  it('shows loading spinner while data is being fetched', () => {
    setupHookReturn({ isLoading: true });
    render(<ResultPage />);
    expect(screen.getByText('Diagnózis betöltése...')).toBeInTheDocument();
  });

  // ── Error state ────────────────────────────────────────────────────────

  it('shows error state when fetching fails', () => {
    setupHookReturn({ error: new Error('Network error') });
    render(<ResultPage />);
    expect(screen.getByText('A diagnózis nem található')).toBeInTheDocument();
    expect(screen.getByText('Újrapróbálás')).toBeInTheDocument();
    expect(screen.getByText('Új diagnózis készítése')).toBeInTheDocument();
  });

  it('calls refetch when retry button is clicked', () => {
    const { refetch } = setupHookReturn({ error: new Error('fail') });
    render(<ResultPage />);
    fireEvent.click(screen.getByText('Újrapróbálás'));
    expect(refetch).toHaveBeenCalledOnce();
  });

  it('navigates to /diagnosis when "new diagnosis" button is clicked in error state', () => {
    setupHookReturn({ error: new Error('fail') });
    render(<ResultPage />);
    fireEvent.click(screen.getByText('Új diagnózis készítése'));
    expect(mockNavigate).toHaveBeenCalledWith('/diagnosis');
  });

  // ── No data state ─────────────────────────────────────────────────────

  it('renders nothing when result is null (no data)', () => {
    setupHookReturn({ data: null });
    const { container } = render(<ResultPage />);
    // The component returns null when !result
    expect(container.innerHTML).toBe('');
  });

  // ── Valid result rendering ─────────────────────────────────────────────

  describe('with valid diagnosis result', () => {
    beforeEach(() => {
      setupHookReturn({ data: createMockResult() });
    });

    it('renders the page title', () => {
      render(<ResultPage />);
      expect(screen.getByText('Javítási javaslat')).toBeInTheDocument();
    });

    it('displays vehicle make, model and year', () => {
      render(<ResultPage />);
      expect(screen.getByText('Volkswagen Golf')).toBeInTheDocument();
      // Year appears as part of a composite text
      expect(screen.getByText(/2018/)).toBeInTheDocument();
    });

    it('displays the primary DTC code', () => {
      render(<ResultPage />);
      expect(screen.getByText('P0300')).toBeInTheDocument();
    });

    it('displays DTC description from probable causes', () => {
      render(<ResultPage />);
      expect(screen.getByText('Több hengeres égéskimaradás')).toBeInTheDocument();
    });

    it('shows customer complaint / symptoms', () => {
      render(<ResultPage />);
      expect(screen.getByText('A motor rázkódik gyorsításkor, különösen hideg indítás után.')).toBeInTheDocument();
    });

    it('renders AI analysis section with root cause text', () => {
      render(<ResultPage />);
      expect(screen.getByText('AI Diagnosztikai Elemzés')).toBeInTheDocument();
      expect(screen.getByText(/főtengely pozíció szenzor/)).toBeInTheDocument();
    });

    it('shows confidence score percentage via DiagnosticConfidence component', () => {
      render(<ResultPage />);
      // confidence_score=0.87 → 87%
      expect(screen.getByTestId('diagnostic-confidence')).toHaveTextContent('87%');
    });

    it('renders repair steps', () => {
      render(<ResultPage />);
      expect(screen.getByTestId('repair-step-1')).toBeInTheDocument();
      expect(screen.getByTestId('repair-step-2')).toBeInTheDocument();
      expect(screen.getByText('Gyújtógyertya csere')).toBeInTheDocument();
      expect(screen.getByText('Gyújtótekercs ellenőrzés')).toBeInTheDocument();
    });

    it('shows repair steps count', () => {
      render(<ResultPage />);
      expect(screen.getByText('2 lépés')).toBeInTheDocument();
    });

    it('renders AI disclaimer', () => {
      render(<ResultPage />);
      expect(screen.getByText(/AI Diagnosztikai Eszköz/)).toBeInTheDocument();
    });
  });

  // ── Parts pricing table ────────────────────────────────────────────────

  describe('parts pricing', () => {
    it('renders parts table when parts_with_prices is present', () => {
      setupHookReturn({ data: createMockResult() });
      render(<ResultPage />);
      expect(screen.getByText('Alkatrészek és Árak')).toBeInTheDocument();
      expect(screen.getByText('2 alkatrész')).toBeInTheDocument();
      expect(screen.getByText('Gyújtógyertya')).toBeInTheDocument();
      expect(screen.getByText('Spark Plug')).toBeInTheDocument();
      expect(screen.getByText('Gyújtótekercs')).toBeInTheDocument();
      expect(screen.getByText('Ignition Coil')).toBeInTheDocument();
    });

    it('renders part categories', () => {
      setupHookReturn({ data: createMockResult() });
      render(<ResultPage />);
      const categoryBadges = screen.getAllByText('Gyújtás');
      expect(categoryBadges.length).toBe(2);
    });

    it('renders labor hours for parts', () => {
      setupHookReturn({ data: createMockResult() });
      render(<ResultPage />);
      expect(screen.getByText('0.5 óra')).toBeInTheDocument();
      expect(screen.getByText('0.75 óra')).toBeInTheDocument();
    });

    it('does not render parts section when parts_with_prices is empty', () => {
      setupHookReturn({ data: createMockResult({ parts_with_prices: [] }) });
      render(<ResultPage />);
      expect(screen.queryByText('Alkatrészek és Árak')).not.toBeInTheDocument();
    });
  });

  // ── Cost estimate section ──────────────────────────────────────────────

  describe('cost estimate', () => {
    it('renders total cost estimate card', () => {
      setupHookReturn({ data: createMockResult() });
      render(<ResultPage />);
      expect(screen.getByText('Becsült Javítási Költség')).toBeInTheDocument();
      expect(screen.getByText('Összesen')).toBeInTheDocument();
    });

    it('shows estimated hours', () => {
      setupHookReturn({ data: createMockResult() });
      render(<ResultPage />);
      expect(screen.getByText('1.25 óra')).toBeInTheDocument();
    });

    it('shows difficulty badge', () => {
      setupHookReturn({ data: createMockResult() });
      render(<ResultPage />);
      expect(screen.getByText('Közepes')).toBeInTheDocument();
    });

    it('shows disclaimer text', () => {
      setupHookReturn({ data: createMockResult() });
      render(<ResultPage />);
      expect(screen.getByText('Az árak tájékoztató jellegűek, a tényleges költség eltérhet.')).toBeInTheDocument();
    });

    it('does not render cost estimate when total_cost_estimate is undefined', () => {
      setupHookReturn({ data: createMockResult({ total_cost_estimate: undefined }) });
      render(<ResultPage />);
      expect(screen.queryByText('Becsült Javítási Költség')).not.toBeInTheDocument();
    });

    it('maps difficulty levels correctly', () => {
      // easy
      setupHookReturn({
        data: createMockResult({
          total_cost_estimate: {
            ...createMockResult().total_cost_estimate!,
            difficulty: 'easy',
          },
        }),
      });
      const { unmount } = render(<ResultPage />);
      expect(screen.getByText('Könnyű')).toBeInTheDocument();
      unmount();

      // hard
      setupHookReturn({
        data: createMockResult({
          total_cost_estimate: {
            ...createMockResult().total_cost_estimate!,
            difficulty: 'hard',
          },
        }),
      });
      const { unmount: unmount2 } = render(<ResultPage />);
      expect(screen.getByText('Nehéz')).toBeInTheDocument();
      unmount2();

      // expert (fallback)
      setupHookReturn({
        data: createMockResult({
          total_cost_estimate: {
            ...createMockResult().total_cost_estimate!,
            difficulty: 'expert',
          },
        }),
      });
      render(<ResultPage />);
      expect(screen.getByText('Szakértő')).toBeInTheDocument();
    });
  });

  // ── PDF and print buttons ──────────────────────────────────────────────

  describe('PDF save and print worksheet', () => {
    beforeEach(() => {
      setupHookReturn({ data: createMockResult() });
    });

    it('calls window.print when PDF save button is clicked', () => {
      render(<ResultPage />);
      fireEvent.click(screen.getByText('Nyomtatás / PDF'));
      expect(window.print).toHaveBeenCalledOnce();
      expect(mockToastInfo).toHaveBeenCalledWith('PDF mentés elindítva...');
    });

    it('calls window.print when print worksheet button is clicked', () => {
      render(<ResultPage />);
      fireEvent.click(screen.getByText('Munkalap nyomtatása'));
      expect(window.print).toHaveBeenCalledOnce();
      expect(mockToastInfo).toHaveBeenCalledWith('Munkalap nyomtatása...');
    });
  });

  // ── Edge cases: missing/undefined arrays ───────────────────────────────

  describe('handles missing/undefined data safely', () => {
    it('handles missing dtc_codes array gracefully (uses fallback P0303)', () => {
      setupHookReturn({
        data: createMockResult({ dtc_codes: undefined as unknown as string[] }),
      });
      render(<ResultPage />);
      // With dtc_codes undefined, the optional chaining ?.[0] returns undefined,
      // so it falls back to 'P0303'
      expect(screen.getByText('P0303')).toBeInTheDocument();
    });

    it('handles empty dtc_codes array (uses fallback P0303)', () => {
      setupHookReturn({ data: createMockResult({ dtc_codes: [] }) });
      render(<ResultPage />);
      expect(screen.getByText('P0303')).toBeInTheDocument();
    });

    it('handles empty recommended_repairs (shows no-repairs message)', () => {
      setupHookReturn({ data: createMockResult({ recommended_repairs: [] }) });
      render(<ResultPage />);
      expect(screen.getByText('Nincs elérhető javítási javaslat ehhez a diagnosztikához.')).toBeInTheDocument();
      expect(screen.getByText('0 lépés')).toBeInTheDocument();
    });

    it('handles repair with empty tools_needed and parts_needed safely (the ?? [] fix)', () => {
      setupHookReturn({
        data: createMockResult({
          recommended_repairs: [
            {
              title: 'Test repair',
              description: 'Test desc',
              estimated_cost_currency: 'HUF',
              difficulty: 'beginner',
              parts_needed: undefined as unknown as string[],
              tools_needed: [],
              expert_tips: [],
            },
          ],
        }),
      });
      // Should not throw - the ?? [] handles undefined parts_needed
      render(<ResultPage />);
      expect(screen.getByTestId('repair-step-1')).toBeInTheDocument();
    });

    it('handles missing probable_causes gracefully', () => {
      setupHookReturn({
        data: createMockResult({
          probable_causes: [],
          root_cause_analysis: undefined,
        }),
      });
      render(<ResultPage />);
      // Falls back to default DTC description
      expect(screen.getByText('Henger 3 Égéskimaradás')).toBeInTheDocument();
      // Falls back to default AI analysis text
      expect(screen.getByText('Az AI elemzés nem tartalmaz részletes leírást ehhez a hibakódhoz.')).toBeInTheDocument();
    });

    it('handles missing symptoms (uses default complaint text)', () => {
      setupHookReturn({ data: createMockResult({ symptoms: '' }) });
      render(<ResultPage />);
      expect(
        screen.getByText(/Reggelente rángat a motor hidegindításnál/),
      ).toBeInTheDocument();
    });
  });

  // ── Navigation links ──────────────────────────────────────────────────

  describe('header navigation', () => {
    beforeEach(() => {
      setupHookReturn({ data: createMockResult() });
    });

    it('renders header with MechanicAI branding', () => {
      render(<ResultPage />);
      expect(screen.getByText('MechanicAI')).toBeInTheDocument();
    });

    it('renders navigation links', () => {
      render(<ResultPage />);
      expect(screen.getByText('Vezérlőpult')).toBeInTheDocument();
      expect(screen.getByText('Előzmények')).toBeInTheDocument();
      expect(screen.getByText('Beállítások')).toBeInTheDocument();
    });
  });
});
