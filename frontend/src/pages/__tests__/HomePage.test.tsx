import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../test/test-utils';
import HomePage from '../HomePage';

// Mutable auth state so tests can flip logged-in / logged-out.
const authState = vi.hoisted(() => ({
  value: {
    user: null as { full_name?: string; email?: string } | null,
    isAuthenticated: false,
    isLoading: false,
  },
}));

vi.mock('../../contexts/AuthContext', () => ({
  useAuth: () => authState.value,
}));

// Mock the hooks barrel that HomePage consumes (dashboard data sources).
const hooksState = vi.hoisted(() => ({
  vehicles: { data: undefined as unknown, isLoading: false },
  history: { data: undefined as unknown, isLoading: false },
  reminders: { data: undefined as unknown, isLoading: false },
}));

vi.mock('../../services/hooks', () => ({
  useVehicles: () => hooksState.vehicles,
  useDiagnosisHistory: () => hooksState.history,
  useUpcomingReminders: () => hooksState.reminders,
}));

beforeEach(() => {
  authState.value = { user: null, isAuthenticated: false, isLoading: false };
  hooksState.vehicles = { data: undefined, isLoading: false };
  hooksState.history = { data: undefined, isLoading: false };
  hooksState.reminders = { data: undefined, isLoading: false };
});

const loginAs = (fullName: string) => {
  authState.value = {
    user: { full_name: fullName, email: 'user@example.com' },
    isAuthenticated: true,
    isLoading: false,
  };
};

describe('HomePage — marketing (logged out)', () => {
  it('should render the hero section with title', () => {
    render(<HomePage />);
    expect(
      screen.getByText('AI-alapú Gépjármű Diagnosztika'),
    ).toBeInTheDocument();
  });

  it('should render the honest platform stat strip', () => {
    render(<HomePage />);
    expect(screen.getByText('26 816')).toBeInTheDocument();
    expect(screen.getByText('35 000+')).toBeInTheDocument();
  });

  it('should render the features section', () => {
    render(<HomePage />);
    expect(screen.getByText('Hogyan működik?')).toBeInTheDocument();
    expect(screen.getByText('Hibakód bevitel')).toBeInTheDocument();
    expect(screen.getByText('AI elemzés')).toBeInTheDocument();
    expect(screen.getByText('Részletes diagnózis')).toBeInTheDocument();
  });

  it('should render diagnosis CTA links', () => {
    render(<HomePage />);
    const ctaLinks = screen.getAllByText('Diagnózis indítása');
    expect(ctaLinks.length).toBeGreaterThanOrEqual(2);
    ctaLinks.forEach((link) => {
      expect(link.closest('a')).toHaveAttribute('href', '/diagnosis');
    });
  });

  it('should render the CTA section', () => {
    render(<HomePage />);
    expect(
      screen.getByText('Készen áll a diagnosztikára?'),
    ).toBeInTheDocument();
  });
});

describe('HomePage — dashboard (logged in)', () => {
  it('greets the user by given name (Hungarian name order) and hides the marketing hero', () => {
    loginAs('Barna Norbert');
    render(<HomePage />);
    expect(screen.getByText(/Üdv újra, Norbert!/)).toBeInTheDocument();
    expect(
      screen.queryByText('AI-alapú Gépjármű Diagnosztika'),
    ).not.toBeInTheDocument();
  });

  it('renders the primary "Új diagnosztika" action linking to /diagnosis', () => {
    loginAs('Barna Norbert');
    render(<HomePage />);
    const cta = screen.getByText('Új diagnosztika');
    expect(cta.closest('a')).toHaveAttribute('href', '/diagnosis');
  });

  it('lists garage vehicles with links to their detail pages', () => {
    loginAs('Barna Norbert');
    hooksState.vehicles = {
      data: {
        vehicles: [
          {
            id: 'v1',
            make: 'Volkswagen',
            model: 'Golf',
            year: 2018,
            license_plate: 'ABC-123',
            nickname: null,
          },
        ],
        total: 1,
      },
      isLoading: false,
    };
    render(<HomePage />);
    const row = screen.getByText('Volkswagen Golf');
    expect(row.closest('a')).toHaveAttribute('href', '/garage/v1');
    expect(screen.getByText('1 jármű')).toBeInTheDocument();
  });

  it('shows an add-vehicle empty state when the garage is empty', () => {
    loginAs('Barna Norbert');
    hooksState.vehicles = { data: { vehicles: [], total: 0 }, isLoading: false };
    render(<HomePage />);
    expect(screen.getByText('Még nincs jármű a garázsban.')).toBeInTheDocument();
    expect(
      screen.getByText('Jármű hozzáadása').closest('a'),
    ).toHaveAttribute('href', '/garage');
  });

  it('lists recent diagnoses with DTC chips linking to the result page', () => {
    loginAs('Barna Norbert');
    hooksState.history = {
      data: {
        items: [
          {
            id: 'd1',
            vehicle_make: 'Volkswagen',
            vehicle_model: 'Golf',
            vehicle_year: 2018,
            dtc_codes: ['P0300', 'P0301'],
            confidence_score: 0.87,
            created_at: '2026-07-01T10:00:00Z',
          },
        ],
        total: 1,
        skip: 0,
        limit: 3,
        has_more: false,
      },
      isLoading: false,
    };
    render(<HomePage />);
    const chip = screen.getByText('P0300');
    expect(chip.closest('a')).toHaveAttribute('href', '/diagnosis/d1');
    expect(screen.getByText('P0301')).toBeInTheDocument();
  });

  it('renders upcoming reminders with urgency copy', () => {
    loginAs('Barna Norbert');
    hooksState.reminders = {
      data: [
        {
          id: 'r1',
          vehicle_id: 'v1',
          title: 'Olajcsere',
          urgency: 'urgent',
          days_until_due: 3,
          due_date: '2026-07-22',
        },
      ],
      isLoading: false,
    };
    render(<HomePage />);
    expect(screen.getByText('Olajcsere')).toBeInTheDocument();
    expect(screen.getByText('3 nap múlva')).toBeInTheDocument();
  });

  it('shows a calm empty state when there are no upcoming reminders', () => {
    loginAs('Barna Norbert');
    hooksState.reminders = { data: [], isLoading: false };
    render(<HomePage />);
    expect(
      screen.getByText('Nincs közelgő teendő a következő 14 napban.'),
    ).toBeInTheDocument();
  });
});
