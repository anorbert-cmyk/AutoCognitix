import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../test/test-utils';
import GaragePage from '../GaragePage';

// Mutable vehicles-query state so each test can supply its own data.
const hooksState = vi.hoisted(() => ({
  vehicles: {
    data: undefined as unknown,
    isLoading: false,
    error: null as unknown,
  },
}));

// GaragePage consumes useVehicles / useCreateVehicle / useDeleteVehicle from the
// direct hook module (not the barrel).
vi.mock('../../services/hooks/useGarage', () => ({
  useVehicles: () => hooksState.vehicles,
  useCreateVehicle: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useDeleteVehicle: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

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

beforeEach(() => {
  hooksState.vehicles = { data: undefined, isLoading: false, error: null };
});

describe('GaragePage — vehicle card health rendering', () => {
  it('renders the real health score and reminder count, never fabricating 75', () => {
    hooksState.vehicles = {
      data: {
        vehicles: [
          {
            id: 'v1',
            make: 'Volkswagen',
            model: 'Golf',
            year: 2018,
            health_score: 42,
            upcoming_reminders_count: 3,
          },
        ],
        total: 1,
      },
      isLoading: false,
      error: null,
    };

    render(<GaragePage />);

    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.queryByText('75')).not.toBeInTheDocument();
    expect(screen.getByText('3 emlékeztető')).toBeInTheDocument();
  });

  it('renders "Nincs adat" when health_score is null, with no fabricated 75', () => {
    hooksState.vehicles = {
      data: {
        vehicles: [
          {
            id: 'v2',
            make: 'Toyota',
            model: 'Corolla',
            year: 2020,
            health_score: null,
            upcoming_reminders_count: 0,
          },
        ],
        total: 1,
      },
      isLoading: false,
      error: null,
    };

    render(<GaragePage />);

    expect(screen.getByText('Nincs adat')).toBeInTheDocument();
    expect(screen.queryByText('75')).not.toBeInTheDocument();
  });
});
