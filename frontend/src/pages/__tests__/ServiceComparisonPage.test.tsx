import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../test/test-utils';
import ServiceComparisonPage from '../ServiceComparisonPage';

// =============================================================================
// Mocks
// =============================================================================

vi.mock('../../services/hooks/useServiceShops', () => ({
  useServiceShops: () => ({
    data: { items: [], total: 0 },
    isLoading: false,
    isError: false,
    error: null,
  }),
}));

vi.mock('../../data/regions', () => ({
  regions: [
    { id: 'budapest', name: 'Budapest' },
    { id: 'pest', name: 'Pest megye' },
    { id: 'gyor', name: 'Gyor-Moson-Sopron' },
  ],
}));

vi.mock('../../components/features/services/RegionSelector', () => ({
  RegionSelector: ({
    regions,
    selectedRegion,
    onChange,
  }: {
    regions: Array<{ id: string; name: string }>;
    selectedRegion: string;
    onChange: (region: string) => void;
  }) => (
    <div data-testid="region-selector">
      <select
        value={selectedRegion}
        onChange={(e) => onChange(e.target.value)}
        data-testid="region-select"
      >
        <option value="">Minden regio</option>
        {regions.map((r) => (
          <option key={r.id} value={r.id}>
            {r.name}
          </option>
        ))}
      </select>
    </div>
  ),
}));

vi.mock('../../components/features/services/ShopFilters', () => ({
  ShopFilters: ({
    filters,
    onFilterChange,
  }: {
    filters: Record<string, string>;
    onFilterChange: (key: string, value: string) => void;
  }) => (
    <div data-testid="shop-filters">
      <select
        data-testid="filter-sort"
        value={filters.sort_by}
        onChange={(e) => onFilterChange('sort_by', e.target.value)}
      >
        <option value="rating">Ertekeles</option>
        <option value="price">Ar</option>
        <option value="distance">Tavolsag</option>
      </select>
    </div>
  ),
}));

vi.mock('../../components/features/services/ShopCard', () => ({
  ShopCard: ({ shop }: { shop: { id: string; name: string } }) => (
    <div data-testid={`shop-card-${shop.id}`}>{shop.name}</div>
  ),
}));

// =============================================================================
// Tests
// =============================================================================

describe('ServiceComparisonPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  it('should render the page title', () => {
    render(<ServiceComparisonPage />);
    expect(
      screen.getByText(/Szerviz.*sszehasonl/),
    ).toBeInTheDocument();
  });

  it('should render the page description', () => {
    render(<ServiceComparisonPage />);
    expect(
      screen.getByText(/Keresd meg a legjobb szervizeket/),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Region Selector
  // ---------------------------------------------------------------------------

  it('should render the region selector', () => {
    render(<ServiceComparisonPage />);
    expect(screen.getByTestId('region-selector')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Filters
  // ---------------------------------------------------------------------------

  it('should render the shop filters', () => {
    render(<ServiceComparisonPage />);
    expect(screen.getByTestId('shop-filters')).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Results Count
  // ---------------------------------------------------------------------------

  it('should display the result count badge', () => {
    render(<ServiceComparisonPage />);
    expect(screen.getByText('0')).toBeInTheDocument();
    expect(screen.getByText(/szerviz tal/)).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Map Placeholder
  // ---------------------------------------------------------------------------

  it('should render the map placeholder', () => {
    render(<ServiceComparisonPage />);
    expect(
      screen.getByText(/rk.*p bet.*lt/),
    ).toBeInTheDocument();
  });

  // ---------------------------------------------------------------------------
  // Empty State
  // ---------------------------------------------------------------------------

  it('should show empty state when no shops found', () => {
    render(<ServiceComparisonPage />);
    expect(
      screen.getByText(/Nincs tal.*lat/),
    ).toBeInTheDocument();
  });
});
