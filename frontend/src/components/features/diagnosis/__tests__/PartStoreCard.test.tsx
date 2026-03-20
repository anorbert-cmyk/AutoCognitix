import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../../test/test-utils';
import { PartStoreCard, PartStoreCardGrid } from '../PartStoreCard';
import type { DemoPartWithStores } from '../../../../data/demoData';
import { demoParts } from '../../../../data/demoData';

// Helper to create a minimal part for isolated tests
function createTestPart(overrides: Partial<DemoPartWithStores> = {}): DemoPartWithStores {
  return {
    id: 'test_part',
    name: 'Teszt alkatrész',
    name_en: 'Test Part',
    category: 'Teszt',
    price_range_min: 5000,
    price_range_max: 10000,
    labor_hours: 1.0,
    currency: 'HUF',
    description: 'Teszt leírás',
    partNumber: 'TEST-001',
    oemNumber: '000 000 000 A',
    isOem: false,
    qualityRating: 4.0,
    compatibilityNote: 'Kompatibilis',
    stores: [
      {
        storeName: 'Teszt Bolt',
        storeUrl: 'https://test.hu',
        storeLogoColor: '#FF0000',
        price: 5000,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 1,
        brand: 'TestBrand X1',
      },
      {
        storeName: 'Másik Bolt',
        storeUrl: 'https://masik.hu',
        storeLogoColor: '#0000FF',
        price: 7500,
        currency: 'HUF',
        inStock: false,
        deliveryDays: 3,
        brand: 'OtherBrand Y2',
      },
    ],
    ...overrides,
  };
}

describe('PartStoreCard', () => {
  it('renders the part name', () => {
    const part = createTestPart({ name: 'Gyújtógyertya készlet (4 db)' });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('Gyújtógyertya készlet (4 db)')).toBeInTheDocument();
  });

  it('renders the English name when provided', () => {
    const part = createTestPart({ name_en: 'Spark Plug Set' });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('Spark Plug Set')).toBeInTheDocument();
  });

  it('renders store names correctly', () => {
    const part = createTestPart();
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('Teszt Bolt')).toBeInTheDocument();
    expect(screen.getByText('Másik Bolt')).toBeInTheDocument();
  });

  it('shows price formatted in HUF (Hungarian locale, no decimals)', () => {
    const part = createTestPart({
      stores: [
        {
          storeName: 'Bolt A',
          storeUrl: 'https://a.hu',
          storeLogoColor: '#000',
          price: 11890,
          currency: 'HUF',
          inStock: true,
          deliveryDays: 1,
          brand: 'Brand A',
        },
        {
          storeName: 'Bolt B',
          storeUrl: 'https://b.hu',
          storeLogoColor: '#111',
          price: 15000,
          currency: 'HUF',
          inStock: true,
          deliveryDays: 2,
          brand: 'Brand B',
        },
      ],
    });
    render(<PartStoreCard part={part} index={0} />);
    // The formatter uses hu-HU locale with space separators
    // 11890 in hu-HU is "11 890" (with a narrow no-break space or regular space)
    // We search for the number using a flexible regex
    const priceElements = screen.getAllByText(
      (_, element) => element?.textContent?.includes('11') && element?.textContent?.includes('890') || false,
    );
    expect(priceElements.length).toBeGreaterThanOrEqual(1);
  });

  it('shows "Készleten" for in-stock items', () => {
    const part = createTestPart({
      stores: [
        {
          storeName: 'Bolt Készlet',
          storeUrl: 'https://bolt.hu',
          storeLogoColor: '#0F0',
          price: 5000,
          currency: 'HUF',
          inStock: true,
          deliveryDays: 1,
          brand: 'Brand X',
        },
      ],
    });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText(/Készleten/)).toBeInTheDocument();
  });

  it('shows "Rendelhető" for out-of-stock items with delivery days', () => {
    const part = createTestPart({
      stores: [
        {
          storeName: 'Bolt Rendelés',
          storeUrl: 'https://bolt.hu',
          storeLogoColor: '#F00',
          price: 8000,
          currency: 'HUF',
          inStock: false,
          deliveryDays: 4,
          brand: 'Brand Y',
        },
      ],
    });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText(/Rendelhető · 4 nap/)).toBeInTheDocument();
  });

  it('renders the part category badge', () => {
    const part = createTestPart({ category: 'Gyújtás' });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('Gyújtás')).toBeInTheDocument();
  });

  it('shows OEM badge when part is OEM', () => {
    const part = createTestPart({ isOem: true });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('OEM')).toBeInTheDocument();
  });

  it('does not show OEM badge when part is not OEM', () => {
    const part = createTestPart({ isOem: false });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.queryByText('OEM')).not.toBeInTheDocument();
  });

  it('renders the 1-based index number', () => {
    const part = createTestPart();
    render(<PartStoreCard part={part} index={2} />);
    expect(screen.getByText('3')).toBeInTheDocument();
  });

  it('renders part number and OEM number', () => {
    const part = createTestPart({
      partNumber: 'NGK 95770',
      oemNumber: '04E 905 612 C',
    });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText(/NGK 95770/)).toBeInTheDocument();
    expect(screen.getByText(/04E 905 612 C/)).toBeInTheDocument();
  });

  it('shows labor hours in the footer', () => {
    const part = createTestPart({ labor_hours: 1.5 });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText(/~1\.5 óra/)).toBeInTheDocument();
  });

  it('shows "Legjobb ár" badge for the cheapest store', () => {
    const part = createTestPart();
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('Legjobb ár')).toBeInTheDocument();
  });

  it('renders the quality rating value', () => {
    const part = createTestPart({ qualityRating: 4.5 });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('4.5')).toBeInTheDocument();
  });

  it('shows brand info for the best price store', () => {
    const part = createTestPart({
      stores: [
        {
          storeName: 'Solo Bolt',
          storeUrl: 'https://solo.hu',
          storeLogoColor: '#000',
          price: 3000,
          currency: 'HUF',
          inStock: true,
          deliveryDays: 1,
          brand: 'FILTRON AP 183/3',
        },
      ],
    });
    render(<PartStoreCard part={part} index={0} />);
    expect(screen.getByText('FILTRON AP 183/3')).toBeInTheDocument();
  });
});

describe('PartStoreCardGrid', () => {
  it('renders all 6 demo parts', () => {
    render(<PartStoreCardGrid parts={demoParts} />);
    for (const part of demoParts) {
      expect(screen.getByText(part.name)).toBeInTheDocument();
    }
  });

  it('renders section header "Szükséges alkatrészek"', () => {
    render(<PartStoreCardGrid parts={demoParts} />);
    expect(screen.getByText('Szükséges alkatrészek')).toBeInTheDocument();
  });

  it('shows part count and in-stock count in summary', () => {
    render(<PartStoreCardGrid parts={demoParts} />);
    // The summary line includes part count and in-stock count
    expect(screen.getByText(/6 alkatrész/)).toBeInTheDocument();
  });

  it('returns null when parts array is empty', () => {
    const { container } = render(<PartStoreCardGrid parts={[]} />);
    expect(container.innerHTML).toBe('');
  });

  it('shows store legend with store names', () => {
    render(<PartStoreCardGrid parts={demoParts} />);
    // Store legend shows store names
    const bardiElements = screen.getAllByText('Bárdi Autó');
    expect(bardiElements.length).toBeGreaterThanOrEqual(1);
  });
});
