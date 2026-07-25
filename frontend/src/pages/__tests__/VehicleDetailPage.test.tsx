import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, within } from '../../test/test-utils';
import userEvent from '@testing-library/user-event';
import VehicleDetailPage from '../VehicleDetailPage';

// A "Gyakori hibák" fül adatállapotai. A CommonIssuesPanel és a lap UGYANABBÓL a
// modulból (`services/hooks/useVehicle`) importál, így egyetlen mock mindkettőt
// lefedi.
const hooksState = vi.hoisted(() => ({
  commonIssues: {
    data: undefined as unknown,
    isLoading: false,
    isError: false,
    error: null as unknown,
  },
  // Az utolsó hívás argumentumai — az "alapértelmezés: minden évjárat"
  // állításhoz.
  commonIssuesArgs: [] as unknown[],
}));

vi.mock('../../services/hooks/useVehicle', () => ({
  useVehicleComplaints: () => ({ data: [], isLoading: false, isError: false }),
  useVehicleCommonIssues: (...args: unknown[]) => {
    hooksState.commonIssuesArgs = args;
    return { ...hooksState.commonIssues, refetch: vi.fn() };
  },
}));

vi.mock('../../services/hooks/useGarage', () => ({
  useVehicle: () => ({
    data: {
      id: 'v1',
      make: 'Volkswagen',
      model: 'Golf',
      year: 2018,
      health_score: 80,
    },
    isLoading: false,
    isError: false,
  }),
  useVehicleHealth: () => ({ data: undefined }),
  useReminders: () => ({ data: undefined }),
  useCosts: () => ({ data: undefined }),
  useVehicleRecalls: () => ({ data: [] }),
  useCreateReminder: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useCompleteReminder: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useDeleteReminder: () => ({ mutateAsync: vi.fn(), isPending: false }),
  useCreateCost: () => ({ mutateAsync: vi.fn(), isPending: false }),
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

/** Egy alkatrészcsoport-sor, minden biztonsági számláló nullán. */
function makeComponent(overrides: Record<string, unknown> = {}) {
  return {
    component: 'ELECTRICAL SYSTEM',
    component_hu: 'Elektromos rendszer',
    complaint_count: 412,
    share: 0.2743,
    crash_count: 0,
    fire_count: 0,
    injury_count: 0,
    death_count: 0,
    ...overrides,
  };
}

/** Rendereli a lapot és megnyitja a "Gyakori hibák" fület. */
async function openCommonIssuesTab() {
  const user = userEvent.setup();
  const view = render(<VehicleDetailPage />);
  await user.click(screen.getByRole('button', { name: /Gyakori hibák/i }));
  return { user, ...view };
}

beforeEach(() => {
  hooksState.commonIssues = {
    data: undefined,
    isLoading: false,
    isError: false,
    error: null,
  };
  hooksState.commonIssuesArgs = [];
});

describe('VehicleDetailPage — "Gyakori hibák" fül', () => {
  it('rangsorolva jeleníti meg az alkatrészcsoportokat aránnyal és darabszámmal', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [],
        components: [
          makeComponent(),
          makeComponent({
            component: 'ENGINE',
            component_hu: 'Motor',
            complaint_count: 150,
            share: 0.0999,
          }),
        ],
        total_complaints: 1502,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    // Magyar címke + arány + nyers darabszám
    expect(screen.getByText('Elektromos rendszer')).toBeInTheDocument();
    expect(screen.getByText('27,4%')).toBeInTheDocument();
    expect(screen.getByText('412 bejelentés')).toBeInTheDocument();

    expect(screen.getByText('Motor')).toBeInTheDocument();
    expect(screen.getByText('10,0%')).toBeInTheDocument();
    expect(screen.getByText('150 bejelentés')).toBeInTheDocument();

    // A rangsor sorrendje: a nagyobb arány elöl
    const rows = screen.getAllByRole('listitem');
    expect(within(rows[0]).getByText('Elektromos rendszer')).toBeInTheDocument();
    expect(within(rows[1]).getByText('Motor')).toBeInTheDocument();

    // A teljes bejelentésszám kontextusként megjelenik — de SOHA nem "összes
    // bejelentés"-ként, mert ez csak a tárolt minta.
    expect(screen.getByText(/bejelentés a mintában/)).toHaveTextContent('1502');
    expect(screen.queryByText(/összes bejelentés/i)).not.toBeInTheDocument();
  });

  it('az arány-sáv szélessége pontosan a share értéke (nem normalizált skála)', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [],
        components: [makeComponent({ share: 0.2743 })],
        total_complaints: 1502,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    const { container } = await openCommonIssuesTab();

    const bar = container.querySelector('.bg-\\[\\#2563eb\\][style]') as HTMLElement | null;
    expect(bar).not.toBeNull();
    // 27,43% — nem 100%, amit egy max-ra normalizált sáv adna.
    expect(bar!.style.width).toBe('27.43%');
  });

  it('component_hu === null esetén a nyers angol címkére esik vissza (nem találgat)', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [],
        components: [
          makeComponent({ component: 'UNKNOWN OR OTHER', component_hu: null, share: 0.05 }),
        ],
        total_complaints: 100,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    expect(screen.getByText('UNKNOWN OR OTHER')).toBeInTheDocument();
  });

  it('a biztonsági jelzéseket csak nem nulla értéknél mutatja', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [],
        components: [makeComponent()],
        total_complaints: 1502,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    expect(screen.queryByText(/baleseti bejelentés/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/tűzeset/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/sérült/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/halálos áldozat/i)).not.toBeInTheDocument();
    // A "0" számláló sehol nem szivárog ki badge-ként
    expect(screen.queryByText('0 sérült')).not.toBeInTheDocument();
  });

  it('a nem nulla biztonsági számlálókat megjeleníti, a halálos áldozatot kiemelt blokkban', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [],
        components: [
          makeComponent({
            crash_count: 3,
            fire_count: 1,
            injury_count: 2,
            death_count: 1,
          }),
        ],
        total_complaints: 1502,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    expect(screen.getByText('3 baleseti bejelentés')).toBeInTheDocument();
    expect(screen.getByText('1 tűzeset')).toBeInTheDocument();
    expect(screen.getByText('2 sérült')).toBeInTheDocument();

    // A halálos áldozat nem apró badge, hanem saját mondatos figyelmeztetés.
    const deathNotice = screen.getByText(/1 halálos áldozat szerepel/i);
    expect(deathNotice).toBeInTheDocument();
    expect(deathNotice.className).toContain('text-red-800');
  });

  it('őszinte üres állapotot mutat, ha nincs NHTSA-adat a járműhöz', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Skoda',
        model: 'Octavia',
        year: null,
        issues: [],
        components: [],
        total_complaints: 0,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    expect(screen.getByText('Erre a járműre nincs NHTSA-adat')).toBeInTheDocument();
    expect(
      screen.getByText(/amerikai piaci fogyasztói bejelentésekből áll/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/nem is jelent hibamentes járművet/i)).toBeInTheDocument();
    // Nem a semmitmondó általános szöveg
    expect(screen.queryByText(/^Nincs adat$/)).not.toBeInTheDocument();
  });

  it('betöltés közben role="status"-szal jelez', async () => {
    hooksState.commonIssues = {
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    const status = screen.getByRole('status');
    expect(status).toBeInTheDocument();
    expect(within(status).getByText('Gyakori hibák betöltése…')).toBeInTheDocument();
  });

  it('hiba esetén hibaállapotot mutat újrapróbálás lehetőséggel', async () => {
    hooksState.commonIssues = {
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error('boom'),
    };

    await openCommonIssuesTab();

    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Nem sikerült betölteni a gyakori hibákat')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Ujraprobalas/i })).toBeInTheDocument();
  });

  it('üres issues tömb esetén nem renderel törött hibakód-szekciót', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [],
        components: [makeComponent()],
        total_complaints: 1502,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    // A komponens-rangsor megvan…
    expect(screen.getByText('Elektromos rendszer')).toBeInTheDocument();
    // …de a DTC szekció fejléce nyomtalanul hiányzik.
    expect(
      screen.queryByRole('heading', { name: /Bejelentésekben említett hibakódok/i })
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/említés$/)).not.toBeInTheDocument();
  });

  it('nem üres issues tömböt másodlagos hibakód-szekcióként jelenít meg', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [
          {
            code: 'P0301',
            description_en: 'Cylinder 1 Misfire Detected',
            description_hu: '1-es henger égéskimaradás',
            severity: 'high',
            frequency: 'common',
            occurrence_count: 7,
          },
        ],
        components: [makeComponent()],
        total_complaints: 1502,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    await openCommonIssuesTab();

    expect(
      screen.getByRole('heading', { name: /Bejelentésekben említett hibakódok/i })
    ).toBeInTheDocument();
    expect(screen.getByText('P0301')).toBeInTheDocument();
    expect(screen.getByText('1-es henger égéskimaradás')).toBeInTheDocument();
    expect(screen.getByText('Magas')).toBeInTheDocument();
    expect(screen.getByText('Gyakori')).toBeInTheDocument();
    expect(screen.getByText('7 említés')).toBeInTheDocument();
  });

  it('alapértelmezésben MINDEN évjáratot kérdez le (year === undefined)', async () => {
    hooksState.commonIssues = {
      data: {
        make: 'Volkswagen',
        model: 'Golf',
        year: null,
        issues: [],
        components: [makeComponent()],
        total_complaints: 1502,
      },
      isLoading: false,
      isError: false,
      error: null,
    };

    const { user } = await openCommonIssuesTab();

    // make, model, year — a jármű 2018-as, mégis undefined évvel kérdezünk.
    expect(hooksState.commonIssuesArgs[0]).toBe('Volkswagen');
    expect(hooksState.commonIssuesArgs[1]).toBe('Golf');
    expect(hooksState.commonIssuesArgs[2]).toBeUndefined();

    // A kapcsoló alapállása és a kontextussor is az összesített nézetet mutatja.
    expect(screen.getByRole('button', { name: 'Minden évjárat' })).toHaveAttribute(
      'aria-pressed',
      'true'
    );
    expect(screen.getByText(/bejelentés a mintában/)).toHaveTextContent('minden évjárat');

    // Az évjárat-szűrés csak kifejezett kapcsolóval történik.
    await user.click(screen.getByRole('button', { name: 'Csak 2018' }));
    expect(hooksState.commonIssuesArgs[2]).toBe(2018);
  });
});

describe('VehicleDetailPage — fülsáv', () => {
  it('mind az öt fül a kanonikus h-9 px-5 font-semibold változatot használja', () => {
    render(<VehicleDetailPage />);

    const labels = [
      /Emlékeztetők/i,
      /Karbantartási log/i,
      /Gyakori hibák/i,
      /Visszahívások/i,
      /Panaszok/i,
    ];

    for (const label of labels) {
      const tab = screen.getByRole('button', { name: label });
      expect(tab.className).toContain('h-9');
      expect(tab.className).toContain('px-5');
      expect(tab.className).toContain('font-semibold');
      expect(tab).toHaveAttribute('aria-pressed');
    }
  });

  it('nem ágyaz be második main landmarkot (a Layout birtokolja az egyetlent)', () => {
    const { container } = render(<VehicleDetailPage />);
    expect(container.querySelectorAll('main')).toHaveLength(0);
  });
});
