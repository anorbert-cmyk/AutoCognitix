/**
 * CommonIssuesPanel — igazmondási tesztek.
 *
 * A panel két FÜGGETLEN adatforrásból dolgozik (`components`: PostgreSQL,
 * `issues`: Neo4j), amelyek külön-külön esnek ki. A backend `sources` mezője
 * mondja meg forrásonként, hogy az üres lista adathiányt vagy kiesést jelent.
 *
 * Amit ez a fájl leszögez:
 *  - a `components` üressége nem nyelheti el a valós `issues` listát,
 *  - kiesett forrásnál TILOS tényként állítani, hogy "ez a modell ott nem volt
 *    forgalomban" — ott őszinte "most nem elérhető" állapot jár újrapróbálással,
 *  - tényleges adathiánynál viszont marad az eredeti, őszinte szöveg.
 *
 * Per-file mock: a `test-utils.tsx` szándékosan nem tartalmaz AuthProvider-t, a
 * hook-modult itt, fájlonként mockoljuk.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../../test/test-utils';
import userEvent from '@testing-library/user-event';
import CommonIssuesPanel from '../CommonIssuesPanel';

const hooksState = vi.hoisted(() => ({
  commonIssues: {
    data: undefined as unknown,
    isLoading: false,
    isError: false,
    error: null as unknown,
  },
  refetch: vi.fn(),
}));

vi.mock('../../../../services/hooks/useVehicle', () => ({
  useVehicleCommonIssues: () => ({
    ...hooksState.commonIssues,
    refetch: hooksState.refetch,
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

/** Egy DTC-sor a másodlagos hibakód-szekcióhoz. */
function makeIssue(overrides: Record<string, unknown> = {}) {
  return {
    code: 'P0301',
    description_en: 'Cylinder 1 Misfire Detected',
    description_hu: '1-es henger égéskimaradás',
    severity: 'high',
    frequency: 'common',
    occurrence_count: 7,
    ...overrides,
  };
}

/** Teljes válasz — a `sources` MINDIG kimondva, ahogy a backend is küldi. */
function response(overrides: Record<string, unknown> = {}) {
  return {
    make: 'Volkswagen',
    model: 'Golf',
    year: null,
    issues: [],
    components: [],
    total_complaints: 0,
    sources: { components: 'ok', issues: 'ok' },
    ...overrides,
  };
}

function setData(data: unknown) {
  hooksState.commonIssues = { data, isLoading: false, isError: false, error: null };
}

const NEVER_SOLD_COPY = /nem volt forgalomban/i;
const UNAVAILABLE_TITLE = 'A bejelentési adatok most nem elérhetők';

beforeEach(() => {
  hooksState.commonIssues = { data: undefined, isLoading: false, isError: false, error: null };
  hooksState.refetch = vi.fn();
});

describe('CommonIssuesPanel — a források függetlensége', () => {
  it('megjeleníti a hibakódokat akkor is, ha a components lista üres', () => {
    setData(response({ issues: [makeIssue()], components: [], total_complaints: 0 }));

    render(<CommonIssuesPanel make="Volkswagen" model="Golf" />);

    // A DTC-rangsor NEM veszhet el csak azért, mert a másik forrás üres.
    expect(
      screen.getByRole('heading', { name: /Bejelentésekben említett hibakódok/i })
    ).toBeInTheDocument();
    expect(screen.getByText('P0301')).toBeInTheDocument();
    expect(screen.getByText('1-es henger égéskimaradás')).toBeInTheDocument();

    // …és a komponens-forrás üres állapota emellett, nem helyette jelenik meg.
    expect(screen.getByText('Erre a járműre nincs NHTSA-adat')).toBeInTheDocument();
  });

  it('kiesett components mellett is megjeleníti a valós hibakódokat', () => {
    setData(
      response({
        issues: [makeIssue({ code: 'P0420', description_hu: 'Katalizátor hatásfok' })],
        components: [],
        sources: { components: 'unavailable', issues: 'ok' },
      })
    );

    render(<CommonIssuesPanel make="Volkswagen" model="Golf" />);

    expect(screen.getByText('P0420')).toBeInTheDocument();
    expect(screen.getByText(UNAVAILABLE_TITLE)).toBeInTheDocument();
    expect(screen.queryByText(NEVER_SOLD_COPY)).not.toBeInTheDocument();
  });
});

describe('CommonIssuesPanel — kiesett forrás vs. tényleges adathiány', () => {
  it('kiesett forrásnál őszinte "most nem elérhető" állapotot mutat, nem adathiányt állít', async () => {
    const user = userEvent.setup();
    setData(response({ sources: { components: 'unavailable', issues: 'ok' } }));

    render(<CommonIssuesPanel make="Volkswagen" model="Golf" />);

    expect(screen.getByText(UNAVAILABLE_TITLE)).toBeInTheDocument();
    expect(screen.getByText(/Ez nem jelenti azt, hogy nincs adat/i)).toBeInTheDocument();

    // A tényállítás egyik fele sem hangozhat el kiesett forrásnál.
    expect(screen.queryByText('Erre a járműre nincs NHTSA-adat')).not.toBeInTheDocument();
    expect(screen.queryByText(NEVER_SOLD_COPY)).not.toBeInTheDocument();
    expect(screen.queryByText(/nincs amerikai bejelentés az adatbázisunkban/i)).not.toBeInTheDocument();

    // Újrapróbálás a megosztott ErrorState primitívből.
    await user.click(screen.getByRole('button', { name: /Ujraprobalas/i }));
    expect(hooksState.refetch).toHaveBeenCalledTimes(1);
  });

  it('tényleges adathiánynál az eredeti őszinte szöveg marad', () => {
    setData(response({ make: 'Skoda', model: 'Octavia' }));

    render(<CommonIssuesPanel make="Skoda" model="Octavia" />);

    expect(screen.getByText('Erre a járműre nincs NHTSA-adat')).toBeInTheDocument();
    expect(screen.getByText(/amerikai piaci fogyasztói bejelentésekből áll/i)).toBeInTheDocument();
    expect(screen.getByText(/nem is jelent hibamentes járművet/i)).toBeInTheDocument();
    expect(screen.queryByText(UNAVAILABLE_TITLE)).not.toBeInTheDocument();
  });

  it('a kiesés az évjárat-szűrős üres állapotot is felülírja', async () => {
    const user = userEvent.setup();
    setData(response({ sources: { components: 'unavailable', issues: 'ok' } }));

    render(<CommonIssuesPanel make="Volkswagen" model="Golf" vehicleYear={2018} />);

    await user.click(screen.getByRole('button', { name: 'Csak 2018' }));

    // Nem az "erre az évjáratra nincs bejelentés" — azt sem tudjuk, van-e.
    expect(screen.getByText(UNAVAILABLE_TITLE)).toBeInTheDocument();
    expect(screen.queryByText(/A 2018\. évjáratra nincs bejelentés/i)).not.toBeInTheDocument();
  });
});

describe('CommonIssuesPanel — a hibakód-forrás kiesése', () => {
  it('jelzi, ha csak a hibakódok nem elérhetők', () => {
    setData(
      response({
        components: [makeComponent()],
        total_complaints: 1502,
        sources: { components: 'ok', issues: 'unavailable' },
      })
    );

    render(<CommonIssuesPanel make="Volkswagen" model="Golf" />);

    // Az alkatrész-rangsor megvan…
    expect(screen.getByText('Elektromos rendszer')).toBeInTheDocument();
    // …a hiányzó hibakód-szekcióról pedig őszintén megmondjuk, miért hiányzik.
    expect(screen.getByText('A hibakódok most nem elérhetők')).toBeInTheDocument();
  });

  it('ép hibakód-forrásnál nyomtalanul eltűnik az üres szekció', () => {
    setData(response({ components: [makeComponent()], total_complaints: 1502 }));

    render(<CommonIssuesPanel make="Volkswagen" model="Golf" />);

    expect(screen.queryByText('A hibakódok most nem elérhetők')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('heading', { name: /Bejelentésekben említett hibakódok/i })
    ).not.toBeInTheDocument();
  });

  it('mindkét forrás kiesésekor nem ismétli meg ugyanazt az üzenetet', () => {
    setData(response({ sources: { components: 'unavailable', issues: 'unavailable' } }));

    render(<CommonIssuesPanel make="Volkswagen" model="Golf" />);

    expect(screen.getByText(UNAVAILABLE_TITLE)).toBeInTheDocument();
    expect(screen.queryByText('A hibakódok most nem elérhetők')).not.toBeInTheDocument();
    expect(screen.getAllByRole('alert')).toHaveLength(1);
  });
});
