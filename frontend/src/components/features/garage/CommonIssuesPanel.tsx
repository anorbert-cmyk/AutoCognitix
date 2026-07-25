/**
 * CommonIssuesPanel — "Gyakori hibák" panel a jármű részletek oldalon.
 *
 * Forrás: `GET /api/v1/vehicles/{make}/{model}/common-issues`
 *
 * Az adat természetéből fakadó, szándékos tervezési döntések:
 *
 * 1. Az ELSŐDLEGES adat a `components` lista (NHTSA panasz-gyakoriság szerinti
 *    alkatrészcsoport-rangsor). Az `issues` (DTC-kódok) lista a gráf ritka
 *    panasz→DTC kapcsolatai miatt a legtöbb járműnél ÜRES — ezért csak
 *    másodlagos, feltételesen megjelenő szekció, és üresen nyomtalanul eltűnik.
 * 2. A két lista két FÜGGETLEN adatbázisból jön (PostgreSQL, illetve Neo4j), és
 *    külön-külön esik üresre. Ezért az `issues` szekció akkor is megjelenik, ha a
 *    `components` üres: az egyik forrás kiesése nem dobhatja el a másik valós
 *    adatát.
 * 3. A panaszkorpusz amerikai piaci NHTSA-adat, a márka/modell választó viszont
 *    európai gyártókat is kínál. A "nincs találat" eset NEM hiba: külön,
 *    őszintén megfogalmazott üres állapotot kap — de CSAK akkor, ha a válasz
 *    `sources` mezője szerint a forrás tényleg válaszolt. Kiesett forrásnál
 *    (`'unavailable'`) az üres lista semmit nem bizonyít a járműről, ezért ott
 *    "most nem elérhető" állapot jár újrapróbálással, nem adathiány-állítás.
 * 4. A számok a NÁLUNK TÁROLT bejelentések mintáján alapulnak (az importáló
 *    márkánként és alkatrészcsoportonként korlátoz), ezért sehol nem hívjuk
 *    őket "összes bejelentés"-nek. Az elsődleges nagyságrend-jelző az arány
 *    (`share`), a nyers darabszám másodlagos.
 * 5. Alapértelmezés: MINDEN évjárat (a végpont `year` nélkül lényegesen
 *    gazdagabb eredményt ad). Évjáratra szűrni csak kifejezett kapcsolóval
 *    lehet.
 * 6. A `component_hu` lehet null — ilyenkor a nyers angol `component` címkére
 *    esünk vissza, soha nem találgatunk fordítást.
 */

import { useState } from 'react'
import { AlertTriangle, Car, Wrench } from 'lucide-react'
import { EmptyState, ErrorState, Skeleton } from '../../ui'
import { useVehicleCommonIssues } from '../../../services/hooks/useVehicle'
import type { VehicleComplaintComponent } from '../../../services/api'

// =============================================================================
// Constants
// =============================================================================

const focusRing =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'

/** A backend `_frequency_bucket` szókészlete. Ismeretlen érték → nem jelenítjük meg. */
const FREQUENCY_LABELS: Record<string, string> = {
  rare: 'Ritka',
  uncommon: 'Nem gyakori',
  common: 'Gyakori',
  very_common: 'Nagyon gyakori',
}

/** WCAG AA-t teljesítő súlyosság-chipek (a dtcService színpárjai ennél halványabbak). */
const SEVERITY_CHIPS: Record<string, { label: string; className: string }> = {
  low: { label: 'Alacsony', className: 'bg-green-100 text-green-800' },
  medium: { label: 'Közepes', className: 'bg-yellow-100 text-yellow-800' },
  high: { label: 'Magas', className: 'bg-orange-100 text-orange-800' },
  critical: { label: 'Kritikus', className: 'bg-red-100 text-red-800' },
}

// =============================================================================
// Helpers
// =============================================================================

const formatCount = (value: number): string => value.toLocaleString('hu-HU')

/** `share` (0..1) → "27,4%" */
const formatShare = (share: number): string =>
  `${(share * 100).toLocaleString('hu-HU', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })}%`

/**
 * A sáv szélessége PONTOSAN a `share` értéke — nincs a legnagyobb elemre
 * normalizálás, ami felnagyítaná a kis arányokat. A számszerű százalék a sáv
 * mellett mindig olvasható, így a nagyon kicsi (alig látható) sávok sem
 * vesztenek információt.
 */
const barWidth = (share: number): string =>
  `${Math.min(100, Math.max(0, share * 100)).toFixed(2)}%`

/** A magyar címke, vagy — ellenőrzött fordítás híján — a nyers NHTSA-címke. */
const componentLabel = (component: VehicleComplaintComponent): string =>
  component.component_hu || component.component

// =============================================================================
// Safety signals
// =============================================================================

/**
 * Biztonsági jelzések. Csak a nem nulla értékek jelennek meg.
 *
 * A halálos kimenetel NEM badge: saját, ikonos figyelmeztető blokkot kap a sor
 * alján, a többi jelzésnél jóval nagyobb vizuális súllyal.
 */
function SafetySignals({ component }: { component: VehicleComplaintComponent }) {
  const badges: Array<{ key: string; text: string; className: string }> = []

  if (component.crash_count > 0) {
    badges.push({
      key: 'crash',
      text: `${formatCount(component.crash_count)} baleseti bejelentés`,
      className: 'bg-red-100 text-red-700',
    })
  }
  if (component.fire_count > 0) {
    badges.push({
      key: 'fire',
      text: `${formatCount(component.fire_count)} tűzeset`,
      className: 'bg-orange-100 text-orange-700',
    })
  }
  if (component.injury_count > 0) {
    badges.push({
      key: 'injury',
      text: `${formatCount(component.injury_count)} sérült`,
      className: 'bg-red-100 text-red-700',
    })
  }

  const hasDeaths = component.death_count > 0

  if (badges.length === 0 && !hasDeaths) {
    return null
  }

  return (
    <>
      {badges.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 mt-2">
          {badges.map((badge) => (
            <span
              key={badge.key}
              className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${badge.className}`}
            >
              {badge.text}
            </span>
          ))}
        </div>
      )}

      {hasDeaths && (
        <div className="flex items-start gap-2 mt-2 rounded-lg border border-red-300 bg-red-50 px-3 py-2">
          <AlertTriangle className="h-4 w-4 text-red-700 flex-shrink-0 mt-0.5" aria-hidden="true" />
          <p className="text-sm font-bold text-red-800 leading-snug">
            {formatCount(component.death_count)} halálos áldozat szerepel az ehhez az
            alkatrészcsoporthoz tartozó bejelentésekben.
          </p>
        </div>
      )}
    </>
  )
}

// =============================================================================
// CommonIssuesPanel
// =============================================================================

interface CommonIssuesPanelProps {
  make: string
  model: string
  /** A jármű évjárata — csak az opcionális évjárat-szűrő kapcsolóhoz kell. */
  vehicleYear?: number
}

export default function CommonIssuesPanel({ make, model, vehicleYear }: CommonIssuesPanelProps) {
  // undefined = minden évjárat. Ez az alapértelmezés: a végpont évszűrő nélkül
  // lényegesen több bejelentést összesít.
  const [yearFilter, setYearFilter] = useState<number | undefined>(undefined)

  const { data, isLoading, isError, error, refetch } = useVehicleCommonIssues(
    make,
    model,
    yearFilter
  )

  const components = data?.components ?? []
  const issues = data?.issues ?? []
  const totalComplaints = data?.total_complaints ?? 0

  // A `sources` POZITÍV jelzés: kizárólag az `'unavailable'` érték jelenti azt,
  // hogy a forrás nem válaszolt. Minden más — beleértve a mező hiányát egy régi
  // backend-válaszban — azt jelenti, hogy a forrás felelt, vagyis pontosan a
  // mező bevezetése előtti viselkedést kapjuk vissza.
  const componentsUnavailable = data?.sources?.components === 'unavailable'
  const issuesUnavailable = data?.sources?.issues === 'unavailable'

  // ── Évjárat kapcsoló ────────────────────────────────────────────────────────

  const yearToggle = vehicleYear ? (
    <div
      className="flex gap-1 p-1 bg-slate-100 rounded-lg w-fit"
      role="group"
      aria-label="Évjárat szűrő"
    >
      <button
        type="button"
        onClick={() => setYearFilter(undefined)}
        aria-pressed={yearFilter === undefined}
        className={`h-8 px-3 rounded-md text-xs font-semibold transition-colors ${focusRing} ${
          yearFilter === undefined
            ? 'bg-white text-slate-900 shadow-sm'
            : 'text-slate-600 hover:text-slate-800'
        }`}
      >
        Minden évjárat
      </button>
      <button
        type="button"
        onClick={() => setYearFilter(vehicleYear)}
        aria-pressed={yearFilter === vehicleYear}
        className={`h-8 px-3 rounded-md text-xs font-semibold transition-colors ${focusRing} ${
          yearFilter === vehicleYear
            ? 'bg-white text-slate-900 shadow-sm'
            : 'text-slate-600 hover:text-slate-800'
        }`}
      >
        Csak {vehicleYear}
      </button>
    </div>
  ) : null

  // ── Betöltés ────────────────────────────────────────────────────────────────

  if (isLoading) {
    return (
      <div
        role="status"
        className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-3"
      >
        <span className="sr-only">Gyakori hibák betöltése…</span>
        <Skeleton className="h-5 w-48 rounded-lg" />
        <Skeleton className="h-12 w-full rounded-xl" />
        <Skeleton className="h-12 w-full rounded-xl" />
        <Skeleton className="h-12 w-full rounded-xl" />
      </div>
    )
  }

  // ── Hiba ────────────────────────────────────────────────────────────────────

  if (isError) {
    return (
      <ErrorState
        error={error}
        title="Nem sikerült betölteni a gyakori hibákat"
        onRetry={() => {
          void refetch()
        }}
      />
    )
  }

  // ── Alkatrészcsoport-rangsor és üres állapotai ──────────────────────────────

  const hasComponents = components.length > 0

  /**
   * A `components` szekció: rangsor, vagy a hozzá tartozó üres/kiesett állapot.
   *
   * Szándékosan NEM korai `return` a komponensből: az `issues` lista önálló
   * forrásból jön, és akkor is meg kell jelennie, ha ez a szekció üres.
   */
  const renderComponents = () => {
    if (!hasComponents && componentsUnavailable) {
      // A forrás nem válaszolt. Az üres lista itt az adat ELÉRÉSÉNEK hiánya, nem
      // az adaté — tilos bármit tényként állítani a járműről.
      return (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
          {yearToggle}
          <ErrorState
            type="server"
            title="A bejelentési adatok most nem elérhetők"
            message="A bejelentéseket kiszolgáló adatbázis nem válaszolt, ezért most nem tudjuk megmutatni, mit jelentettek erről a járműről. Ez nem jelenti azt, hogy nincs adat — próbáld újra kicsit később."
            onRetry={() => {
              void refetch()
            }}
          />
        </div>
      )
    }

    // Évjáratra szűrve nincs találat, de az összes évjárat nézetben lehet.
    if (!hasComponents && yearFilter !== undefined) {
      return (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6 space-y-4">
          {yearToggle}
          <EmptyState
            icon={<Car className="h-6 w-6 text-muted-foreground" aria-hidden="true" />}
            title={`A ${yearFilter}. évjáratra nincs bejelentés`}
            description="Más évjáratokról viszont lehet adat — nézd meg az összesített képet."
            action={{
              label: 'Minden évjárat megtekintése',
              onClick: () => setYearFilter(undefined),
            }}
          />
        </div>
      )
    }

    // A tényleges "nincs adat" eset — a forrás válaszolt, csak nincs mit mondania.
    // Ez NEM hiba: a korpusz amerikai piaci NHTSA-adat, és a választható márkák
    // egy része sosem került ki az USA-ba.
    if (!hasComponents) {
      return (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm p-6">
          <EmptyState
            icon={<Car className="h-6 w-6 text-muted-foreground" aria-hidden="true" />}
            title="Erre a járműre nincs NHTSA-adat"
            description="Az adatbázis amerikai piaci fogyasztói bejelentésekből áll, ez a modell pedig ott nem volt forgalomban."
          />
          <p className="mx-auto max-w-lg text-center text-sm text-slate-600 leading-relaxed">
            Ez nem hiba és nem is jelent hibamentes járművet — csak annyit jelent, hogy erről a
            modellről nincs amerikai bejelentés az adatbázisunkban.
          </p>
        </div>
      )
    }

    return (
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        {/* Fejléc + kontextus */}
        <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 id="common-issues-heading" className="text-base font-bold text-slate-900">
              Leggyakrabban bejelentett alkatrészcsoportok
            </h2>
            <p className="text-sm text-slate-600 mt-0.5">
              {formatCount(totalComplaints)} bejelentés a mintában
              {/* Sorszámos alak ("2018. évjárat"): a toldalékos "-as/-es/-ös"
                  változat évszámonként más lenne (2018-as, 2015-ös, 2016-os). */}
              {yearFilter === undefined ? ' · minden évjárat' : ` · ${yearFilter}. évjárat`}
            </p>
          </div>
          {yearToggle}
        </div>

        {/* Rangsor */}
        <ol className="divide-y divide-slate-100">
          {components.map((component, idx) => (
            <li key={component.component} className="px-6 py-4">
              <div className="flex items-baseline justify-between gap-3">
                <div className="flex items-baseline gap-2.5 min-w-0">
                  <span className="text-xs font-bold text-slate-500 tabular-nums flex-shrink-0">
                    {idx + 1}.
                  </span>
                  <span className="text-sm font-semibold text-slate-900 break-words">
                    {componentLabel(component)}
                  </span>
                </div>
                <span className="text-sm font-black text-slate-900 tabular-nums flex-shrink-0">
                  {formatShare(component.share)}
                </span>
              </div>

              {/* Arány-sáv: a szélesség maga a `share`, nem normalizált skála. */}
              <div className="h-2 mt-2 rounded-full bg-slate-100 overflow-hidden" aria-hidden="true">
                <div
                  className="h-full rounded-full bg-[#2563eb]"
                  style={{ width: barWidth(component.share) }}
                />
              </div>

              <p className="text-xs text-slate-600 mt-1.5">
                {formatCount(component.complaint_count)} bejelentés
              </p>

              <SafetySignals component={component} />
            </li>
          ))}
        </ol>

        {/* Lábjegyzet — mit is mutatnak pontosan ezek a számok */}
        <div className="px-6 py-4 border-t border-slate-100 bg-slate-50">
          <p className="text-xs text-slate-600 leading-relaxed">
            A számok az adatbázisunkba importált NHTSA-bejelentések mintáján alapulnak (gyártónként
            és alkatrészcsoportonként korlátozott darabszám, a régebbi évjáratok ritkítva), ezért nem
            a NHTSA teljes bejelentésszámát mutatják. Az arány a járműre tárolt{' '}
            {formatCount(totalComplaints)} bejelentéshez viszonyítva értendő.
          </p>
        </div>
      </div>
    )
  }

  // ── Tartalom ────────────────────────────────────────────────────────────────

  // A szekció neve a rangsor címéből jön, ha az látszik; egyébként állandó
  // felirat, hogy soha ne maradjon lógó `aria-labelledby` hivatkozás.
  const sectionLabel = hasComponents
    ? { 'aria-labelledby': 'common-issues-heading' }
    : { 'aria-label': 'Gyakori hibák' }

  return (
    <section className="space-y-4" {...sectionLabel}>
      {renderComponents()}

      {/* ── Hibakódok (másodlagos) ──────────────────────────────────────────────
          A gráfban kevés a panasz→DTC kapcsolat, ezért ez a szekció a legtöbb
          járműnél egyáltalán nem jelenik meg. Üresen nem hagy maga után se
          címet, se keretet — nem néz ki törött szekciónak.

          Saját forrásból jön, ezért a `components` üressége (vagy kiesése) nem
          nyomja el: ha van kód, itt megjelenik. */}
      {issues.length > 0 && (
        <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-100">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Wrench className="h-4 w-4 text-slate-500" aria-hidden="true" />
              Bejelentésekben említett hibakódok
            </h3>
            <p className="text-sm text-slate-600 mt-0.5">
              Fogyasztói leírásokból kinyert DTC-kódok — csak töredékük említ konkrét kódot.
            </p>
          </div>
          <ul className="divide-y divide-slate-100">
            {issues.map((issue) => {
              const severityChip = issue.severity ? SEVERITY_CHIPS[issue.severity] : undefined
              const frequencyLabel = issue.frequency
                ? FREQUENCY_LABELS[issue.frequency]
                : undefined
              const description = issue.description_hu || issue.description_en

              return (
                <li key={issue.code} className="px-6 py-3">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="inline-flex items-center px-2.5 py-1 rounded-lg bg-slate-100 text-slate-700 text-xs font-mono font-bold">
                      {issue.code}
                    </span>
                    {severityChip && (
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase ${severityChip.className}`}
                      >
                        {severityChip.label}
                      </span>
                    )}
                    {frequencyLabel && (
                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-slate-100 text-slate-700">
                        {frequencyLabel}
                      </span>
                    )}
                    {issue.occurrence_count != null && (
                      <span className="text-xs text-slate-600">
                        {formatCount(issue.occurrence_count)} említés
                      </span>
                    )}
                  </div>
                  {description && (
                    <p className="text-sm text-slate-700 mt-1.5 leading-relaxed">{description}</p>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      )}

      {/* A hibakód-forrás kiesése: a hiányzó szekció önmagában semmit nem állít,
          de az sem igaz, hogy nincs kód — ezt egy sorban, őszintén jelezzük.
          Ha a másik forrás IS kiesett, a fenti nagy állapot már elmondta
          ugyanezt, ezért ott nem ismételjük meg. */}
      {issues.length === 0 && issuesUnavailable && !componentsUnavailable && (
        <ErrorState
          compact
          type="server"
          title="A hibakódok most nem elérhetők"
          message="A kódokat kiszolgáló adatbázis nem válaszolt."
          onRetry={() => {
            void refetch()
          }}
        />
      )}
    </section>
  )
}
