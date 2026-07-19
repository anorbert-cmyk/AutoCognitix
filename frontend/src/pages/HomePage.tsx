import { Link } from 'react-router-dom'
import {
  ArrowRight,
  Bell,
  Calculator,
  Car,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  Eye,
  History,
  MapPin,
  MessageSquare,
  Plus,
  Shield,
  Sparkles,
  Wrench,
  Zap,
} from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useDiagnosisHistory, useUpcomingReminders, useVehicles } from '@/services/hooks'
import { cn } from '@/lib/utils'

// Shared focus ring for links/buttons on light surfaces
const focusRing =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'
// Focus ring variant for elements sitting on the dark hero gradient
const focusRingDark =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-primary-800'

export default function HomePage() {
  const { isAuthenticated, isLoading } = useAuth()

  // Avoid a marketing-page flash while the session is being restored
  if (isLoading) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8" role="status" aria-busy="true">
        <span className="sr-only">Betöltés…</span>
        <div className="h-44 animate-pulse rounded-2xl bg-muted" aria-hidden="true" />
      </div>
    )
  }

  return isAuthenticated ? <DashboardHome /> : <MarketingHome />
}

// =============================================================================
// Marketing view (logged out)
// =============================================================================

// Real platform numbers — see CLAUDE.md (Neo4j 26 816 nodes, Qdrant 35 000+ vectors)
const PLATFORM_STATS = [
  { value: '26 816', label: 'diagnosztikai csomópont' },
  { value: '35 000+', label: 'szemantikus vektor' },
  { value: '768 dim', label: 'magyar HuBERT modell' },
  { value: '0 Ft', label: 'hardverigény' },
]

const HOW_IT_WORKS = [
  {
    step: '01',
    icon: Wrench,
    title: 'Hibakód bevitel',
    text: 'Adja meg az OBD-II hibakódokat és írja le a tapasztalt tüneteket magyarul.',
  },
  {
    step: '02',
    icon: Zap,
    title: 'AI elemzés',
    text: 'A rendszer elemzi az adatokat TSB-k, fórum-bejegyzések és szakértői tudás alapján.',
  },
  {
    step: '03',
    icon: Shield,
    title: 'Részletes diagnózis',
    text: 'Kapjon részletes diagnózist lehetséges okokkal és javítási javaslatokkal.',
  },
]

const TOOL_CARDS = [
  {
    icon: ClipboardCheck,
    title: 'Műszaki Vizsga Felkészítő',
    text: 'DTC kódok alapján elemezze, milyen kockázattal járna a műszaki vizsgán.',
    to: '/inspection',
  },
  {
    icon: Calculator,
    title: 'Megéri Megjavítani?',
    text: 'Hasonlítsa össze a javítási költséget a jármű piaci értékével.',
    to: '/calculator',
  },
  {
    icon: MessageSquare,
    title: 'AI Chat Asszisztens',
    text: 'Beszélgessen az AI-val magyarul a jármű problémáiról.',
    to: '/chat',
  },
  {
    icon: MapPin,
    title: 'Szerviz Összehasonlítás',
    text: 'Keresse meg a legközelebbi szervizeket interaktív térképen.',
    to: '/services',
  },
]

function MarketingHome() {
  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-br from-primary-700 via-primary-800 to-primary-950 text-white">
        <div aria-hidden="true" className="pointer-events-none absolute inset-0">
          <div className="absolute -right-24 -top-32 h-96 w-96 rounded-full bg-primary-400/20 blur-3xl" />
          <div className="absolute -bottom-40 -left-24 h-96 w-96 rounded-full bg-primary-300/10 blur-3xl" />
          <div className="absolute inset-0 opacity-[0.06] bg-[radial-gradient(rgba(255,255,255,0.9)_1px,transparent_1px)] [background-size:28px_28px]" />
        </div>

        <div className="relative mx-auto max-w-7xl px-4 py-24 sm:px-6 lg:px-8 lg:py-28">
          <div className="text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-white/10 px-4 py-1.5 text-sm font-medium text-primary-100 backdrop-blur-sm">
              <Sparkles className="h-4 w-4" aria-hidden="true" />
              <span className="text-white">Magyar nyelvű AI-diagnosztika</span>
            </span>

            <h1 className="mt-6 text-4xl font-bold tracking-tight md:text-6xl">
              AI-alapú Gépjármű Diagnosztika
            </h1>
            <p className="mx-auto mt-6 max-w-3xl text-xl text-primary-100 md:text-2xl">
              Gyors és pontos hibakód-elemzés mesterséges intelligenciával. Hardver nélkül,
              bárhonnan elérhető.
            </p>

            <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Link
                to="/diagnosis"
                className={cn(
                  'inline-flex items-center gap-2 rounded-lg bg-white px-8 py-4 text-lg font-semibold text-primary-700',
                  'shadow-lg shadow-primary-950/25 transition-colors duration-150 hover:bg-primary-50',
                  focusRingDark
                )}
              >
                Diagnózis indítása
                <ArrowRight className="h-5 w-5" aria-hidden="true" />
              </Link>
              <Link
                to="/demo"
                className={cn(
                  'inline-flex items-center gap-2 rounded-lg border border-white/40 bg-white/10 px-8 py-4 text-lg font-semibold text-white',
                  'backdrop-blur-sm transition-colors duration-150 hover:bg-white/20',
                  focusRingDark
                )}
              >
                <Eye className="h-5 w-5" aria-hidden="true" />
                Demo megtekintése
              </Link>
            </div>
          </div>

          {/* Honest platform stats */}
          <div className="mt-16 grid grid-cols-2 gap-y-8 border-t border-white/15 pt-8 sm:grid-cols-4">
            {PLATFORM_STATS.map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-2xl font-bold tracking-tight md:text-3xl">{stat.value}</p>
                <p className="mt-1 text-sm text-primary-200">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="mb-16 text-center">
            <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
              Hogyan működik?
            </h2>
            <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
              Egyszerűen adja meg a hibakódokat és a tüneteket, az AI elvégzi az elemzést.
            </p>
          </div>

          <ol className="grid gap-8 md:grid-cols-3">
            {HOW_IT_WORKS.map(({ step, icon: Icon, title, text }) => (
              <li key={step} className="card relative p-8">
                <span
                  className="absolute right-6 top-6 font-mono text-sm font-semibold tracking-widest text-primary-500"
                  aria-hidden="true"
                >
                  {step}
                </span>
                <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-primary-50 ring-1 ring-primary-100">
                  <Icon className="h-7 w-7 text-primary-600" aria-hidden="true" />
                </div>
                <h3 className="mb-3 text-xl font-semibold text-foreground">{title}</h3>
                <p className="text-muted-foreground">{text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* Tools */}
      <section className="bg-muted/40 py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="mb-16 text-center">
            <h2 className="mb-4 text-3xl font-bold text-foreground md:text-4xl">
              Eszközök a teljes folyamathoz
            </h2>
            <p className="mx-auto max-w-2xl text-lg text-muted-foreground">
              Fedezze fel az eszközöket a jármű karbantartásához és javításához.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {TOOL_CARDS.map(({ icon: Icon, title, text, to }) => (
              <Link
                key={to}
                to={to}
                className={cn(
                  'card group p-8 text-center transition-all duration-150',
                  'hover:border-primary-200 hover:shadow-lg',
                  focusRing
                )}
              >
                <div className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-primary-50 ring-1 ring-primary-100 transition-transform duration-150 group-hover:scale-105 motion-reduce:transition-none">
                  <Icon className="h-7 w-7 text-primary-600" aria-hidden="true" />
                </div>
                <h3 className="mb-3 text-xl font-semibold text-foreground">{title}</h3>
                <p className="mb-4 text-muted-foreground">{text}</p>
                <ArrowRight
                  className="mx-auto h-5 w-5 text-primary-600 opacity-0 transition-opacity duration-150 group-hover:opacity-100"
                  aria-hidden="true"
                />
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-primary-700 via-primary-800 to-primary-950 px-8 py-16 text-center shadow-xl md:px-12">
            <div aria-hidden="true" className="pointer-events-none absolute inset-0">
              <div className="absolute -right-20 -top-24 h-72 w-72 rounded-full bg-primary-400/20 blur-3xl" />
              <div className="absolute inset-0 opacity-[0.06] bg-[radial-gradient(rgba(255,255,255,0.9)_1px,transparent_1px)] [background-size:28px_28px]" />
            </div>
            <div className="relative">
              <h2 className="mb-4 text-3xl font-bold text-white">
                Készen áll a diagnosztikára?
              </h2>
              <p className="mx-auto mb-8 max-w-2xl text-lg text-primary-100">
                Nem kell drága hardver - csak adja meg a hibakódokat és a tüneteket. Az AI segít
                megtalálni a problémát.
              </p>
              <Link
                to="/diagnosis"
                className={cn(
                  'inline-flex items-center gap-2 rounded-lg bg-white px-8 py-4 text-lg font-semibold text-primary-700',
                  'shadow-lg shadow-primary-950/25 transition-colors duration-150 hover:bg-primary-50',
                  focusRingDark
                )}
              >
                Diagnózis indítása
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

// =============================================================================
// Dashboard view (logged in)
// =============================================================================

const DASHBOARD_TOOLS = [
  { icon: ClipboardCheck, label: 'Műszaki vizsga', to: '/inspection' },
  { icon: Calculator, label: 'Költségkalkulátor', to: '/calculator' },
  { icon: MapPin, label: 'Szervizkereső', to: '/services' },
  { icon: History, label: 'Előzmények', to: '/history' },
]

function DashboardHome() {
  const { user } = useAuth()
  const { data: vehiclesData, isLoading: vehiclesLoading } = useVehicles()
  const { data: historyData, isLoading: historyLoading } = useDiagnosisHistory({ limit: 3 })
  const { data: upcomingReminders, isLoading: remindersLoading } = useUpcomingReminders(14)

  const vehicles = (vehiclesData?.vehicles ?? []).slice(0, 3)
  const recentDiagnoses = historyData?.items ?? []
  const reminders = (upcomingReminders ?? []).slice(0, 3)

  // Hungarian name order: family name first → the given name is the last token
  const fullName = user?.full_name?.trim()
  const firstName = fullName
    ? fullName.split(/\s+/).slice(-1)[0]
    : (user?.email?.split('@')[0] ?? '')

  return (
    <div>
      {/* Greeting band */}
      <section className="border-b border-border bg-gradient-to-b from-primary-50/70 to-background">
        <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 md:py-14 lg:px-8">
          <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wider text-primary-600">
                Vezérlőpult
              </p>
              <h1 className="mt-1 text-3xl font-bold tracking-tight text-foreground md:text-4xl">
                Üdv újra{firstName ? `, ${firstName}` : ''}!
              </h1>
              <p className="mt-2 text-muted-foreground">
                A garázsa, a legutóbbi diagnózisok és a közelgő teendők egy helyen.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Link
                to="/diagnosis"
                className={cn(
                  'inline-flex items-center gap-2 rounded-lg bg-primary-600 px-5 py-3 text-sm font-semibold text-white',
                  'shadow-sm transition-colors duration-150 hover:bg-primary-700',
                  focusRing
                )}
              >
                <Plus className="h-4 w-4" aria-hidden="true" />
                Új diagnosztika
              </Link>
              <Link
                to="/chat"
                className={cn(
                  'inline-flex items-center gap-2 rounded-lg border border-border bg-card px-5 py-3 text-sm font-semibold text-foreground',
                  'transition-colors duration-150 hover:bg-muted',
                  focusRing
                )}
              >
                <MessageSquare className="h-4 w-4" aria-hidden="true" />
                AI Chat
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Cards */}
      <section className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Garage */}
          <DashboardCard
            icon={<Car className="h-5 w-5 text-primary-600" aria-hidden="true" />}
            title="Garázs"
            badge={vehiclesData ? `${vehiclesData.total} jármű` : undefined}
            footerTo="/garage"
            footerLabel="Garázs megnyitása"
          >
            {vehiclesLoading ? (
              <CardSkeleton rows={2} />
            ) : vehicles.length > 0 ? (
              <ul className="space-y-2">
                {vehicles.map((vehicle) => (
                  <li key={vehicle.id}>
                    <Link
                      to={`/garage/${vehicle.id}`}
                      className={cn(
                        'group flex items-center justify-between rounded-xl border border-border bg-background px-4 py-3',
                        'transition-colors duration-150 hover:border-primary-200 hover:bg-primary-50/50',
                        focusRing
                      )}
                    >
                      <span className="min-w-0">
                        <span className="block truncate font-medium text-foreground">
                          {vehicle.nickname || `${vehicle.make} ${vehicle.model}`}
                        </span>
                        <span className="block text-sm text-muted-foreground">
                          {vehicle.year}
                          {vehicle.license_plate ? ` · ${vehicle.license_plate}` : ''}
                        </span>
                      </span>
                      <ChevronRight
                        className="h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-150 group-hover:translate-x-0.5 motion-reduce:transition-none"
                        aria-hidden="true"
                      />
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <CardEmpty
                icon={<Car className="h-6 w-6 text-muted-foreground" aria-hidden="true" />}
                text="Még nincs jármű a garázsban."
                actionTo="/garage"
                actionLabel="Jármű hozzáadása"
              />
            )}
          </DashboardCard>

          {/* Recent diagnoses */}
          <DashboardCard
            icon={<History className="h-5 w-5 text-primary-600" aria-hidden="true" />}
            title="Friss diagnózisok"
            footerTo="/history"
            footerLabel="Összes előzmény"
          >
            {historyLoading ? (
              <CardSkeleton rows={2} />
            ) : recentDiagnoses.length > 0 ? (
              <ul className="space-y-2">
                {recentDiagnoses.map((item) => (
                  <li key={item.id}>
                    <Link
                      to={`/diagnosis/${item.id}`}
                      className={cn(
                        'group block rounded-xl border border-border bg-background px-4 py-3',
                        'transition-colors duration-150 hover:border-primary-200 hover:bg-primary-50/50',
                        focusRing
                      )}
                    >
                      <span className="flex items-center justify-between gap-3">
                        <span className="flex min-w-0 flex-wrap items-center gap-1.5">
                          {item.dtc_codes.slice(0, 3).map((code) => (
                            <span
                              key={code}
                              className="rounded-md bg-primary-50 px-2 py-0.5 font-mono text-xs font-semibold text-primary-700"
                            >
                              {code}
                            </span>
                          ))}
                          {item.dtc_codes.length > 3 && (
                            <span className="text-xs text-muted-foreground">
                              +{item.dtc_codes.length - 3}
                            </span>
                          )}
                        </span>
                        <ChevronRight
                          className="h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-150 group-hover:translate-x-0.5 motion-reduce:transition-none"
                          aria-hidden="true"
                        />
                      </span>
                      <span className="mt-1 block text-sm text-muted-foreground">
                        {item.vehicle_make} {item.vehicle_model} ·{' '}
                        {new Date(item.created_at).toLocaleDateString('hu-HU', {
                          year: 'numeric',
                          month: 'short',
                          day: 'numeric',
                        })}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            ) : (
              <CardEmpty
                icon={<History className="h-6 w-6 text-muted-foreground" aria-hidden="true" />}
                text="Még nincs diagnózis."
                actionTo="/diagnosis"
                actionLabel="Első diagnózis indítása"
              />
            )}
          </DashboardCard>

          {/* Upcoming reminders */}
          <DashboardCard
            icon={<Bell className="h-5 w-5 text-primary-600" aria-hidden="true" />}
            title="Közelgő teendők"
            footerTo="/garage"
            footerLabel="Emlékeztetők kezelése"
          >
            {remindersLoading ? (
              <CardSkeleton rows={2} />
            ) : reminders.length > 0 ? (
              <ul className="space-y-2">
                {reminders.map((reminder) => (
                  <li
                    key={reminder.id}
                    className={cn(
                      'rounded-xl border px-4 py-3',
                      reminder.urgency === 'overdue'
                        ? 'border-red-200 bg-red-50'
                        : reminder.urgency === 'urgent'
                          ? 'border-orange-200 bg-orange-50'
                          : 'border-yellow-200 bg-yellow-50'
                    )}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-foreground">
                          {reminder.title}
                        </p>
                        <p className="text-xs text-slate-600">
                          {reminder.days_until_due !== null && reminder.days_until_due !== undefined
                            ? reminder.days_until_due < 0
                              ? `${Math.abs(reminder.days_until_due)} napja lejárt`
                              : reminder.days_until_due === 0
                                ? 'Ma esedékes'
                                : `${reminder.days_until_due} nap múlva`
                            : (reminder.due_date ?? 'Határidő nincs megadva')}
                        </p>
                      </div>
                      <Link
                        to={`/garage/${reminder.vehicle_id}`}
                        aria-label={`Megtekint: ${reminder.title}`}
                        className={cn(
                          'shrink-0 px-1 py-2 -my-2 text-xs font-semibold text-primary-600 hover:text-primary-700 hover:underline',
                          focusRing,
                          'rounded'
                        )}
                      >
                        Megtekint
                      </Link>
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="flex flex-col items-center gap-2 py-8 text-center">
                <CheckCircle2 className="h-6 w-6 text-status-success" aria-hidden="true" />
                <p className="text-sm text-muted-foreground">
                  Nincs közelgő teendő a következő 14 napban.
                </p>
              </div>
            )}
          </DashboardCard>
        </div>

        {/* Tools row */}
        <div className="mt-8 grid grid-cols-2 gap-4 md:grid-cols-4">
          {DASHBOARD_TOOLS.map(({ icon: Icon, label, to }) => (
            <Link
              key={to}
              to={to}
              className={cn(
                'group flex items-center justify-between rounded-xl border border-border bg-card px-4 py-3.5',
                'transition-colors duration-150 hover:border-primary-200 hover:bg-primary-50/50',
                focusRing
              )}
            >
              <span className="flex items-center gap-3">
                <Icon className="h-5 w-5 text-primary-600" aria-hidden="true" />
                <span className="text-sm font-medium text-foreground">{label}</span>
              </span>
              <ChevronRight
                className="h-4 w-4 text-muted-foreground transition-transform duration-150 group-hover:translate-x-0.5 motion-reduce:transition-none"
                aria-hidden="true"
              />
            </Link>
          ))}
        </div>
      </section>
    </div>
  )
}

// =============================================================================
// Dashboard building blocks
// =============================================================================

interface DashboardCardProps {
  icon: React.ReactNode
  title: string
  badge?: string
  footerTo: string
  footerLabel: string
  children: React.ReactNode
}

function DashboardCard({ icon, title, badge, footerTo, footerLabel, children }: DashboardCardProps) {
  return (
    <div className="card flex flex-col rounded-2xl p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="flex items-center gap-2.5 text-lg font-semibold text-foreground">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-50">
            {icon}
          </span>
          {title}
        </h2>
        {badge && (
          <span className="rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
            {badge}
          </span>
        )}
      </div>
      <div className="flex-1">{children}</div>
      <div className="mt-4 border-t border-border pt-3">
        <Link
          to={footerTo}
          className={cn(
            'group inline-flex items-center gap-1 py-1 -my-1 text-sm font-semibold text-primary-600 hover:text-primary-700',
            focusRing,
            'rounded'
          )}
        >
          {footerLabel}
          <ChevronRight
            className="h-4 w-4 transition-transform duration-150 group-hover:translate-x-0.5 motion-reduce:transition-none"
            aria-hidden="true"
          />
        </Link>
      </div>
    </div>
  )
}

function CardSkeleton({ rows }: { rows: number }) {
  return (
    <div className="space-y-2" aria-hidden="true">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-14 animate-pulse rounded-xl bg-muted" />
      ))}
    </div>
  )
}

interface CardEmptyProps {
  icon: React.ReactNode
  text: string
  actionTo: string
  actionLabel: string
}

function CardEmpty({ icon, text, actionTo, actionLabel }: CardEmptyProps) {
  return (
    <div className="flex flex-col items-center gap-3 py-6 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
        {icon}
      </span>
      <p className="text-sm text-muted-foreground">{text}</p>
      <Link
        to={actionTo}
        className={cn(
          'inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-4 py-2 text-sm font-semibold text-foreground',
          'transition-colors duration-150 hover:border-primary-200 hover:bg-primary-50/50',
          focusRing
        )}
      >
        <Plus className="h-4 w-4" aria-hidden="true" />
        {actionLabel}
      </Link>
    </div>
  )
}
