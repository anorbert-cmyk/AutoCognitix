/**
 * HistoryPage - Exact match to provided HTML design
 * Primary: #0055d4, Font: Space Grotesk
 */

import { useState, type MouseEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  BarChart3,
  Zap,
  PlusCircle,
  Trash2,
  History,
} from 'lucide-react';

import {
  useDiagnosisHistory,
  useDiagnosisStats,
  useDeleteDiagnosis,
} from '@/services/hooks';
import { formatDate, type HistoryParams } from '@/services/diagnosisService';
import { EmptyState, Skeleton } from '@/components/ui';
import { useToast } from '../contexts/ToastContext';

const PAGE_SIZE = 10;

interface HistoryFilters {
  dtcCode: string;
  vehicleMake: string;
  dateFrom: string;
  dateTo: string;
}

const EMPTY_FILTERS: HistoryFilters = {
  dtcCode: '',
  vehicleMake: '',
  dateFrom: '',
  dateTo: '',
};

export default function HistoryPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const del = useDeleteDiagnosis();

  // Draft holds live input values; applied is what actually drives the query.
  // Splitting them means typing never refetches — only "Szűrők alkalmazása" does.
  const [draft, setDraft] = useState<HistoryFilters>(EMPTY_FILTERS);
  const [applied, setApplied] = useState<HistoryFilters>(EMPTY_FILTERS);
  const [page, setPage] = useState(1);

  // Server-driven params. Inline object is fine — useDiagnosisHistory hashes
  // the params into its query key, so a new reference alone won't refetch.
  const params: HistoryParams = {
    skip: (page - 1) * PAGE_SIZE,
    limit: PAGE_SIZE,
    ...(applied.vehicleMake.trim() && { vehicleMake: applied.vehicleMake.trim() }),
    ...(applied.dtcCode.trim() && { dtcCode: applied.dtcCode.trim().toUpperCase() }),
    ...(applied.dateFrom && { dateFrom: applied.dateFrom }),
    // Inclusive end-of-day: backend filters created_at <= date_to.
    ...(applied.dateTo && { dateTo: `${applied.dateTo}T23:59:59` }),
  };

  const { data, isLoading, error, refetch } = useDiagnosisHistory(params);
  const rows = data?.items ?? [];

  const { data: apiStats, isLoading: statsLoading } = useDiagnosisStats();
  const stats = apiStats
    ? {
        totalDiagnoses: apiStats.total_diagnoses,
        aiAccuracy: apiStats.avg_confidence * 100,
      }
    : null;

  // Server-authoritative pagination.
  const total = data?.total ?? 0;
  const hasMore = data?.has_more ?? false;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const windowStart = Math.min(Math.max(1, page - 1), Math.max(1, totalPages - 2));
  const pageWindow = [windowStart, windowStart + 1, windowStart + 2].filter(
    (p) => p <= totalPages,
  );

  const hasActiveFilter = !!(
    applied.dtcCode ||
    applied.vehicleMake ||
    applied.dateFrom ||
    applied.dateTo
  );

  const applyFilters = () => {
    setApplied(draft);
    setPage(1);
  };

  const handleDelete = async (id: string, e: MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Biztosan törli ezt a diagnózist?')) return;
    try {
      await del.mutateAsync(id);
      toast.success('Diagnózis törölve');
      // Fell off the last row of a non-first page → step back so we don't land
      // on an empty page.
      if (rows.length === 1 && page > 1) setPage((p) => p - 1);
    } catch {
      toast.error('A törlés sikertelen');
    }
  };

  return (
    <div className="bg-white text-slate-900">
      {/* Main Content — page chrome (header/nav/footer) is provided by Layout */}
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
        {/* Page Header */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-6 mb-8">
          <div className="flex flex-col gap-1">
            <h1 className="text-slate-900 text-4xl font-black leading-tight tracking-tight uppercase italic">
              Műhely Előzmények
            </h1>
            <p className="text-slate-600 text-base font-medium">
              Diagnosztikai rekordok ellenőrzése és áttekintése MI-alapú pontossággal.
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/diagnosis')}
              className="flex items-center justify-center rounded-lg h-11 bg-[#0055d4] text-white px-6 font-bold uppercase tracking-widest text-xs gap-2 hover:bg-blue-700 transition-all shadow-lg shadow-blue-200"
            >
              <PlusCircle className="w-4 h-4" />
              Új bejegyzés
            </button>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="bg-white rounded-xl border border-slate-300 p-6 mb-8 shadow-sm">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Search Input (whole DTC code) */}
            <div className="lg:col-span-4">
              <label className="flex flex-col w-full h-12">
                <div className="flex w-full flex-1 items-stretch rounded-lg overflow-hidden border border-slate-300 group focus-within:border-[#0055d4] focus-within:ring-1 focus-within:ring-[#0055d4]">
                  <div className="text-slate-500 flex bg-slate-50 items-center justify-center px-4 border-r border-slate-300">
                    <Search className="w-5 h-5" />
                  </div>
                  <input
                    type="text"
                    value={draft.dtcCode}
                    onChange={(e) => setDraft((d) => ({ ...d, dtcCode: e.target.value }))}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') applyFilters();
                    }}
                    className="flex w-full min-w-0 flex-1 border-none bg-white text-slate-900 placeholder:text-slate-500 px-4 focus:ring-0 text-base font-medium"
                    placeholder="Keresés hibakód alapján (pl. P0301)"
                    aria-label="Keresés hibakód alapján"
                  />
                </div>
              </label>
            </div>

            {/* Manufacturer Filter */}
            <div className="lg:col-span-2">
              <label className="flex flex-col w-full h-12">
                <div className="flex w-full flex-1 items-stretch rounded-lg overflow-hidden border border-slate-300 group focus-within:border-[#0055d4] focus-within:ring-1 focus-within:ring-[#0055d4]">
                  <div className="text-slate-500 flex bg-slate-50 items-center justify-center px-3 border-r border-slate-300">
                    <Filter className="w-5 h-5" />
                  </div>
                  <input
                    type="text"
                    value={draft.vehicleMake}
                    onChange={(e) => setDraft((d) => ({ ...d, vehicleMake: e.target.value }))}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') applyFilters();
                    }}
                    className="flex w-full min-w-0 flex-1 border-none bg-white text-slate-900 placeholder:text-slate-500 px-3 focus:ring-0 text-sm font-medium"
                    placeholder="Gyártó"
                    aria-label="Gyártó szűrő"
                  />
                </div>
              </label>
            </div>

            {/* Date From */}
            <div className="lg:col-span-2">
              <span className="mb-1 block text-xs font-bold uppercase tracking-wider text-slate-500">
                Kezdő dátum
              </span>
              <input
                type="date"
                value={draft.dateFrom}
                max={draft.dateTo || undefined}
                onChange={(e) => setDraft((d) => ({ ...d, dateFrom: e.target.value }))}
                className="w-full h-12 rounded-lg border border-slate-300 bg-white text-slate-700 px-3 text-sm font-medium focus:border-[#0055d4] focus:ring-1 focus:ring-[#0055d4]"
                aria-label="Kezdő dátum"
              />
            </div>

            {/* Date To */}
            <div className="lg:col-span-2">
              <span className="mb-1 block text-xs font-bold uppercase tracking-wider text-slate-500">
                Záró dátum
              </span>
              <input
                type="date"
                value={draft.dateTo}
                min={draft.dateFrom || undefined}
                onChange={(e) => setDraft((d) => ({ ...d, dateTo: e.target.value }))}
                className="w-full h-12 rounded-lg border border-slate-300 bg-white text-slate-700 px-3 text-sm font-medium focus:border-[#0055d4] focus:ring-1 focus:ring-[#0055d4]"
                aria-label="Záró dátum"
              />
            </div>

            {/* Apply Filters Button */}
            <div className="lg:col-span-2">
              <button
                onClick={applyFilters}
                className="flex w-full h-12 items-center justify-center rounded-lg bg-slate-900 text-white px-4 font-bold uppercase text-xs tracking-widest hover:bg-slate-800 transition-colors shadow-md"
              >
                Szűrők alkalmazása
              </button>
            </div>
          </div>
        </div>

        {/* Table */}
        <div className="overflow-hidden rounded-xl border border-slate-300 bg-white shadow-xl shadow-slate-200/50">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-100 border-b border-slate-300">
                  <th scope="col" className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Alvázszám
                  </th>
                  <th scope="col" className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Jármű adatok
                  </th>
                  <th scope="col" className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Hibakód
                  </th>
                  <th scope="col" className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Fő tünet
                  </th>
                  <th scope="col" className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Diagnosztika dátuma
                  </th>
                  <th
                    scope="col"
                    className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600 text-right"
                  >
                    Művelet
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {isLoading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12" role="status">
                      <span className="sr-only">Betöltés...</span>
                      <div className="space-y-3">
                        <Skeleton className="h-10 w-full" />
                        <Skeleton className="h-10 w-full" />
                        <Skeleton className="h-10 w-full" />
                      </div>
                    </td>
                  </tr>
                ) : error ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center">
                      <p className="text-red-600 font-bold mb-3">
                        Hiba történt az adatok betöltésekor. Próbálja újra!
                      </p>
                      <button
                        onClick={() => refetch()}
                        className="inline-flex items-center justify-center rounded-lg h-10 bg-[#0055d4] text-white px-5 font-bold uppercase tracking-widest text-xs hover:bg-blue-700 transition-all shadow-lg shadow-blue-200"
                      >
                        Újrapróbálás
                      </button>
                    </td>
                  </tr>
                ) : rows.length === 0 && hasActiveFilter ? (
                  <tr>
                    <td colSpan={6}>
                      <EmptyState title="Nincs a szűrőknek megfelelő találat." />
                    </td>
                  </tr>
                ) : rows.length === 0 ? (
                  <tr>
                    <td colSpan={6}>
                      <EmptyState
                        icon={<History className="h-6 w-6 text-muted-foreground" aria-hidden="true" />}
                        title="Még nincs diagnosztikai előzmény."
                        action={{ label: 'Új diagnózis indítása', to: '/diagnosis' }}
                      />
                    </td>
                  </tr>
                ) : (
                  rows.map((item) => (
                    <tr
                      key={item.id}
                      className="hover:bg-blue-50 transition-colors group cursor-pointer"
                      onClick={() => navigate(`/diagnosis/${item.id}`)}
                    >
                      <td className="px-6 py-5 text-slate-700 font-medium">
                        {item.vehicle_vin || '—'}
                      </td>
                      <td className="px-6 py-5">
                        <div className="flex flex-col">
                          <span className="text-slate-900 font-bold">
                            {item.vehicle_make} {item.vehicle_model}
                          </span>
                          <span className="text-slate-600 text-xs font-medium">
                            {item.vehicle_year}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-5">
                        <div className="flex items-center gap-2">
                          <span className="bg-white text-[#0055d4] border border-[#0055d4]/30 px-3 py-1 rounded font-mono font-bold text-sm shadow-sm">
                            {item.dtc_codes[0] ?? '—'}
                          </span>
                          {item.dtc_codes.length > 1 && (
                            <span className="text-[10px] font-black uppercase tracking-widest text-slate-500 bg-slate-100 border border-slate-200 px-1.5 py-0.5 rounded">
                              +{item.dtc_codes.length - 1}
                            </span>
                          )}
                        </div>
                      </td>
                      <td
                        className="px-6 py-5 text-slate-700 max-w-xs truncate font-medium"
                        title={item.symptoms_text}
                      >
                        {item.symptoms_text || '—'}
                      </td>
                      <td className="px-6 py-5 text-slate-700 font-medium">
                        {formatDate(item.created_at)}
                      </td>
                      <td className="px-6 py-5 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              navigate(`/diagnosis/${item.id}`);
                            }}
                            className="text-[#0055d4] hover:text-white hover:bg-[#0055d4] font-bold uppercase text-[10px] tracking-widest border border-[#0055d4] px-3 py-1.5 rounded-lg transition-all"
                          >
                            Részletek
                          </button>
                          <button
                            onClick={(e) => handleDelete(item.id, e)}
                            disabled={del.isPending}
                            aria-label={`Törlés: ${item.vehicle_make} ${item.vehicle_model}`}
                            className="text-slate-400 hover:text-white hover:bg-red-500 border border-slate-300 hover:border-red-500 p-1.5 rounded-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                          >
                            <Trash2 className="w-4 h-4" aria-hidden="true" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="px-6 py-4 bg-slate-50 border-t border-slate-300 flex items-center justify-between">
            <p className="text-slate-600 text-xs font-bold uppercase">
              Megjelenítve: {rows.length} / {total.toLocaleString()} rekord
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page <= 1}
                aria-label="Előző oldal"
                className="w-8 h-8 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-600 hover:bg-slate-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              {pageWindow.map((p) => (
                <button
                  key={p}
                  onClick={() => setPage(p)}
                  aria-current={p === page ? 'page' : undefined}
                  className={
                    p === page
                      ? 'w-8 h-8 flex items-center justify-center rounded border border-[#0055d4] bg-[#0055d4] text-white font-bold text-xs shadow-md'
                      : 'w-8 h-8 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-600 font-bold text-xs hover:bg-slate-100 transition-colors'
                  }
                >
                  {p}
                </button>
              ))}
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={!hasMore}
                aria-label="Következő oldal"
                className="w-8 h-8 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-600 hover:bg-slate-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Total Diagnoses */}
          <div className="bg-white border border-slate-300 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <BarChart3 className="w-6 h-6 text-[#0055d4]" />
              <span className="text-xs font-black uppercase tracking-widest text-slate-500">
                Összes diagnosztika
              </span>
            </div>
            <div className="text-3xl font-black text-slate-900 uppercase italic">
              {statsLoading ? '...' : (stats?.totalDiagnoses.toLocaleString() ?? '—')}
            </div>
          </div>

          {/* AI Confidence */}
          <div className="bg-white border border-slate-300 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <Zap className="w-6 h-6 text-yellow-600" />
              <span className="text-xs font-black uppercase tracking-widest text-slate-500">
                Átlagos AI konfidencia
              </span>
            </div>
            <div className="text-3xl font-black text-slate-900 uppercase italic">
              {statsLoading ? '...' : (stats ? `${stats.aiAccuracy.toFixed(1)}%` : '—')}
            </div>
            <div className="mt-2 text-xs font-bold text-slate-600">
              Korábbi diagnózisok alapján
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
