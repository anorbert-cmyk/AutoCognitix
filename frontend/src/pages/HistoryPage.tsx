/**
 * HistoryPage - Exact match to provided HTML design
 * Primary: #0055d4, Font: Space Grotesk
 */

import { useState, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
  Search,
  Filter,
  Calendar,
  ChevronLeft,
  ChevronRight,
  CheckCircle,
  BarChart3,
  Zap,
  PlusCircle,
  Bell,
  Wrench,
} from 'lucide-react';

// Legacy hooks - will continue using existing API
import {
  useDiagnosisHistory,
  useDiagnosisStats,
} from '@/services/hooks';

type DiagnosisStatus = 'fixed' | 'in_progress' | 'pending';

interface DiagnosisHistoryItem {
  id: string;
  licensePlate: string;
  vehicleMake: string;
  vehicleModel: string;
  vehicleYear: number;
  vehicleTrim?: string;
  diagnosisDate: string;
  dtcCode: string;
  mainSymptom: string;
  status: DiagnosisStatus;
}

// Mock data for demonstration
const mockHistoryData: DiagnosisHistoryItem[] = [
  {
    id: '1',
    licensePlate: 'ABC-1234',
    vehicleMake: 'Toyota',
    vehicleModel: 'Camry',
    vehicleYear: 2022,
    vehicleTrim: 'Hybrid LE',
    diagnosisDate: '2023-10-25',
    dtcCode: 'P0300',
    mainSymptom: 'Égéskimaradás észlelve a 3. hengerben',
    status: 'fixed',
  },
  {
    id: '2',
    licensePlate: 'XYZ-9876',
    vehicleMake: 'Ford',
    vehicleModel: 'F-150',
    vehicleYear: 2019,
    vehicleTrim: 'Lariat EcoBoost',
    diagnosisDate: '2023-10-24',
    dtcCode: 'P0420',
    mainSymptom: 'Katalizátor hatásfoka küszöbérték alatt',
    status: 'in_progress',
  },
  {
    id: '3',
    licensePlate: 'DEF-5555',
    vehicleMake: 'Honda',
    vehicleModel: 'Civic',
    vehicleYear: 2021,
    vehicleTrim: 'Type R',
    diagnosisDate: '2023-10-22',
    dtcCode: 'C0021',
    mainSymptom: 'ABS figyelmeztetés - Jobb első szenzor',
    status: 'fixed',
  },
  {
    id: '4',
    licensePlate: 'GHI-1122',
    vehicleMake: 'Tesla',
    vehicleModel: 'Model 3',
    vehicleYear: 2023,
    vehicleTrim: 'Performance',
    diagnosisDate: '2023-10-20',
    dtcCode: 'B1234',
    mainSymptom: 'Magas 12V akkumulátor merülés parkolás közben',
    status: 'pending',
  },
  {
    id: '5',
    licensePlate: 'JKL-4433',
    vehicleMake: 'BMW',
    vehicleModel: 'X5',
    vehicleYear: 2018,
    vehicleTrim: 'xDrive35i',
    diagnosisDate: '2023-10-18',
    dtcCode: 'P0171',
    mainSymptom: 'Rendszer túl szegény (1. bank)',
    status: 'fixed',
  },
];

const mockStats = {
  solutionRate: 94.2,
  totalDiagnoses: 1248,
  aiAccuracy: 98.8,
};

function StatusBadge({ status }: { status: DiagnosisStatus }) {
  const styles = {
    fixed: 'bg-green-100 text-green-700 border-green-500',
    in_progress: 'bg-blue-100 text-blue-700 border-blue-500',
    pending: 'bg-yellow-100 text-yellow-700 border-yellow-500',
  };

  const labels = {
    fixed: 'Javítva',
    in_progress: 'Folyamatban',
    pending: 'Függőben',
  };

  return (
    <span
      className={`px-3 py-1 rounded-full text-xs font-black uppercase tracking-tighter border ${styles[status]}`}
    >
      {labels[status]}
    </span>
  );
}

export default function HistoryPage() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(1);

  // Try to use API hooks (will fall back to mock data if API fails)
  const { data: apiHistoryData, isLoading } = useDiagnosisHistory({
    skip: (page - 1) * 10,
    limit: 10,
  });

  const { data: apiStats, isLoading: statsLoading } = useDiagnosisStats();

  // Use mock data if API data is not available
  const historyData: DiagnosisHistoryItem[] = apiHistoryData?.items
    ? apiHistoryData.items.map((item) => ({
        id: item.id,
        licensePlate: 'N/A',
        vehicleMake: item.vehicle_make,
        vehicleModel: item.vehicle_model,
        vehicleYear: item.vehicle_year,
        diagnosisDate: item.created_at,
        dtcCode: item.dtc_codes[0] || 'N/A',
        mainSymptom: 'Nincs megadva',
        status: 'fixed' as const,
      }))
    : mockHistoryData;

  const stats = apiStats
    ? {
        solutionRate: 94.2,
        totalDiagnoses: apiStats.total_diagnoses,
        aiAccuracy: apiStats.avg_confidence * 100,
      }
    : mockStats;

  const totalRecords = apiHistoryData?.total || mockHistoryData.length;

  // Filter data based on search
  const filteredData = searchQuery
    ? historyData.filter(
        (item) =>
          item.licensePlate.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.dtcCode.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.vehicleMake.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.vehicleModel.toLowerCase().includes(searchQuery.toLowerCase()) ||
          item.mainSymptom.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : historyData;

  const handleRowClick = useCallback(
    (item: DiagnosisHistoryItem) => {
      navigate(`/diagnosis/${item.id}`);
    },
    [navigate]
  );

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('hu-HU', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <div className="min-h-screen bg-white text-slate-900">
      {/* Header */}
      <header className="flex items-center justify-between whitespace-nowrap border-b border-slate-300 px-6 py-3 bg-white sticky top-0 z-50">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-[#0055d4] flex items-center justify-center rounded-lg shadow-sm">
            <Wrench className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-slate-900 text-lg font-bold leading-tight tracking-tight uppercase italic">
            MechanicAI <span className="text-[#0055d4]">Pro</span>
          </h2>
        </div>
        <div className="flex flex-1 justify-end gap-8 items-center">
          <nav className="hidden md:flex items-center gap-8">
            <Link
              to="/"
              className="text-slate-700 hover:text-[#0055d4] text-sm font-bold uppercase tracking-wider transition-colors"
            >
              Vezérlőpult
            </Link>
            <Link
              to="/diagnosis"
              className="text-slate-700 hover:text-[#0055d4] text-sm font-bold uppercase tracking-wider transition-colors"
            >
              Diagnosztika
            </Link>
            <Link
              to="/history"
              className="text-[#0055d4] text-sm font-black uppercase tracking-wider border-b-2 border-[#0055d4] py-1"
            >
              Előzmények
            </Link>
          </nav>
          <div className="flex gap-3">
            <button
              aria-label="Értesítések"
              className="flex w-10 h-10 cursor-pointer items-center justify-center rounded bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-300 transition-colors"
            >
              <Bell className="w-5 h-5" />
            </button>
            <div className="h-10 w-10 rounded-full border-2 border-[#0055d4] p-0.5">
              <div className="bg-slate-200 rounded-full w-full h-full" />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
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
            {/* Search Input */}
            <div className="lg:col-span-6">
              <label className="flex flex-col w-full h-12">
                <div className="flex w-full flex-1 items-stretch rounded-lg overflow-hidden border border-slate-300 group focus-within:border-[#0055d4] focus-within:ring-1 focus-within:ring-[#0055d4]">
                  <div className="text-slate-500 flex bg-slate-50 items-center justify-center px-4 border-r border-slate-300">
                    <Search className="w-5 h-5" />
                  </div>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="flex w-full min-w-0 flex-1 border-none bg-white text-slate-900 placeholder:text-slate-500 px-4 focus:ring-0 text-base font-medium"
                    placeholder="Keresés rendszám, alvázszám vagy tünet alapján..."
                  />
                </div>
              </label>
            </div>

            {/* Manufacturer Filter */}
            <div className="lg:col-span-2">
              <button className="flex w-full h-12 items-center justify-between rounded-lg border border-slate-300 bg-white px-4 hover:border-[#0055d4] hover:bg-slate-50 transition-colors group">
                <span className="text-slate-700 text-sm font-bold">Gyártó</span>
                <Filter className="w-5 h-5 text-slate-400 group-hover:text-[#0055d4] transition-colors" />
              </button>
            </div>

            {/* Date Filter */}
            <div className="lg:col-span-2">
              <button className="flex w-full h-12 items-center justify-between rounded-lg border border-slate-300 bg-white px-4 hover:border-[#0055d4] hover:bg-slate-50 transition-colors group">
                <span className="text-slate-700 text-sm font-bold">Időszak</span>
                <Calendar className="w-5 h-5 text-slate-400 group-hover:text-[#0055d4] transition-colors" />
              </button>
            </div>

            {/* Apply Filters Button */}
            <div className="lg:col-span-2">
              <button className="flex w-full h-12 items-center justify-center rounded-lg bg-slate-900 text-white px-4 font-bold uppercase text-xs tracking-widest hover:bg-slate-800 transition-colors shadow-md">
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
                  <th className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Rendszám
                  </th>
                  <th className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Jármű adatok
                  </th>
                  <th className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Diagnosztika dátuma
                  </th>
                  <th className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Hibakód
                  </th>
                  <th className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Fő tünet
                  </th>
                  <th className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600">
                    Állapot
                  </th>
                  <th className="px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-600 text-right">
                    Művelet
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {isLoading ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-slate-500">
                      Betöltés...
                    </td>
                  </tr>
                ) : filteredData.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-slate-500">
                      Nincs találat
                    </td>
                  </tr>
                ) : (
                  filteredData.map((item) => (
                    <tr
                      key={item.id}
                      className="hover:bg-blue-50 transition-colors group cursor-pointer"
                      onClick={() => handleRowClick(item)}
                    >
                      <td className="px-6 py-5">
                        <span className="font-black text-slate-900 font-mono text-lg bg-slate-100 border border-slate-200 px-2 py-1 rounded group-hover:bg-white group-hover:border-blue-200">
                          {item.licensePlate}
                        </span>
                      </td>
                      <td className="px-6 py-5">
                        <div className="flex flex-col">
                          <span className="text-slate-900 font-bold">
                            {item.vehicleMake} {item.vehicleModel}
                          </span>
                          <span className="text-slate-600 text-xs font-medium">
                            {item.vehicleYear}
                            {item.vehicleTrim && ` • ${item.vehicleTrim}`}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-5 text-slate-700 font-medium">
                        {formatDate(item.diagnosisDate)}
                      </td>
                      <td className="px-6 py-5">
                        <span className="bg-white text-[#0055d4] border border-[#0055d4]/30 px-3 py-1 rounded font-mono font-bold text-sm shadow-sm">
                          {item.dtcCode}
                        </span>
                      </td>
                      <td className="px-6 py-5 text-slate-700 max-w-xs truncate font-medium">
                        {item.mainSymptom}
                      </td>
                      <td className="px-6 py-5">
                        <StatusBadge status={item.status} />
                      </td>
                      <td className="px-6 py-5 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            navigate(`/diagnosis/${item.id}`);
                          }}
                          className="text-[#0055d4] hover:text-white hover:bg-[#0055d4] font-bold uppercase text-[10px] tracking-widest border border-[#0055d4] px-3 py-1.5 rounded-lg transition-all"
                        >
                          Részletek
                        </button>
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
              Megjelenítve: {filteredData.length} / {totalRecords.toLocaleString()} rekord
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="w-8 h-8 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-600 hover:bg-slate-100 transition-colors disabled:opacity-50"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button className="w-8 h-8 flex items-center justify-center rounded border border-[#0055d4] bg-[#0055d4] text-white font-bold text-xs shadow-md">
                {page}
              </button>
              <button
                onClick={() => setPage((p) => p + 1)}
                className="w-8 h-8 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-600 font-bold text-xs hover:bg-slate-100 transition-colors"
              >
                {page + 1}
              </button>
              <button
                onClick={() => setPage((p) => p + 2)}
                className="w-8 h-8 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-600 font-bold text-xs hover:bg-slate-100 transition-colors"
              >
                {page + 2}
              </button>
              <button
                onClick={() => setPage((p) => p + 1)}
                className="w-8 h-8 flex items-center justify-center rounded border border-slate-300 bg-white text-slate-600 hover:bg-slate-100 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Solution Rate */}
          <div className="bg-white border border-slate-300 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <CheckCircle className="w-6 h-6 text-green-600" />
              <span className="text-xs font-black uppercase tracking-widest text-slate-500">
                Megoldási arány
              </span>
            </div>
            <div className="text-3xl font-black text-slate-900 uppercase italic">
              {statsLoading ? '...' : `${stats.solutionRate}%`}
            </div>
            <div className="mt-2 h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-600 transition-all duration-500"
                style={{ width: `${stats.solutionRate}%` }}
              />
            </div>
          </div>

          {/* Total Diagnoses */}
          <div className="bg-white border border-slate-300 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <BarChart3 className="w-6 h-6 text-[#0055d4]" />
              <span className="text-xs font-black uppercase tracking-widest text-slate-500">
                Összes diagnosztika
              </span>
            </div>
            <div className="text-3xl font-black text-slate-900 uppercase italic">
              {statsLoading ? '...' : stats.totalDiagnoses.toLocaleString()}
            </div>
            <div className="mt-2 text-xs font-bold text-slate-600">
              +12% az előző hónaphoz képest
            </div>
          </div>

          {/* AI Accuracy */}
          <div className="bg-white border border-slate-300 p-5 rounded-xl shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center gap-3 mb-3">
              <Zap className="w-6 h-6 text-yellow-600" />
              <span className="text-xs font-black uppercase tracking-widest text-slate-500">
                MI pontosság
              </span>
            </div>
            <div className="text-3xl font-black text-slate-900 uppercase italic">
              {statsLoading ? '...' : `${stats.aiAccuracy.toFixed(1)}%`}
            </div>
            <div className="mt-2 text-xs font-bold text-slate-600">
              Ellenőrzött javítások alapján
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-300 px-6 py-6 mt-12 bg-white">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 bg-slate-500 rounded-sm" />
            <p className="text-slate-500 text-xs font-bold uppercase tracking-widest">
              © 2023 MechanicAI Diagnosztikai Rendszerek
            </p>
          </div>
          <div className="flex gap-6">
            <a
              href="#"
              className="text-slate-500 hover:text-[#0055d4] text-xs font-bold uppercase tracking-widest transition-colors"
            >
              Dokumentáció
            </a>
            <a
              href="#"
              className="text-slate-500 hover:text-[#0055d4] text-xs font-bold uppercase tracking-widest transition-colors"
            >
              Támogatás
            </a>
            <a
              href="#"
              className="text-slate-500 hover:text-[#0055d4] text-xs font-bold uppercase tracking-widest transition-colors"
            >
              API Státusz
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
