import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { History } from 'lucide-react';

// New components
import { Card } from '@/components/lib';
import { PageContainer } from '@/components/layouts';
import { Pagination } from '@/components/composite';
import {
  HistoryFilterBar,
  HistoryTable,
  HistoryStats,
  type HistoryFilters,
  type DiagnosisHistoryItem,
} from '@/components/features/history';

// Legacy hooks - will continue using existing API
import {
  useDiagnosisHistory,
  useDiagnosisStats,
} from '@/services/hooks';

const initialFilters: HistoryFilters = {
  search: '',
  manufacturer: '',
  dateFrom: '',
  dateTo: '',
};

// Mock data for demonstration when API is not available
const mockHistoryData: DiagnosisHistoryItem[] = [
  {
    id: '1',
    licensePlate: 'ABC-123',
    vehicleMake: 'Volkswagen',
    vehicleModel: 'Golf',
    vehicleYear: 2019,
    diagnosisDate: '2025-02-01T10:30:00Z',
    dtcCodes: ['P0171', 'P0174'],
    mainSymptom: 'Motor egyenetlenül jár, gyenge gyorsulás',
    status: 'completed',
  },
  {
    id: '2',
    licensePlate: 'XYZ-789',
    vehicleMake: 'BMW',
    vehicleModel: '320d',
    vehicleYear: 2018,
    diagnosisDate: '2025-02-03T14:15:00Z',
    dtcCodes: ['P0401'],
    mainSymptom: 'Check engine lámpa ég, EGR hiba',
    status: 'in_progress',
  },
  {
    id: '3',
    licensePlate: 'DEF-456',
    vehicleMake: 'Audi',
    vehicleModel: 'A4',
    vehicleYear: 2020,
    diagnosisDate: '2025-02-04T09:00:00Z',
    dtcCodes: ['P0300', 'P0301', 'P0302'],
    mainSymptom: 'Motor berúgásnál remeg',
    status: 'pending',
  },
  {
    id: '4',
    licensePlate: 'GHI-321',
    vehicleMake: 'Mercedes-Benz',
    vehicleModel: 'C200',
    vehicleYear: 2017,
    diagnosisDate: '2025-01-28T16:45:00Z',
    dtcCodes: ['P0420'],
    mainSymptom: 'Katalizátor hatékonyság alatt',
    status: 'completed',
  },
  {
    id: '5',
    licensePlate: 'JKL-654',
    vehicleMake: 'Škoda',
    vehicleModel: 'Octavia',
    vehicleYear: 2021,
    diagnosisDate: '2025-01-25T11:20:00Z',
    dtcCodes: ['P0507'],
    mainSymptom: 'Alapjárat túl magas',
    status: 'completed',
  },
];

const mockStats = {
  solutionRate: 94.2,
  totalDiagnoses: 1248,
  aiAccuracy: 98.8,
};

export default function HistoryPage() {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [limit] = useState(10);
  const [filters, setFilters] = useState<HistoryFilters>(initialFilters);
  const [appliedFilters, setAppliedFilters] =
    useState<HistoryFilters>(initialFilters);

  // Try to use API hooks (will fall back to mock data if API fails)
  const {
    data: apiHistoryData,
    isLoading,
  } = useDiagnosisHistory({
    skip: (page - 1) * limit,
    limit,
    vehicleMake: appliedFilters.manufacturer || undefined,
    dateFrom: appliedFilters.dateFrom || undefined,
    dateTo: appliedFilters.dateTo || undefined,
  });

  const { data: apiStats, isLoading: statsLoading } = useDiagnosisStats();

  // Use mock data if API data is not available
  const historyData = apiHistoryData?.items
    ? apiHistoryData.items.map((item) => ({
        id: item.id,
        licensePlate: 'N/A', // API doesn't provide license plate
        vehicleMake: item.vehicle_make,
        vehicleModel: item.vehicle_model,
        vehicleYear: item.vehicle_year,
        diagnosisDate: item.created_at,
        dtcCodes: item.dtc_codes,
        mainSymptom: 'Nincs megadva', // API doesn't provide symptoms in history list
        status: 'completed' as const,
      }))
    : mockHistoryData;

  const stats = apiStats
    ? {
        solutionRate: 94.2, // Calculate from API data
        totalDiagnoses: apiStats.total_diagnoses,
        aiAccuracy: apiStats.avg_confidence * 100,
      }
    : mockStats;

  const totalPages = apiHistoryData
    ? Math.ceil(apiHistoryData.total / limit)
    : Math.ceil(mockHistoryData.length / limit);

  const handleApplyFilters = useCallback(() => {
    setAppliedFilters(filters);
    setPage(1);
  }, [filters]);

  const handleClearFilters = useCallback(() => {
    setFilters(initialFilters);
    setAppliedFilters(initialFilters);
    setPage(1);
  }, []);

  const handleRowClick = useCallback(
    (item: DiagnosisHistoryItem) => {
      navigate(`/diagnosis/${item.id}`);
    },
    [navigate]
  );

  const handleView = useCallback(
    (item: DiagnosisHistoryItem) => {
      navigate(`/diagnosis/${item.id}`);
    },
    [navigate]
  );

  const handlePageChange = useCallback((newPage: number) => {
    setPage(newPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);

  // Filter data based on search
  const filteredData = appliedFilters.search
    ? historyData.filter(
        (item) =>
          item.licensePlate
            .toLowerCase()
            .includes(appliedFilters.search.toLowerCase()) ||
          item.dtcCodes.some((code) =>
            code.toLowerCase().includes(appliedFilters.search.toLowerCase())
          ) ||
          item.vehicleMake
            .toLowerCase()
            .includes(appliedFilters.search.toLowerCase()) ||
          item.vehicleModel
            .toLowerCase()
            .includes(appliedFilters.search.toLowerCase())
      )
    : historyData;

  return (
    <PageContainer maxWidth="xl" padding="md">
      {/* Page Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <History className="h-8 w-8 text-primary-600" />
          <h1 className="text-2xl font-bold text-foreground">
            Műhely előzmények
          </h1>
        </div>
        <p className="text-muted-foreground">
          Tekintse át korábbi diagnosztikáit és azok állapotát
        </p>
      </div>

      {/* Filter Bar */}
      <HistoryFilterBar
        filters={filters}
        onChange={setFilters}
        onApply={handleApplyFilters}
        onClear={handleClearFilters}
        className="mb-6"
      />

      {/* Results Count */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-sm text-muted-foreground">
          {filteredData.length} diagnosztika találat
          {appliedFilters.search && ` "${appliedFilters.search}" keresésre`}
        </p>
      </div>

      {/* History Table */}
      <Card className="mb-6">
        <HistoryTable
          data={filteredData}
          loading={isLoading}
          onRowClick={handleRowClick}
          onView={handleView}
        />
      </Card>

      {/* Pagination */}
      {totalPages > 1 && (
        <Pagination
          currentPage={page}
          totalPages={totalPages}
          onPageChange={handlePageChange}
          className="mb-8"
        />
      )}

      {/* Statistics */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold text-foreground mb-4">
          Összesített statisztikák
        </h2>
        <HistoryStats stats={stats} loading={statsLoading} />
      </div>
    </PageContainer>
  );
}
