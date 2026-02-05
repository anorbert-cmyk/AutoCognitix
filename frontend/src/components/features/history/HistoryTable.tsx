import { Eye, MoreHorizontal } from 'lucide-react';
import { Table, type TableColumn } from '@/components/composite';
import { Badge, type BadgeVariant, Button } from '@/components/lib';

export type DiagnosisStatus = 'completed' | 'in_progress' | 'pending';

export interface DiagnosisHistoryItem {
  id: string;
  licensePlate: string;
  vehicleMake: string;
  vehicleModel: string;
  vehicleYear: number;
  diagnosisDate: string;
  dtcCodes: string[];
  mainSymptom: string;
  status: DiagnosisStatus;
}

export interface HistoryTableProps {
  /** Array of diagnosis history items */
  data: DiagnosisHistoryItem[];
  /** Loading state */
  loading?: boolean;
  /** Callback when row is clicked */
  onRowClick?: (item: DiagnosisHistoryItem) => void;
  /** Callback when view button is clicked */
  onView?: (item: DiagnosisHistoryItem) => void;
  /** Additional className */
  className?: string;
}

const statusConfig: Record<
  DiagnosisStatus,
  { label: string; variant: BadgeVariant }
> = {
  completed: { label: 'JAVÍTVA', variant: 'success' },
  in_progress: { label: 'FOLYAMATBAN', variant: 'warning' },
  pending: { label: 'FÜGGŐBEN', variant: 'pending' },
};

/**
 * History table component for displaying diagnosis history.
 *
 * @example
 * ```tsx
 * <HistoryTable
 *   data={diagnosisHistory}
 *   onRowClick={(item) => navigate(`/diagnosis/${item.id}`)}
 *   onView={(item) => openDetailModal(item)}
 * />
 * ```
 */
export function HistoryTable({
  data,
  loading = false,
  onRowClick,
  onView,
  className,
}: HistoryTableProps) {
  const columns: TableColumn<DiagnosisHistoryItem>[] = [
    {
      key: 'licensePlate',
      header: 'Rendszám',
      render: (value) => (
        <span className="font-mono font-medium text-foreground">
          {String(value)}
        </span>
      ),
    },
    {
      key: 'vehicle',
      header: 'Jármű adatok',
      render: (_, row) => (
        <div className="flex flex-col">
          <span className="font-medium text-foreground">
            {row.vehicleMake} {row.vehicleModel}
          </span>
          <span className="text-xs text-muted-foreground">{row.vehicleYear}</span>
        </div>
      ),
      hideOnMobile: true,
    },
    {
      key: 'diagnosisDate',
      header: 'Diagnosztika dátuma',
      render: (value) => {
        const date = new Date(String(value));
        return (
          <span className="text-muted-foreground">
            {date.toLocaleDateString('hu-HU', {
              year: 'numeric',
              month: 'short',
              day: 'numeric',
            })}
          </span>
        );
      },
      hideOnMobile: true,
    },
    {
      key: 'dtcCodes',
      header: 'Hibakód',
      render: (value) => {
        const codes = value as string[];
        return (
          <div className="flex flex-wrap gap-1">
            {codes.slice(0, 2).map((code) => (
              <span
                key={code}
                className="px-2 py-0.5 text-xs font-mono bg-muted rounded"
              >
                {code}
              </span>
            ))}
            {codes.length > 2 && (
              <span className="text-xs text-muted-foreground">
                +{codes.length - 2}
              </span>
            )}
          </div>
        );
      },
    },
    {
      key: 'mainSymptom',
      header: 'Fő tünet',
      render: (value) => (
        <span className="text-sm text-muted-foreground line-clamp-1">
          {String(value)}
        </span>
      ),
      hideOnMobile: true,
    },
    {
      key: 'status',
      header: 'Állapot',
      render: (value) => {
        const config = statusConfig[value as DiagnosisStatus];
        return <Badge variant={config.variant}>{config.label}</Badge>;
      },
    },
    {
      key: 'actions',
      header: 'Művelet',
      align: 'right',
      render: (_, row) => (
        <div className="flex items-center justify-end gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              onView?.(row);
            }}
            aria-label="Megtekintés"
          >
            <Eye className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => e.stopPropagation()}
            aria-label="További műveletek"
          >
            <MoreHorizontal className="h-4 w-4" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <Table
      data={data}
      columns={columns}
      loading={loading}
      onRowClick={onRowClick}
      emptyMessage="Nincs megjeleníthető diagnosztika"
      rowKey={(row) => row.id}
      className={className}
    />
  );
}

export default HistoryTable;
