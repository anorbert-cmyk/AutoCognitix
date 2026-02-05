import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface TableColumn<T> {
  /** Unique key for the column */
  key: keyof T | string;
  /** Column header text */
  header: string;
  /** Custom render function for cell content */
  render?: (value: unknown, row: T, index: number) => ReactNode;
  /** Enable sorting for this column */
  sortable?: boolean;
  /** Column width (CSS value) */
  width?: string;
  /** Align cell content */
  align?: 'left' | 'center' | 'right';
  /** Hide on mobile */
  hideOnMobile?: boolean;
}

export interface TableProps<T> {
  /** Array of data to display */
  data: T[];
  /** Column definitions */
  columns: TableColumn<T>[];
  /** Show loading state */
  loading?: boolean;
  /** Message to show when data is empty */
  emptyMessage?: string;
  /** Callback when row is clicked */
  onRowClick?: (row: T, index: number) => void;
  /** Get unique key for each row */
  rowKey?: (row: T, index: number) => string | number;
  /** Custom row className */
  rowClassName?: (row: T, index: number) => string;
  /** Additional className for the table */
  className?: string;
}

/**
 * Generic table component with sorting and custom rendering.
 *
 * @example
 * ```tsx
 * <Table
 *   data={diagnosisHistory}
 *   columns={[
 *     { key: 'licensePlate', header: 'Rendszám' },
 *     { key: 'vehicle', header: 'Jármű adatok' },
 *     {
 *       key: 'status',
 *       header: 'Állapot',
 *       render: (value) => <Badge variant={getStatusVariant(value)}>{value}</Badge>
 *     },
 *   ]}
 *   onRowClick={(row) => navigate(`/diagnosis/${row.id}`)}
 * />
 * ```
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function Table<T extends Record<string, any>>({
  data,
  columns,
  loading = false,
  emptyMessage = 'Nincs megjeleníthető adat',
  onRowClick,
  rowKey,
  rowClassName,
  className,
}: TableProps<T>) {
  const getRowKey = (row: T, index: number): string | number => {
    if (rowKey) return rowKey(row, index);
    if ('id' in row) return row.id as string | number;
    return index;
  };

  const getCellValue = (row: T, column: TableColumn<T>): unknown => {
    const key = column.key as keyof T;
    return row[key];
  };

  const alignmentClasses = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right',
  };

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="w-full border-collapse">
        {/* Table Header */}
        <thead>
          <tr className="border-b border-border bg-muted/50">
            {columns.map((column) => (
              <th
                key={String(column.key)}
                className={cn(
                  'px-4 py-3 text-sm font-medium text-muted-foreground',
                  alignmentClasses[column.align || 'left'],
                  column.hideOnMobile && 'hidden md:table-cell'
                )}
                style={{ width: column.width }}
              >
                {column.header}
              </th>
            ))}
          </tr>
        </thead>

        {/* Table Body */}
        <tbody>
          {loading ? (
            // Loading state
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-12 text-center text-muted-foreground"
              >
                <div className="flex items-center justify-center gap-2">
                  <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary-600 border-t-transparent" />
                  <span>Betöltés...</span>
                </div>
              </td>
            </tr>
          ) : data.length === 0 ? (
            // Empty state
            <tr>
              <td
                colSpan={columns.length}
                className="px-4 py-12 text-center text-muted-foreground"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            // Data rows
            data.map((row, rowIndex) => (
              <tr
                key={getRowKey(row, rowIndex)}
                onClick={() => onRowClick?.(row, rowIndex)}
                className={cn(
                  'border-b border-border transition-colors',
                  onRowClick && 'cursor-pointer hover:bg-muted/50',
                  rowClassName?.(row, rowIndex)
                )}
              >
                {columns.map((column) => {
                  const value = getCellValue(row, column);
                  return (
                    <td
                      key={String(column.key)}
                      className={cn(
                        'px-4 py-3 text-sm',
                        alignmentClasses[column.align || 'left'],
                        column.hideOnMobile && 'hidden md:table-cell'
                      )}
                    >
                      {column.render
                        ? column.render(value, row, rowIndex)
                        : String(value ?? '')}
                    </td>
                  );
                })}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

export default Table;
