import { useState } from 'react';
import { Filter, Calendar } from 'lucide-react';
import { Button } from '@/components/lib';
import { SearchInput } from '@/components/composite';
import { Select, type SelectOption } from '@/components/lib/Select';
import { cn } from '@/lib/utils';

export interface HistoryFilters {
  search: string;
  manufacturer: string;
  dateFrom: string;
  dateTo: string;
}

export interface HistoryFilterBarProps {
  /** Current filter values */
  filters: HistoryFilters;
  /** Callback when filters change */
  onChange: (filters: HistoryFilters) => void;
  /** Callback when apply button is clicked */
  onApply: () => void;
  /** Callback to clear all filters */
  onClear?: () => void;
  /** List of manufacturer options */
  manufacturers?: SelectOption[];
  /** Additional className */
  className?: string;
}

const defaultManufacturers: SelectOption[] = [
  { value: '', label: 'Összes gyártó' },
  { value: 'audi', label: 'Audi' },
  { value: 'bmw', label: 'BMW' },
  { value: 'ford', label: 'Ford' },
  { value: 'mercedes', label: 'Mercedes-Benz' },
  { value: 'opel', label: 'Opel' },
  { value: 'skoda', label: 'Škoda' },
  { value: 'toyota', label: 'Toyota' },
  { value: 'volkswagen', label: 'Volkswagen' },
];

/**
 * Filter bar component for the history page.
 *
 * @example
 * ```tsx
 * <HistoryFilterBar
 *   filters={filters}
 *   onChange={setFilters}
 *   onApply={handleApplyFilters}
 * />
 * ```
 */
export function HistoryFilterBar({
  filters,
  onChange,
  onApply,
  onClear,
  manufacturers = defaultManufacturers,
  className,
}: HistoryFilterBarProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const updateFilter = <K extends keyof HistoryFilters>(
    key: K,
    value: HistoryFilters[K]
  ) => {
    onChange({ ...filters, [key]: value });
  };

  const hasActiveFilters =
    filters.search ||
    filters.manufacturer ||
    filters.dateFrom ||
    filters.dateTo;

  return (
    <div
      className={cn(
        'bg-card border border-border rounded-lg p-4',
        className
      )}
    >
      {/* Main Filter Row */}
      <div className="flex flex-col md:flex-row gap-3">
        {/* Search Input */}
        <div className="flex-1 min-w-0">
          <SearchInput
            value={filters.search}
            onChange={(value) => updateFilter('search', value)}
            placeholder="Keresés rendszám, hibakód..."
            onSearch={onApply}
          />
        </div>

        {/* Manufacturer Select - Always visible on md+ */}
        <div className="w-full md:w-48">
          <Select
            options={manufacturers}
            value={filters.manufacturer}
            onChange={(value) => updateFilter('manufacturer', value)}
            placeholder="Összes gyártó"
          />
        </div>

        {/* Date Range - Toggleable on mobile */}
        <div
          className={cn(
            'flex gap-2',
            !isExpanded && 'hidden md:flex'
          )}
        >
          <div className="relative flex-1 md:w-36">
            <input
              type="date"
              value={filters.dateFrom}
              onChange={(e) => updateFilter('dateFrom', e.target.value)}
              className={cn(
                'w-full h-10 px-3 text-sm',
                'bg-background border border-input rounded-lg',
                'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2'
              )}
              aria-label="Kezdő dátum"
            />
          </div>
          <span className="flex items-center text-muted-foreground">-</span>
          <div className="relative flex-1 md:w-36">
            <input
              type="date"
              value={filters.dateTo}
              onChange={(e) => updateFilter('dateTo', e.target.value)}
              className={cn(
                'w-full h-10 px-3 text-sm',
                'bg-background border border-input rounded-lg',
                'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2'
              )}
              aria-label="Záró dátum"
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          {/* Mobile: Expand Filters Button */}
          <Button
            variant="ghost"
            size="md"
            onClick={() => setIsExpanded(!isExpanded)}
            className="md:hidden"
            aria-expanded={isExpanded}
            aria-label={isExpanded ? 'Szűrők elrejtése' : 'Több szűrő'}
          >
            <Calendar className="h-4 w-4" />
          </Button>

          {/* Apply Filter Button */}
          <Button
            variant="outline"
            size="md"
            onClick={onApply}
            leftIcon={<Filter className="h-4 w-4" />}
          >
            Szűrés
          </Button>

          {/* Clear Filters (optional) */}
          {onClear && hasActiveFilters && (
            <Button
              variant="ghost"
              size="md"
              onClick={onClear}
              className="text-muted-foreground"
            >
              Törlés
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}

export default HistoryFilterBar;
