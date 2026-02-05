import { Clock, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface RecentAnalysisItem {
  id: string;
  vehicleInfo: string;
  dtcCode: string;
  timestamp: string;
}

export interface RecentAnalysisListProps {
  /** Array of recent analysis items */
  items: RecentAnalysisItem[];
  /** Maximum number of items to show */
  maxItems?: number;
  /** Callback when an item is clicked */
  onItemClick?: (item: RecentAnalysisItem) => void;
  /** Additional className */
  className?: string;
}

/**
 * Horizontal scrollable list of recent analyses for the floating bottom bar.
 *
 * @example
 * ```tsx
 * <RecentAnalysisList
 *   items={recentDiagnoses}
 *   onItemClick={(item) => navigate(`/diagnosis/${item.id}`)}
 * />
 * ```
 */
export function RecentAnalysisList({
  items,
  maxItems = 5,
  onItemClick,
  className,
}: RecentAnalysisListProps) {
  const displayItems = items.slice(0, maxItems);

  if (displayItems.length === 0) {
    return (
      <div className={cn('flex items-center gap-2 text-muted-foreground', className)}>
        <Clock className="h-4 w-4" />
        <span className="text-sm">Nincs korábbi elemzés</span>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col gap-2', className)}>
      <div className="flex items-center gap-2 text-muted-foreground">
        <Clock className="h-4 w-4" />
        <span className="text-sm font-medium">Utolsó elemzések</span>
      </div>

      <div className="flex gap-3 overflow-x-auto pb-1 -mx-1 px-1">
        {displayItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onItemClick?.(item)}
            className={cn(
              'flex-shrink-0 flex items-center gap-2 px-3 py-2',
              'bg-muted/50 hover:bg-muted rounded-lg',
              'transition-colors duration-fast',
              'text-sm text-left',
              'max-w-[180px]'
            )}
          >
            <div className="flex-1 min-w-0">
              <p className="font-medium text-foreground truncate">
                {item.vehicleInfo}
              </p>
              <p className="text-xs text-muted-foreground">
                <span className="font-mono">{item.dtcCode}</span>
                {' · '}
                {formatTimestamp(item.timestamp)}
              </p>
            </div>
            <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          </button>
        ))}
      </div>
    </div>
  );
}

/**
 * Format timestamp to relative time (e.g., "5 perc")
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) return 'most';
  if (diffMins < 60) return `${diffMins} perc`;
  if (diffHours < 24) return `${diffHours} óra`;
  if (diffDays < 7) return `${diffDays} nap`;

  return date.toLocaleDateString('hu-HU', {
    month: 'short',
    day: 'numeric',
  });
}

export default RecentAnalysisList;
