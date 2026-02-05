import { CheckCircle, BarChart3, Cpu } from 'lucide-react';
import { StatCard } from '@/components/composite';
import { cn } from '@/lib/utils';

export interface HistoryStatsData {
  /** Solution rate percentage */
  solutionRate: number;
  /** Total number of diagnoses */
  totalDiagnoses: number;
  /** AI accuracy percentage */
  aiAccuracy: number;
}

export interface HistoryStatsProps {
  /** Statistics data */
  stats: HistoryStatsData;
  /** Loading state */
  loading?: boolean;
  /** Additional className */
  className?: string;
}

/**
 * Statistics bar for the history page.
 *
 * @example
 * ```tsx
 * <HistoryStats
 *   stats={{
 *     solutionRate: 94.2,
 *     totalDiagnoses: 1248,
 *     aiAccuracy: 98.8,
 *   }}
 * />
 * ```
 */
export function HistoryStats({
  stats,
  loading = false,
  className,
}: HistoryStatsProps) {
  if (loading) {
    return (
      <div
        className={cn(
          'grid grid-cols-1 md:grid-cols-3 gap-4',
          className
        )}
      >
        {[1, 2, 3].map((i) => (
          <div
            key={i}
            className="bg-card border border-border rounded-lg p-4 animate-pulse"
          >
            <div className="h-4 w-24 bg-muted rounded mb-2" />
            <div className="h-8 w-16 bg-muted rounded" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div
      className={cn(
        'grid grid-cols-1 md:grid-cols-3 gap-4',
        className
      )}
    >
      <StatCard
        label="Megoldási arány"
        value={`${stats.solutionRate.toFixed(1)}%`}
        color="success"
        icon={<CheckCircle className="h-5 w-5" />}
      />

      <StatCard
        label="Összes diagnosztika"
        value={stats.totalDiagnoses}
        color="primary"
        icon={<BarChart3 className="h-5 w-5" />}
      />

      <StatCard
        label="AI pontosság"
        value={`${stats.aiAccuracy.toFixed(1)}%`}
        color="primary"
        icon={<Cpu className="h-5 w-5" />}
      />
    </div>
  );
}

export default HistoryStats;
