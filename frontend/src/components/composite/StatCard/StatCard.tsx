import { type ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface StatCardProps {
  /** Stat label */
  label: string;
  /** Stat value */
  value: string | number;
  /** Optional icon */
  icon?: ReactNode;
  /** Optional trend information */
  trend?: {
    direction: 'up' | 'down' | 'neutral';
    value: string;
  };
  /** Card color theme */
  color?: 'primary' | 'success' | 'warning' | 'error' | 'default';
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Additional className */
  className?: string;
}

const colorStyles = {
  primary: 'text-primary-600',
  success: 'text-status-success',
  warning: 'text-status-warning',
  error: 'text-status-error',
  default: 'text-foreground',
};

const trendColors = {
  up: 'text-status-success',
  down: 'text-status-error',
  neutral: 'text-muted-foreground',
};

const TrendIcons = {
  up: TrendingUp,
  down: TrendingDown,
  neutral: Minus,
};

/**
 * Statistic card component for displaying metrics.
 *
 * @example
 * ```tsx
 * <StatCard
 *   label="Megoldási arány"
 *   value="94.2%"
 *   color="success"
 *   trend={{ direction: 'up', value: '+2.1%' }}
 * />
 *
 * <StatCard
 *   label="Összes diagnosztika"
 *   value={1248}
 *   color="primary"
 * />
 * ```
 */
export function StatCard({
  label,
  value,
  icon,
  trend,
  color = 'default',
  size = 'md',
  className,
}: StatCardProps) {
  const TrendIcon = trend ? TrendIcons[trend.direction] : null;

  const sizeStyles = {
    sm: {
      container: 'p-3',
      label: 'text-xs',
      value: 'text-xl',
      icon: 'h-8 w-8',
    },
    md: {
      container: 'p-4',
      label: 'text-sm',
      value: 'text-2xl',
      icon: 'h-10 w-10',
    },
    lg: {
      container: 'p-6',
      label: 'text-base',
      value: 'text-3xl',
      icon: 'h-12 w-12',
    },
  };

  const styles = sizeStyles[size];

  return (
    <div
      className={cn(
        'bg-card rounded-lg border border-border',
        styles.container,
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          {/* Label */}
          <p className={cn('text-muted-foreground font-medium', styles.label)}>
            {label}
          </p>

          {/* Value */}
          <p
            className={cn(
              'font-bold tracking-tight mt-1',
              styles.value,
              colorStyles[color]
            )}
          >
            {typeof value === 'number' ? value.toLocaleString('hu-HU') : value}
          </p>

          {/* Trend */}
          {trend && TrendIcon && (
            <div
              className={cn(
                'flex items-center gap-1 mt-2',
                trendColors[trend.direction]
              )}
            >
              <TrendIcon className="h-4 w-4" />
              <span className="text-sm font-medium">{trend.value}</span>
            </div>
          )}
        </div>

        {/* Icon */}
        {icon && (
          <div
            className={cn(
              'flex-shrink-0 p-2 rounded-lg bg-muted',
              colorStyles[color],
              styles.icon
            )}
          >
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}

export default StatCard;
