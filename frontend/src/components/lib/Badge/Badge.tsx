import { type HTMLAttributes, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export type BadgeVariant =
  | 'success'
  | 'warning'
  | 'pending'
  | 'error'
  | 'info'
  | 'default';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Badge color variant */
  variant?: BadgeVariant;
  /** Badge size */
  size?: 'sm' | 'md';
  /** Badge content */
  children: ReactNode;
}

const variantStyles: Record<BadgeVariant, string> = {
  success: 'bg-status-success-light text-status-success-dark',
  warning: 'bg-status-warning-light text-status-warning-dark',
  pending: 'bg-status-pending-light text-status-pending-dark',
  error: 'bg-status-error-light text-status-error-dark',
  info: 'bg-status-info-light text-status-info-dark',
  default: 'bg-muted text-muted-foreground',
};

const sizeStyles = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-sm',
};

/**
 * Badge component for status indicators.
 *
 * @example
 * ```tsx
 * <Badge variant="success">JAVÍTVA</Badge>
 * <Badge variant="warning">FOLYAMATBAN</Badge>
 * <Badge variant="pending">FÜGGŐBEN</Badge>
 * <Badge variant="error">HIBA</Badge>
 * ```
 */
export function Badge({
  variant = 'default',
  size = 'md',
  className,
  children,
  ...props
}: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center font-medium rounded-full',
        'whitespace-nowrap',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

export default Badge;
