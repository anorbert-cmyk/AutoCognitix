/**
 * Skeleton — decorative loading placeholder primitive.
 *
 * Purely visual: it is marked aria-hidden so screen readers ignore it. The
 * caller owns the accessible loading announcement (e.g. a role="status" region
 * with an sr-only label wrapping one or more Skeletons).
 */
export function Skeleton({ className = '' }: { className?: string }) {
  return <div aria-hidden="true" className={`animate-pulse rounded-md bg-muted ${className}`} />
}
