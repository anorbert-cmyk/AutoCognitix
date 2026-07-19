/**
 * EmptyState — borderless "nothing here yet" placeholder.
 *
 * Intentionally chromeless so it drops cleanly into a table cell (<td>), a
 * `.card`, or a DashboardCard without doubling borders/padding. An empty result
 * is not an error, so the container carries no role="alert" / role="status".
 *
 * The optional action renders as a router <Link> when given a `to`, or as a
 * <button> when given an `onClick`.
 */
import { Link } from 'react-router-dom'

type Action = { label: string; to: string } | { label: string; onClick: () => void }

interface EmptyStateProps {
  icon?: React.ReactNode
  title: string
  description?: string
  action?: Action
  className?: string
}

export function EmptyState({ icon, title, description, action, className = '' }: EmptyStateProps) {
  const cls =
    'inline-flex items-center gap-1.5 rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white ' +
    'transition-colors hover:bg-primary-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'

  return (
    <div className={`flex flex-col items-center gap-3 py-10 text-center ${className}`}>
      {icon && (
        <span
          className="flex h-12 w-12 items-center justify-center rounded-full bg-muted"
          aria-hidden="true"
        >
          {icon}
        </span>
      )}
      <p className="font-medium text-foreground">{title}</p>
      {description && <p className="max-w-xs text-sm text-muted-foreground">{description}</p>}
      {action &&
        ('to' in action ? (
          <Link to={action.to} className={cls}>
            {action.label}
          </Link>
        ) : (
          <button type="button" onClick={action.onClick} className={cls}>
            {action.label}
          </button>
        ))}
    </div>
  )
}
