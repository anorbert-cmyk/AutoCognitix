/**
 * InspectionCategoryCard - Category card for each inspection failing/warning item
 *
 * Displays: category name (Hungarian), issue, related DTC badge,
 * severity badge, fix recommendation, and cost range.
 */

import { AlertTriangle, XCircle } from 'lucide-react'
import type { FailingItem } from '../../../services/inspectionService'

interface InspectionCategoryCardProps {
  item: FailingItem
}

const severityConfig = {
  fail: {
    border: 'border-l-red-500',
    badge: 'bg-red-100 text-red-700',
    badgeLabel: 'NEM FELEL MEG',
    Icon: XCircle,
    iconColor: 'text-red-500',
  },
  warning: {
    border: 'border-l-yellow-500',
    badge: 'bg-yellow-100 text-yellow-700',
    badgeLabel: 'FIGYELMEZTETÉS',
    Icon: AlertTriangle,
    iconColor: 'text-yellow-500',
  },
  pass: {
    border: 'border-l-green-500',
    badge: 'bg-green-100 text-green-700',
    badgeLabel: 'MEGFELELT',
    Icon: AlertTriangle,
    iconColor: 'text-green-500',
  },
} as const

export default function InspectionCategoryCard({ item }: InspectionCategoryCardProps) {
  const config = severityConfig[item.severity]
  const Icon = config.Icon

  const costFormatter = new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency: 'HUF',
    maximumFractionDigits: 0,
  })

  return (
    <div
      className={`bg-white rounded-xl border border-slate-200 border-l-4 ${config.border} shadow-sm overflow-hidden`}
    >
      <div className="p-5">
        {/* Header: Category + Severity badge */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <Icon className={`h-5 w-5 flex-shrink-0 ${config.iconColor}`} aria-hidden="true" />
            <h3 className="font-bold text-slate-900 text-base">
              {item.category_hu}
            </h3>
          </div>
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider whitespace-nowrap ${config.badge}`}
          >
            {config.badgeLabel}
          </span>
        </div>

        {/* Issue description */}
        <p className="text-sm text-slate-700 mb-3 leading-relaxed">
          {item.issue}
        </p>

        {/* Related DTC badge */}
        {item.related_dtc && (
          <div className="mb-3">
            <span className="inline-flex items-center px-2.5 py-1 rounded-lg bg-slate-100 text-slate-700 text-xs font-mono font-bold">
              {item.related_dtc}
            </span>
          </div>
        )}

        {/* Fix recommendation */}
        <div className="bg-slate-50 rounded-lg p-3 mb-3">
          <p className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">
            Javitasi javaslat
          </p>
          <p className="text-sm text-slate-700">
            {item.fix_recommendation}
          </p>
        </div>

        {/* Cost range */}
        <div className="flex items-center justify-between pt-2 border-t border-slate-100">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Becsult koltseg
          </span>
          <span className="text-sm font-bold text-slate-900">
            {item.estimated_cost_min === 0 && item.estimated_cost_max === 0
              ? 'Nincs becsles'
              : `${costFormatter.format(item.estimated_cost_min)} - ${costFormatter.format(item.estimated_cost_max)}`}
          </span>
        </div>
      </div>
    </div>
  )
}
