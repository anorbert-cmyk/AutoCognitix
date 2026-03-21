/**
 * RecommendationCard Component
 *
 * Large recommendation display with icon, color-coded background,
 * text explanation, and ratio badge.
 */

import { Wrench, TrendingDown, Trash2 } from 'lucide-react';

interface RecommendationCardProps {
  recommendation: 'repair' | 'sell' | 'scrap';
  text: string;
  ratio: number;
}

const recommendationConfig = {
  repair: {
    label: 'Erdemes megjavitani',
    icon: Wrench,
    bgClass: 'bg-gradient-to-br from-green-50 to-emerald-50',
    borderClass: 'border-green-200',
    iconBgClass: 'bg-green-100',
    iconColorClass: 'text-green-600',
    titleColorClass: 'text-green-800',
    textColorClass: 'text-green-700',
    badgeBgClass: 'bg-green-600',
  },
  sell: {
    label: 'Erdemes eladni',
    icon: TrendingDown,
    bgClass: 'bg-gradient-to-br from-yellow-50 to-amber-50',
    borderClass: 'border-yellow-200',
    iconBgClass: 'bg-yellow-100',
    iconColorClass: 'text-yellow-600',
    titleColorClass: 'text-yellow-800',
    textColorClass: 'text-yellow-700',
    badgeBgClass: 'bg-yellow-600',
  },
  scrap: {
    label: 'Bontasra javasolt',
    icon: Trash2,
    bgClass: 'bg-gradient-to-br from-red-50 to-rose-50',
    borderClass: 'border-red-200',
    iconBgClass: 'bg-red-100',
    iconColorClass: 'text-red-600',
    titleColorClass: 'text-red-800',
    textColorClass: 'text-red-700',
    badgeBgClass: 'bg-red-600',
  },
} as const;

export function RecommendationCard({ recommendation, text, ratio }: RecommendationCardProps) {
  const config = recommendationConfig[recommendation];
  const IconComponent = config.icon;

  return (
    <div
      className={`rounded-2xl border ${config.borderClass} ${config.bgClass} p-6 sm:p-8 shadow-sm`}
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
        {/* Icon */}
        <div
          className={`flex items-center justify-center w-16 h-16 rounded-2xl ${config.iconBgClass} flex-shrink-0`}
        >
          <IconComponent className={`h-8 w-8 ${config.iconColorClass}`} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 mb-2 flex-wrap">
            <h2 className={`text-xl sm:text-2xl font-bold ${config.titleColorClass}`}>
              {config.label}
            </h2>
            <span
              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold text-white ${config.badgeBgClass}`}
            >
              {ratio}%
            </span>
          </div>
          <p className={`text-sm sm:text-base ${config.textColorClass} leading-relaxed`}>
            {text}
          </p>
        </div>
      </div>
    </div>
  );
}

export default RecommendationCard;
