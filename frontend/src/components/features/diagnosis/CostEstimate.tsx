/**
 * CostEstimate Component
 * Displays repair cost breakdown with visual chart
 * Mobile-first responsive design
 */

import { useMemo } from 'react';

// =============================================================================
// Types
// =============================================================================

export interface RepairCostEstimate {
  dtc_code?: string;
  repair_name: string;
  repair_description?: string;
  parts_cost_min: number;
  parts_cost_max: number;
  labor_cost_min: number;
  labor_cost_max: number;
  total_cost_min: number;
  total_cost_max: number;
  currency: string;
  estimated_hours: number;
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  confidence: number;
  notes?: string;
  vehicle_info?: string;
  disclaimer?: string;
}

export interface CostEstimateProps {
  estimate: RepairCostEstimate;
  className?: string;
}

// =============================================================================
// Helper Functions
// =============================================================================

function formatPrice(amount: number, currency: string): string {
  return new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency: currency,
    maximumFractionDigits: 0,
  }).format(amount);
}

function formatPriceRange(min: number, max: number, currency: string): string {
  if (min === max) {
    return formatPrice(min, currency);
  }
  return `${formatPrice(min, currency)} - ${formatPrice(max, currency)}`;
}

function getDifficultyConfig(difficulty: string): { label: string; color: string; icon: string } {
  switch (difficulty) {
    case 'easy':
      return { label: 'Egyszer\u0171', color: 'text-green-600 bg-green-50 border-green-200', icon: 'thumb_up' };
    case 'medium':
      return { label: 'K\u00f6zepes', color: 'text-amber-600 bg-amber-50 border-amber-200', icon: 'engineering' };
    case 'hard':
      return { label: 'Neh\u00e9z', color: 'text-orange-600 bg-orange-50 border-orange-200', icon: 'warning' };
    case 'expert':
      return { label: 'Szak\u00e9rt\u0151i', color: 'text-red-600 bg-red-50 border-red-200', icon: 'precision_manufacturing' };
    default:
      return { label: 'Ismeretlen', color: 'text-slate-600 bg-slate-50 border-slate-200', icon: 'help' };
  }
}

function getConfidenceLevel(confidence: number): { label: string; color: string } {
  if (confidence >= 0.8) return { label: 'Magas', color: 'text-green-600' };
  if (confidence >= 0.5) return { label: 'K\u00f6zepes', color: 'text-amber-600' };
  return { label: 'Alacsony', color: 'text-red-600' };
}

// =============================================================================
// Material Icon Component
// =============================================================================

function MaterialIcon({ name, className = '' }: { name: string; className?: string }) {
  return (
    <span
      className={`material-symbols-outlined ${className}`}
      style={{ fontVariationSettings: "'FILL' 0, 'wght' 300, 'GRAD' 0, 'opsz' 24" }}
      aria-hidden="true"
    >
      {name}
    </span>
  );
}

// =============================================================================
// Cost Bar Chart Component
// =============================================================================

interface CostBarProps {
  partsPercent: number;
  laborPercent: number;
}

function CostBar({ partsPercent, laborPercent }: CostBarProps) {
  return (
    <div className="w-full h-4 rounded-full overflow-hidden bg-slate-100 flex">
      <div
        className="h-full bg-blue-500 transition-all duration-500"
        style={{ width: `${partsPercent}%` }}
        title={`Alkatr\u00e9szek: ${partsPercent.toFixed(0)}%`}
      />
      <div
        className="h-full bg-emerald-500 transition-all duration-500"
        style={{ width: `${laborPercent}%` }}
        title={`Munkad\u00edj: ${laborPercent.toFixed(0)}%`}
      />
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

export function CostEstimate({ estimate, className = '' }: CostEstimateProps) {
  const difficultyConfig = getDifficultyConfig(estimate.difficulty);
  const confidenceLevel = getConfidenceLevel(estimate.confidence);

  // Calculate percentages for the bar chart
  const { partsPercent, laborPercent } = useMemo(() => {
    const avgParts = (estimate.parts_cost_min + estimate.parts_cost_max) / 2;
    const avgLabor = (estimate.labor_cost_min + estimate.labor_cost_max) / 2;
    const total = avgParts + avgLabor;
    if (total === 0) return { partsPercent: 50, laborPercent: 50 };
    return {
      partsPercent: (avgParts / total) * 100,
      laborPercent: (avgLabor / total) * 100,
    };
  }, [estimate]);

  return (
    <section className={`bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden ${className}`} aria-labelledby="cost-estimate-title">
      {/* Header */}
      <div className="p-5 md:p-6 bg-gradient-to-r from-slate-900 to-slate-800 text-white">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 id="cost-estimate-title" className="text-xl md:text-2xl font-bold font-['Space_Grotesk',sans-serif]">
              Becs\u00fclt jav\u00edt\u00e1si k\u00f6lts\u00e9g
            </h2>
            <p className="text-slate-300 text-sm mt-1">{estimate.repair_name}</p>
            {estimate.vehicle_info && (
              <p className="text-slate-400 text-xs mt-1">{estimate.vehicle_info}</p>
            )}
          </div>
          {estimate.dtc_code && (
            <span className="px-3 py-1.5 bg-white/10 rounded-lg text-sm font-mono font-medium">
              {estimate.dtc_code}
            </span>
          )}
        </div>
      </div>

      {/* Main Cost Display */}
      <div className="p-5 md:p-6 border-b border-slate-100">
        <div className="text-center mb-6">
          <span className="text-sm text-slate-500 uppercase tracking-wider font-semibold block mb-2">
            \u00d6sszes k\u00f6lts\u00e9g
          </span>
          <span className="text-3xl md:text-4xl font-bold text-slate-900">
            {formatPriceRange(estimate.total_cost_min, estimate.total_cost_max, estimate.currency)}
          </span>
        </div>

        {/* Cost Breakdown Bar */}
        <div className="mb-4">
          <CostBar partsPercent={partsPercent} laborPercent={laborPercent} />
        </div>

        {/* Legend */}
        <div className="flex justify-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-blue-500" />
            <span className="text-slate-600">Alkatr\u00e9szek ({partsPercent.toFixed(0)}%)</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-emerald-500" />
            <span className="text-slate-600">Munkad\u00edj ({laborPercent.toFixed(0)}%)</span>
          </div>
        </div>
      </div>

      {/* Cost Details Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-slate-100">
        {/* Parts Cost */}
        <div className="p-5 md:p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-blue-50 flex items-center justify-center">
              <MaterialIcon name="inventory_2" className="text-blue-600 text-xl" />
            </div>
            <div>
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold block">
                Alkatr\u00e9szek
              </span>
              <span className="text-lg font-bold text-slate-900">
                {formatPriceRange(estimate.parts_cost_min, estimate.parts_cost_max, estimate.currency)}
              </span>
            </div>
          </div>
        </div>

        {/* Labor Cost */}
        <div className="p-5 md:p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center">
              <MaterialIcon name="construction" className="text-emerald-600 text-xl" />
            </div>
            <div>
              <span className="text-xs text-slate-500 uppercase tracking-wider font-semibold block">
                Munkad\u00edj (~{estimate.estimated_hours.toFixed(1)} \u00f3ra)
              </span>
              <span className="text-lg font-bold text-slate-900">
                {formatPriceRange(estimate.labor_cost_min, estimate.labor_cost_max, estimate.currency)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Metadata */}
      <div className="p-5 md:p-6 bg-slate-50 border-t border-slate-100">
        <div className="flex flex-wrap items-center gap-3 mb-4">
          {/* Difficulty Badge */}
          <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${difficultyConfig.color}`}>
            <MaterialIcon name={difficultyConfig.icon} className="text-sm" />
            Neh\u00e9zs\u00e9g: {difficultyConfig.label}
          </div>

          {/* Confidence Badge */}
          <div className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold bg-white border border-slate-200 ${confidenceLevel.color}`}>
            <MaterialIcon name="verified" className="text-sm" />
            Megb\u00edzhat\u00f3s\u00e1g: {confidenceLevel.label} ({(estimate.confidence * 100).toFixed(0)}%)
          </div>
        </div>

        {/* Notes */}
        {estimate.notes && (
          <div className="p-3 rounded-lg bg-amber-50 border border-amber-100 mb-4">
            <div className="flex items-start gap-2">
              <MaterialIcon name="lightbulb" className="text-amber-600 text-base flex-shrink-0 mt-0.5" />
              <p className="text-sm text-amber-800">{estimate.notes}</p>
            </div>
          </div>
        )}

        {/* Disclaimer */}
        {estimate.disclaimer && (
          <p className="text-xs text-slate-500 italic">
            <MaterialIcon name="info" className="text-xs inline-block mr-1" />
            {estimate.disclaimer}
          </p>
        )}
      </div>
    </section>
  );
}

export default CostEstimate;
