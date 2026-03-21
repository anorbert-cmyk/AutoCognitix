/**
 * ValueComparison Component
 *
 * Visual bar comparison between vehicle value and repair cost.
 * Animated horizontal bars with ratio percentage display.
 */

import { useMemo } from 'react';
import { Car, Wrench } from 'lucide-react';

interface ValueComparisonProps {
  vehicleValue: number;
  repairCost: number;
  currency: string;
}

function formatValue(amount: number, currency: string): string {
  return new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(amount);
}

export function ValueComparison({ vehicleValue, repairCost, currency }: ValueComparisonProps) {
  const maxValue = useMemo(() => Math.max(vehicleValue, repairCost, 1), [vehicleValue, repairCost]);

  const vehiclePercent = useMemo(
    () => Math.round((vehicleValue / maxValue) * 100),
    [vehicleValue, maxValue]
  );
  const repairPercent = useMemo(
    () => Math.round((repairCost / maxValue) * 100),
    [repairCost, maxValue]
  );

  const ratio = useMemo(
    () => (vehicleValue > 0 ? Math.round((repairCost / vehicleValue) * 100) : 0),
    [repairCost, vehicleValue]
  );

  const ratioColor = ratio <= 40 ? 'text-green-600' : ratio <= 70 ? 'text-yellow-600' : 'text-red-600';

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-5 sm:p-6 shadow-sm">
      <div className="flex items-center justify-between mb-5">
        <h3 className="text-base font-bold text-slate-900">
          Ertek vs. Javitasi koltseg
        </h3>
        <span className={`text-sm font-bold ${ratioColor} tabular-nums`}>
          Arany: {ratio}%
        </span>
      </div>

      <div className="space-y-5">
        {/* Vehicle Value Bar */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-slate-700">
              <Car className="h-4 w-4 text-blue-500" />
              <span className="font-medium">Jarmu erteke</span>
            </div>
            <span className="text-sm font-bold text-slate-900 tabular-nums">
              {formatValue(vehicleValue, currency)}
            </span>
          </div>
          <div className="h-6 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-400 to-blue-600 rounded-full transition-all duration-1000 ease-out"
              style={{ width: `${vehiclePercent}%` }}
            />
          </div>
        </div>

        {/* Repair Cost Bar */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2 text-sm text-slate-700">
              <Wrench className="h-4 w-4 text-orange-500" />
              <span className="font-medium">Javitasi koltseg</span>
            </div>
            <span className="text-sm font-bold text-slate-900 tabular-nums">
              {formatValue(repairCost, currency)}
            </span>
          </div>
          <div className="h-6 bg-slate-100 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-1000 ease-out ${
                ratio <= 40
                  ? 'bg-gradient-to-r from-green-400 to-green-600'
                  : ratio <= 70
                  ? 'bg-gradient-to-r from-yellow-400 to-orange-500'
                  : 'bg-gradient-to-r from-orange-400 to-red-600'
              }`}
              style={{ width: `${repairPercent}%` }}
            />
          </div>
        </div>
      </div>

      {/* Ratio Explanation */}
      <p className="mt-4 text-xs text-slate-500">
        A javitasi koltseg a jarmu ertekenek <strong className={ratioColor}>{ratio}%-a</strong>.
        {ratio <= 40 && ' Ez kedvezo arany, erdemes megjavitani.'}
        {ratio > 40 && ratio <= 70 && ' Ez kozepes arany, megfontolando.'}
        {ratio > 70 && ' Ez magas arany, erdemes mast megoldast keresni.'}
      </p>
    </div>
  );
}

export default ValueComparison;
