/**
 * PartsList Component
 * Displays a responsive grid of parts needed for repair
 * Mobile-first design with TailwindCSS
 */

import { useMemo } from 'react';

// =============================================================================
// Types
// =============================================================================

export interface PriceSource {
  name: string;
  price_min: number;
  price_max: number;
  currency: string;
  in_stock: boolean;
  delivery_days?: number;
  url?: string;
}

export interface PartInfo {
  id: string;
  name: string;
  name_en?: string;
  category: string;
  part_number?: string;
  oem_number?: string;
  description?: string;
  price_range_min: number;
  price_range_max: number;
  currency: string;
  sources: PriceSource[];
  is_oem?: boolean;
  quality_rating?: number;
  compatibility_notes?: string;
}

export interface VehicleInfo {
  make: string;
  model: string;
  year: number;
  engine?: string;
}

export interface PartsListProps {
  parts: PartInfo[];
  vehicleInfo?: VehicleInfo;
  className?: string;
}

// =============================================================================
// Helper Functions
// =============================================================================

function formatPrice(min: number, max: number, currency: string): string {
  const formatter = new Intl.NumberFormat('hu-HU', {
    style: 'currency',
    currency: currency,
    maximumFractionDigits: 0,
  });
  if (min === max) {
    return formatter.format(min);
  }
  return `${formatter.format(min)} - ${formatter.format(max)}`;
}

function getStockStatusColor(inStock: boolean): string {
  return inStock
    ? 'text-green-600 bg-green-50 border-green-200'
    : 'text-orange-600 bg-orange-50 border-orange-200';
}

function getStockStatusText(inStock: boolean): string {
  return inStock ? 'Készleten' : 'Rendelhető';
}

function getQualityStars(rating: number | undefined): string[] {
  const stars: string[] = [];
  const normalizedRating = rating ?? 3;
  for (let i = 0; i < 5; i++) {
    stars.push(i < normalizedRating ? 'star' : 'star_outline');
  }
  return stars;
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
// Part Card Component
// =============================================================================

interface PartCardProps {
  part: PartInfo;
  vehicleInfo?: VehicleInfo;
}

function PartCard({ part, vehicleInfo }: PartCardProps) {
  const hasMultipleSources = part.sources.length > 1;
  const _bestSource = useMemo(() => {
    if (part.sources.length === 0) return null;
    return part.sources.reduce((best, current) =>
      current.price_min < best.price_min ? current : best
    );
  }, [part.sources]);
  const inStockSources = part.sources.filter((s) => s.in_stock);
  const hasStock = inStockSources.length > 0;

  // Suppress unused variable warning
  void _bestSource;

  return (
    <article
      className="bg-white rounded-2xl border border-slate-200 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden group"
      aria-label={`${part.name} alkatrész`}
    >
      <div className="p-4 md:p-5 border-b border-slate-100">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              {part.is_oem && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-blue-100 text-blue-700 border border-blue-200">
                  OEM
                </span>
              )}
              <span className="text-[10px] font-medium text-slate-400 uppercase tracking-wider">
                {part.category}
              </span>
            </div>
            <h3 className="text-base font-bold text-slate-900 truncate group-hover:text-blue-600 transition-colors">
              {part.name}
            </h3>
            {part.part_number && (
              <p className="text-xs font-mono text-slate-500 mt-0.5">
                {part.part_number}
                {part.oem_number && ` / OEM: ${part.oem_number}`}
              </p>
            )}
          </div>
          {part.quality_rating !== undefined && (
            <div className="flex items-center gap-0.5" role="img" aria-label={`${part.quality_rating} csillag az 5-ből`}>
              {getQualityStars(part.quality_rating).map((icon, idx) => (
                <MaterialIcon key={idx} name={icon} className={`text-sm ${icon === 'star' ? 'text-amber-400' : 'text-slate-300'}`} />
              ))}
            </div>
          )}
        </div>
        {part.description && <p className="text-sm text-slate-600 mt-2 line-clamp-2">{part.description}</p>}
      </div>
      <div className="p-4 md:p-5 bg-slate-50">
        <div className="flex items-center justify-between mb-3">
          <div>
            <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider block mb-0.5">Ár tartomány</span>
            <span className="text-xl font-bold text-slate-900">{formatPrice(part.price_range_min, part.price_range_max, part.currency)}</span>
          </div>
          <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${getStockStatusColor(hasStock)}`}>
            <MaterialIcon name={hasStock ? 'check_circle' : 'schedule'} className="text-sm" />
            {getStockStatusText(hasStock)}
          </div>
        </div>
        {part.sources.length > 0 && (
          <div className="space-y-2">
            <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider">Források ({part.sources.length})</span>
            <div className="space-y-1.5">
              {part.sources.slice(0, 3).map((source, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm bg-white rounded-lg px-3 py-2 border border-slate-100">
                  <div className="flex items-center gap-2">
                    <MaterialIcon name="storefront" className="text-slate-400 text-base" />
                    <span className="font-medium text-slate-700">{source.name}</span>
                    {source.in_stock && <span className="w-2 h-2 rounded-full bg-green-500" title="Készleten" />}
                  </div>
                  <div className="text-right">
                    <span className="font-semibold text-slate-900">{formatPrice(source.price_min, source.price_max, source.currency)}</span>
                    {source.delivery_days && <span className="text-xs text-slate-500 ml-2">{source.delivery_days} nap</span>}
                  </div>
                </div>
              ))}
              {hasMultipleSources && part.sources.length > 3 && (
                <button className="w-full text-center text-sm text-blue-600 hover:text-blue-700 font-medium py-1" aria-label={`További ${part.sources.length - 3} forrás megtekintése`}>
                  + {part.sources.length - 3} további forrás
                </button>
              )}
            </div>
          </div>
        )}
        {part.compatibility_notes && vehicleInfo && (
          <div className="mt-3 p-2.5 rounded-lg bg-amber-50 border border-amber-100">
            <div className="flex items-start gap-2">
              <MaterialIcon name="info" className="text-amber-600 text-base flex-shrink-0 mt-0.5" />
              <p className="text-xs text-amber-800">{part.compatibility_notes}</p>
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

function EmptyState() {
  return (
    <div className="text-center py-12 px-4">
      <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <MaterialIcon name="inventory_2" className="text-3xl text-slate-400" />
      </div>
      <h3 className="text-lg font-semibold text-slate-900 mb-2">Nincsenek alkatrészek</h3>
      <p className="text-sm text-slate-600 max-w-sm mx-auto">Ehhez a diagnózishoz nem szükségesek külön alkatrészek, vagy az alkatrész információk még nem elérhetőek.</p>
    </div>
  );
}

export function PartsList({ parts, vehicleInfo, className = '' }: PartsListProps) {
  if (!parts || parts.length === 0) {
    return <EmptyState />;
  }
  const totalParts = parts.length;
  const oemParts = parts.filter((p) => p.is_oem).length;
  const inStockParts = parts.filter((p) => p.sources.some((s) => s.in_stock)).length;
  return (
    <section className={className} aria-labelledby="parts-list-title">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
        <div>
          <h2 id="parts-list-title" className="text-2xl font-bold text-slate-900 font-['Space_Grotesk',sans-serif]">Szükséges alkatrészek</h2>
          <p className="text-sm text-slate-600 mt-1">{totalParts} alkatrész a javításhoz</p>
        </div>
        <div className="flex items-center gap-4">
          {oemParts > 0 && (
            <div className="flex items-center gap-1.5 text-sm">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
              <span className="text-slate-600">{oemParts} OEM</span>
            </div>
          )}
          <div className="flex items-center gap-1.5 text-sm">
            <span className="w-2.5 h-2.5 rounded-full bg-green-500" />
            <span className="text-slate-600">{inStockParts} készleten</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 md:gap-6" role="list" aria-label="Alkatrész lista">
        {parts.map((part) => (
          <div key={part.id} role="listitem">
            <PartCard part={part} vehicleInfo={vehicleInfo} />
          </div>
        ))}
      </div>
    </section>
  );
}

export default PartsList;
