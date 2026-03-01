/**
 * PartStoreCard Component
 *
 * Kártyás megjelenítés alkatrészekhez bolt-specifikus árakkal.
 * Megjeleníti a Bárdi Autó, Uni Autó és AUTODOC árakat egymás mellett.
 * Design: Navy theme (#0D1B2A), Space Grotesk font, Material Symbols icons
 */

import type { DemoPartWithStores, StorePricing } from '../../../data/demoData';

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
// Helper Functions
// =============================================================================

function formatHUF(amount: number): string {
  return new Intl.NumberFormat('hu-HU', {
    maximumFractionDigits: 0,
  }).format(amount);
}

function getQualityStars(rating: number): JSX.Element[] {
  const stars: JSX.Element[] = [];
  const full = Math.floor(rating);
  const hasHalf = rating - full >= 0.3;

  for (let i = 0; i < 5; i++) {
    if (i < full) {
      stars.push(
        <MaterialIcon key={i} name="star" className="text-sm text-amber-400" />
      );
    } else if (i === full && hasHalf) {
      stars.push(
        <MaterialIcon key={i} name="star_half" className="text-sm text-amber-400" />
      );
    } else {
      stars.push(
        <MaterialIcon key={i} name="star" className="text-sm text-slate-300" />
      );
    }
  }
  return stars;
}

function getBestPrice(stores: StorePricing[]): StorePricing | null {
  if (stores.length === 0) return null;
  return stores.reduce((best, current) =>
    current.price < best.price ? current : best
  );
}

// =============================================================================
// Store Price Row Component
// =============================================================================

interface StorePriceRowProps {
  store: StorePricing;
  isBest: boolean;
}

function StorePriceRow({ store, isBest }: StorePriceRowProps) {
  return (
    <div
      className={`flex items-center justify-between px-4 py-3 rounded-xl border transition-all duration-200 ${
        isBest
          ? 'bg-green-50 border-green-200 ring-1 ring-green-200'
          : 'bg-white border-slate-100 hover:border-slate-200'
      }`}
    >
      <div className="flex items-center gap-3 min-w-0">
        {/* Store logo circle */}
        <div
          className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-white text-[10px] font-black"
          style={{ backgroundColor: store.storeLogoColor }}
        >
          {store.storeName.charAt(0)}
        </div>

        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-slate-900 truncate">
              {store.storeName}
            </span>
            {isBest && (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider bg-green-100 text-green-700 border border-green-200">
                Legjobb ár
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[10px] text-slate-500 truncate">{store.brand}</span>
            <span className="text-slate-300">·</span>
            {store.inStock ? (
              <span className="flex items-center gap-0.5 text-[10px] text-green-600 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                Készleten
              </span>
            ) : (
              <span className="flex items-center gap-0.5 text-[10px] text-orange-600 font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-orange-400" />
                {store.deliveryDays} nap
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="text-right flex-shrink-0 ml-3">
        <div className="text-base font-bold text-slate-900">
          {formatHUF(store.price)} Ft
        </div>
        {store.priceMax && store.priceMax > store.price && (
          <div className="text-[10px] text-slate-400">
            max. {formatHUF(store.priceMax)} Ft
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// Main PartStoreCard Component
// =============================================================================

interface PartStoreCardProps {
  part: DemoPartWithStores;
  index: number;
}

export function PartStoreCard({ part, index }: PartStoreCardProps) {
  const bestPrice = getBestPrice(part.stores);

  return (
    <article className="bg-white rounded-2xl border border-slate-200 shadow-sm hover:shadow-lg transition-all duration-300 overflow-hidden group">
      {/* Card Header */}
      <div className="p-5 pb-4 border-b border-slate-100">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            {/* Category + Index badge */}
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-[#0D1B2A] text-white text-[10px] font-bold">
                {index + 1}
              </span>
              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-blue-50 text-blue-700 border border-blue-100">
                {part.category}
              </span>
              {part.isOem && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-amber-50 text-amber-700 border border-amber-100">
                  OEM
                </span>
              )}
            </div>

            {/* Part Name */}
            <h3 className="text-lg font-bold text-slate-900 group-hover:text-blue-600 transition-colors font-['Space_Grotesk',sans-serif]">
              {part.name}
            </h3>
            {part.name_en && (
              <p className="text-xs text-slate-400 mt-0.5">{part.name_en}</p>
            )}

            {/* Part numbers */}
            <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500 font-mono">
              <span title="Utángyártott cikkszám">
                <MaterialIcon name="tag" className="text-xs inline-block mr-0.5 align-middle" />
                {part.partNumber}
              </span>
              <span title="OEM cikkszám">
                <MaterialIcon name="verified" className="text-xs inline-block mr-0.5 align-middle" />
                OEM: {part.oemNumber}
              </span>
            </div>
          </div>

          {/* Quality Rating */}
          <div className="flex flex-col items-end gap-1">
            <div className="flex items-center gap-0.5" title={`${part.qualityRating} / 5`}>
              {getQualityStars(part.qualityRating)}
            </div>
            <span className="text-[10px] text-slate-400 font-medium">
              {part.qualityRating.toFixed(1)}/5
            </span>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-slate-600 leading-relaxed mt-3">
          {part.description}
        </p>
      </div>

      {/* Store Prices Section */}
      <div className="p-5 bg-slate-50/50">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-[10px] font-bold uppercase text-slate-400 tracking-widest flex items-center gap-1.5">
            <MaterialIcon name="storefront" className="text-sm" />
            Elérhető boltok ({part.stores.length})
          </h4>
          <div className="text-right">
            <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider">
              Ár tartomány
            </span>
            <div className="text-sm font-bold text-slate-900">
              {formatHUF(part.price_range_min)} – {formatHUF(part.price_range_max)} Ft
            </div>
          </div>
        </div>

        {/* Store price rows */}
        <div className="space-y-2">
          {part.stores.map((store, idx) => (
            <StorePriceRow
              key={idx}
              store={store}
              isBest={bestPrice !== null && store.price === bestPrice.price}
            />
          ))}
        </div>
      </div>

      {/* Footer with labor info + compatibility */}
      <div className="px-5 py-3 border-t border-slate-100 bg-white">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1 text-slate-500">
              <MaterialIcon name="schedule" className="text-sm text-slate-400" />
              Beszerelés: ~{part.labor_hours} óra
            </span>
          </div>
          {part.compatibilityNote && (
            <span className="text-slate-400 truncate max-w-[50%]" title={part.compatibilityNote}>
              <MaterialIcon name="check_circle" className="text-sm text-green-500 inline-block mr-0.5 align-middle" />
              Kompatibilis
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

// =============================================================================
// PartStoreCardGrid - Grid wrapper for multiple cards
// =============================================================================

interface PartStoreCardGridProps {
  parts: DemoPartWithStores[];
  className?: string;
}

export function PartStoreCardGrid({ parts, className = '' }: PartStoreCardGridProps) {
  if (!parts || parts.length === 0) return null;

  // Stat calculations
  const inStockCount = parts.filter((p) =>
    p.stores.some((s) => s.inStock)
  ).length;
  const totalMinPrice = parts.reduce((sum, p) => sum + p.price_range_min, 0);
  const totalMaxPrice = parts.reduce((sum, p) => sum + p.price_range_max, 0);

  return (
    <section className={className}>
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-[#0D1B2A] text-white shadow-lg shadow-[#0D1B2A]/20">
              <MaterialIcon name="shopping_cart" className="text-2xl" />
            </div>
            <h3 className="text-2xl font-bold text-slate-900 font-['Space_Grotesk',sans-serif]">
              Szükséges alkatrészek
            </h3>
          </div>
          <p className="text-sm text-slate-600 ml-[52px]">
            {parts.length} alkatrész · {inStockCount} készleten · Összesen{' '}
            <span className="font-bold text-slate-900">{formatHUF(totalMinPrice)} – {formatHUF(totalMaxPrice)} Ft</span>
          </p>
        </div>

        {/* Store legend */}
        <div className="flex items-center gap-3 flex-wrap">
          {['Bárdi Autó', 'Uni Autó', 'AUTODOC'].map((name) => {
            const store = parts[0]?.stores.find((s) => s.storeName === name);
            if (!store) return null;
            return (
              <div key={name} className="flex items-center gap-1.5">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: store.storeLogoColor }}
                />
                <span className="text-xs text-slate-600 font-medium">{name}</span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Cards Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {parts.map((part, index) => (
          <PartStoreCard key={part.id} part={part} index={index} />
        ))}
      </div>
    </section>
  );
}

export default PartStoreCard;
