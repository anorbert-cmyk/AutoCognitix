/**
 * PartStoreCard Component
 *
 * Demo pricing card for 3-store comparison (Bárdi, Unix, AUTODOC).
 * Layout is optimized for fast scanning: store, price, stock, delivery, delta.
 */

import type { DemoPartWithStores, StorePricing } from '../../../data/demoData';
import { MaterialIcon } from '../../ui/MaterialIcon';

// =============================================================================
// Helpers
// =============================================================================

const hufFormatter = new Intl.NumberFormat('hu-HU', {
  maximumFractionDigits: 0,
});

function formatHUF(amount: number): string {
  return hufFormatter.format(amount);
}

function getQualityStars(rating: number): React.ReactElement[] {
  const stars: React.ReactElement[] = [];
  const full = Math.floor(rating);
  const hasHalf = rating - full >= 0.3;

  for (let i = 0; i < 5; i++) {
    if (i < full) {
      stars.push(<MaterialIcon key={i} name="star" className="text-sm text-amber-400" />);
    } else if (i === full && hasHalf) {
      stars.push(<MaterialIcon key={i} name="star_half" className="text-sm text-amber-400" />);
    } else {
      stars.push(<MaterialIcon key={i} name="star" className="text-sm text-slate-200" />);
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

function getMaxPrice(stores: StorePricing[]): number {
  return Math.max(...stores.map((s) => s.price));
}

interface StoreComparisonRowProps {
  store: StorePricing;
  isBest: boolean;
  deltaFromBest: number;
}

function StoreComparisonRow({ store, isBest, deltaFromBest }: StoreComparisonRowProps) {
  const hasPriceRange = Boolean(store.priceMax && store.priceMax > store.price);

  return (
    <div
      className={`px-4 py-3 sm:px-5 transition-colors ${
        isBest ? 'bg-emerald-50/70' : 'bg-white hover:bg-slate-50/80'
      }`}
    >
      <div className="flex flex-col gap-2 sm:grid sm:grid-cols-[minmax(0,1.6fr)_auto_auto_auto] sm:items-center sm:gap-4">
        <div className="min-w-0 flex items-center gap-2.5">
          <span
            className="h-2.5 w-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: store.storeLogoColor }}
          />
          <div className="min-w-0">
            <p className="text-sm font-semibold text-slate-900">{store.storeName}</p>
            <p className="text-[11px] text-slate-500 font-mono truncate" title={store.brand}>
              {store.brand}
            </p>
          </div>
        </div>

        <div className="flex items-baseline gap-1 sm:justify-self-end">
          <span
            className={`tabular-nums font-['Space_Grotesk',sans-serif] ${
              isBest ? 'text-lg font-black text-emerald-700' : 'text-base font-bold text-slate-900'
            }`}
          >
            {formatHUF(store.price)}
          </span>
          <span className={`text-[10px] font-semibold ${isBest ? 'text-emerald-600' : 'text-slate-500'}`}>
            Ft
          </span>
          {hasPriceRange && (
            <span className="ml-1.5 text-[10px] text-slate-400">
              max. {formatHUF(store.priceMax!)}
            </span>
          )}
        </div>

        <div className="text-[11px] sm:justify-self-end">
          {store.inStock ? (
            <span className="inline-flex items-center gap-1.5 font-medium text-emerald-700">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              Készleten
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 font-medium text-amber-700">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
              Rendelhető
            </span>
          )}
        </div>

        <div className="flex items-center justify-between sm:justify-self-end sm:gap-2">
          <span className="text-[11px] text-slate-500">{store.deliveryDays} nap</span>
          {isBest ? (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-700 border border-emerald-200">
              <MaterialIcon name="workspace_premium" className="text-[11px]" />
              Legjobb ár
            </span>
          ) : deltaFromBest === 0 ? (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 text-slate-600">
              Azonos ár
            </span>
          ) : (
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-100 text-slate-600">
              +{formatHUF(deltaFromBest)} Ft
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Main PartStoreCard
// =============================================================================

interface PartStoreCardProps {
  part: DemoPartWithStores;
  index: number;
}

export function PartStoreCard({ part, index }: PartStoreCardProps) {
  const bestPrice = getBestPrice(part.stores);
  const maxPrice = getMaxPrice(part.stores);
  const savings = bestPrice ? maxPrice - bestPrice.price : 0;
  const sortedStores = [...part.stores].sort((a, b) => a.price - b.price);

  return (
    <article className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden transition-all duration-300 hover:-translate-y-0.5 hover:shadow-lg hover:border-slate-300 group">
      {/* ═══════════════════════════════════════════════════════════════════
          ZONE A: Part Identity
          ═══════════════════════════════════════════════════════════════════ */}
      <div className="p-5 pb-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            {/* Badges row */}
            <div className="flex items-center gap-2 mb-2 flex-wrap">
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

            {/* Part name */}
            <h3 className="text-lg font-bold text-slate-900 group-hover:text-[#0D1B2A] transition-colors font-['Space_Grotesk',sans-serif]">
              {part.name}
            </h3>
            {part.name_en && (
              <p className="text-[11px] text-slate-400 mt-0.5">{part.name_en}</p>
            )}

            {/* Part numbers */}
            <div className="flex items-center gap-3 mt-2 text-[10px] text-slate-500 font-mono">
              <span title="Utángyártott cikkszám">
                <MaterialIcon name="tag" className="text-xs inline-block mr-0.5 align-middle" />
                {part.partNumber}
              </span>
              <span title="OEM cikkszám">
                <MaterialIcon name="verified" className="text-xs inline-block mr-0.5 align-middle text-blue-400" />
                OEM: {part.oemNumber}
              </span>
            </div>
          </div>

          {/* Quality rating */}
          <div className="flex flex-col items-end gap-1 flex-shrink-0">
            <div className="flex items-center gap-0.5" title={`${part.qualityRating} / 5`}>
              {getQualityStars(part.qualityRating)}
            </div>
            <span className="text-[10px] text-slate-400 font-medium tabular-nums">
              {part.qualityRating.toFixed(1)}/5
            </span>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-slate-600 leading-relaxed mt-3 line-clamp-2">
          {part.description}
        </p>
      </div>

      {/* Ár-összehasonlítás */}
      <div className="border-t border-slate-100">
        <div className="px-5 py-3 bg-slate-50 border-b border-slate-100">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1.5">
            <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
              Ár-összehasonlítás · 3 bolt
            </span>
            {savings > 0 && (
              <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-emerald-700">
                <MaterialIcon name="savings" className="text-sm" />
                Max. különbség: {formatHUF(savings)} Ft
              </span>
            )}
          </div>
        </div>

        <div className="divide-y divide-slate-100">
          {sortedStores.map((store) => (
            <StoreComparisonRow
              key={store.storeName}
              store={store}
              isBest={bestPrice !== null && store.storeName === bestPrice.storeName}
              deltaFromBest={bestPrice ? store.price - bestPrice.price : 0}
            />
          ))}
        </div>
      </div>

      {/* ═══════════════════════════════════════════════════════════════════
          ZONE C: Footer — labor, compatibility, price range
          ═══════════════════════════════════════════════════════════════════ */}
      <div className="px-5 py-3 border-t border-slate-100 bg-white">
        <div className="flex items-center justify-between text-[11px]">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-slate-500">
              <MaterialIcon name="schedule" className="text-sm text-slate-400" />
              Beszerelés: ~{part.labor_hours} óra
            </span>
            <span className="hidden sm:flex items-center gap-1.5 text-slate-400">
              <MaterialIcon name="payments" className="text-sm" />
              {formatHUF(part.price_range_min)} – {formatHUF(part.price_range_max)} Ft
            </span>
          </div>
          {part.compatibilityNote && (
            <span className="flex items-center gap-1 text-emerald-600 font-medium" title={part.compatibilityNote}>
              <MaterialIcon name="check_circle" className="text-sm" />
              <span className="hidden sm:inline">Kompatibilis</span>
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

// =============================================================================
// PartStoreCardGrid — Section wrapper with header + stats
// =============================================================================

interface PartStoreCardGridProps {
  parts: DemoPartWithStores[];
  className?: string;
}

export function PartStoreCardGrid({ parts, className = '' }: PartStoreCardGridProps) {
  if (!parts || parts.length === 0) return null;

  const inStockCount = parts.filter((p) => p.stores.some((s) => s.inStock)).length;
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
        <div className="flex items-center gap-4 flex-wrap">
          {['Bárdi Autó', 'Unix Autó', 'AUTODOC'].map((name) => {
            const store = parts[0]?.stores.find((s) => s.storeName === name);
            if (!store) return null;
            return (
              <div key={name} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full ring-2 ring-white shadow-sm"
                  style={{ backgroundColor: store.storeLogoColor }}
                />
                <span className="text-xs text-slate-700 font-semibold">{name}</span>
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
