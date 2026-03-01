/**
 * PartStoreCard Component — Redesigned v2
 *
 * 3-column "Color-Topped Columns" layout for store price comparison.
 * Each store (Bárdi Autó, Unix Autó, AUTODOC) gets a branded color lane.
 * Best price visually elevated with emerald accent + savings badge.
 *
 * Design: Navy theme (#0D1B2A), Space Grotesk, Material Symbols
 * Patterns: Bento card zones, Von Restorff effect, progressive disclosure
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

// =============================================================================
// Store Column — single store in the 3-column comparison
// =============================================================================

interface StoreColumnProps {
  store: StorePricing;
  isBest: boolean;
  savings: number;
}

function StoreColumn({ store, isBest, savings }: StoreColumnProps) {
  return (
    <div className="relative flex flex-col">
      {/* Brand color top strip */}
      <div
        className="h-1 w-full flex-shrink-0"
        style={{ backgroundColor: store.storeLogoColor }}
      />

      <div
        className={`flex-1 flex flex-col items-center px-3 py-4 sm:px-4 sm:py-5 transition-colors duration-200 ${
          isBest ? 'bg-emerald-50/60' : 'bg-white'
        }`}
      >
        {/* Store logo circle */}
        <div
          className="w-9 h-9 sm:w-10 sm:h-10 rounded-full flex items-center justify-center text-white text-xs sm:text-sm font-black shadow-sm flex-shrink-0"
          style={{ backgroundColor: store.storeLogoColor }}
        >
          {store.storeName.charAt(0)}
        </div>

        {/* Store name */}
        <p className="text-[11px] sm:text-xs font-bold text-slate-900 mt-2 text-center leading-tight">
          {store.storeName}
        </p>

        {/* Brand */}
        <p className="text-[10px] text-slate-400 mt-0.5 text-center truncate w-full font-mono" title={store.brand}>
          {store.brand}
        </p>

        {/* Price — hero element */}
        <div className="mt-3 mb-1.5">
          <span
            className={`tabular-nums font-['Space_Grotesk',sans-serif] ${
              isBest
                ? 'text-lg sm:text-xl font-black text-emerald-700'
                : 'text-base sm:text-lg font-bold text-slate-800'
            }`}
          >
            {formatHUF(store.price)}
          </span>
          <span className={`text-[10px] ml-0.5 ${isBest ? 'text-emerald-600' : 'text-slate-400'}`}>
            Ft
          </span>
        </div>

        {/* Best price badge + savings */}
        {isBest && (
          <div className="flex flex-col items-center gap-0.5 mb-1.5">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[9px] sm:text-[10px] font-bold bg-emerald-100 text-emerald-700 border border-emerald-200">
              <MaterialIcon name="trending_down" className="text-[10px]" />
              Legjobb ár
            </span>
            {savings > 0 && (
              <span className="text-[9px] sm:text-[10px] font-medium text-emerald-600">
                -{formatHUF(savings)} Ft
              </span>
            )}
          </div>
        )}

        {/* Price range if available */}
        {store.priceMax && store.priceMax > store.price && !isBest && (
          <p className="text-[9px] text-slate-400 mb-1.5">
            max. {formatHUF(store.priceMax)} Ft
          </p>
        )}

        {/* Stock status */}
        <div className="mt-auto pt-2">
          {store.inStock ? (
            <div className="flex items-center gap-1">
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
              </span>
              <span className="text-[10px] sm:text-[11px] font-medium text-emerald-700">Készleten</span>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <span className="h-1.5 w-1.5 rounded-full bg-amber-400 flex-shrink-0" />
              <span className="text-[10px] sm:text-[11px] font-medium text-amber-700">
                {store.deliveryDays} nap
              </span>
            </div>
          )}
          {store.inStock && store.deliveryDays <= 1 && (
            <p className="text-[9px] text-slate-400 mt-0.5 text-center">holnap kézbesítve</p>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Mobile Store Card — horizontal scroll version
// =============================================================================

function MobileStoreCard({ store, isBest, savings }: StoreColumnProps) {
  return (
    <div
      className={`snap-start flex-shrink-0 w-[70vw] max-w-[260px] rounded-xl overflow-hidden border transition-all duration-200 ${
        isBest
          ? 'border-emerald-200 bg-emerald-50/40 ring-1 ring-emerald-200/60'
          : 'border-slate-100 bg-white'
      }`}
    >
      {/* Brand strip */}
      <div className="h-1.5 w-full" style={{ backgroundColor: store.storeLogoColor }} />

      <div className="p-4 flex flex-col items-center">
        {/* Logo + Name */}
        <div
          className="w-10 h-10 rounded-full flex items-center justify-center text-white text-sm font-black shadow-sm"
          style={{ backgroundColor: store.storeLogoColor }}
        >
          {store.storeName.charAt(0)}
        </div>
        <p className="text-xs font-bold text-slate-900 mt-2">{store.storeName}</p>
        <p className="text-[10px] text-slate-400 font-mono truncate w-full text-center" title={store.brand}>
          {store.brand}
        </p>

        {/* Price */}
        <div className="mt-3 mb-2">
          <span
            className={`tabular-nums font-['Space_Grotesk',sans-serif] ${
              isBest ? 'text-xl font-black text-emerald-700' : 'text-lg font-bold text-slate-800'
            }`}
          >
            {formatHUF(store.price)}
          </span>
          <span className={`text-[10px] ml-0.5 ${isBest ? 'text-emerald-600' : 'text-slate-400'}`}>Ft</span>
        </div>

        {isBest && savings > 0 && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-700 border border-emerald-200 mb-2">
            <MaterialIcon name="trending_down" className="text-[10px]" />
            -{formatHUF(savings)} Ft olcsóbb
          </span>
        )}

        {/* Stock */}
        <div className="flex items-center gap-1 mt-auto">
          {store.inStock ? (
            <>
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
              </span>
              <span className="text-[11px] font-medium text-emerald-700">Készleten</span>
            </>
          ) : (
            <>
              <span className="h-1.5 w-1.5 rounded-full bg-amber-400" />
              <span className="text-[11px] font-medium text-amber-700">Rendelhető · {store.deliveryDays} nap</span>
            </>
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

      {/* ═══════════════════════════════════════════════════════════════════
          ZONE B: 3-Column Store Price Comparison
          ═══════════════════════════════════════════════════════════════════ */}

      {/* Desktop: 3-column grid */}
      <div className="hidden sm:grid grid-cols-3 divide-x divide-slate-100 border-t border-slate-100">
        {part.stores.map((store) => (
          <StoreColumn
            key={store.storeName}
            store={store}
            isBest={bestPrice !== null && store.storeName === bestPrice.storeName}
            savings={savings}
          />
        ))}
      </div>

      {/* Mobile: horizontal snap scroll */}
      <div className="sm:hidden border-t border-slate-100 bg-slate-50/50">
        <div className="px-4 pt-3 pb-1">
          <span className="text-[10px] font-bold uppercase text-slate-400 tracking-widest">
            Ár-összehasonlítás
          </span>
        </div>
        <div className="flex gap-3 overflow-x-auto snap-x snap-mandatory pb-4 px-4 scrollbar-hide">
          {part.stores.map((store) => (
            <MobileStoreCard
              key={store.storeName}
              store={store}
              isBest={bestPrice !== null && store.storeName === bestPrice.storeName}
              savings={savings}
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
