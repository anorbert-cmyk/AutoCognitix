/**
 * PartStoreCard Component — Redesigned v3
 *
 * Row-based "Price Ladder" bento layout for store comparison.
 * Navy header ties into page theme. Best price gets hero treatment.
 * Secondary stores shown as compact comparison rows.
 *
 * Design: Navy theme (#0D1B2A), Space Grotesk, Material Symbols
 */

import { useMemo } from 'react';
import type { DemoPartWithStores, StorePricing } from '../../../data/demoData';
import { MaterialIcon } from '../../ui/MaterialIcon';

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
      stars.push(<MaterialIcon key={i} name="star" className="text-[13px] text-amber-400" />);
    } else if (i === full && hasHalf) {
      stars.push(<MaterialIcon key={i} name="star_half" className="text-[13px] text-amber-400" />);
    } else {
      stars.push(<MaterialIcon key={i} name="star" className="text-[13px] text-white/20" />);
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

function StockBadge({ store }: { store: StorePricing }) {
  if (store.inStock) {
    return (
      <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-emerald-700">
        <span className="relative flex h-1.5 w-1.5">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
        </span>
        Készleten
        {store.deliveryDays <= 1 && (
          <span className="text-emerald-500 font-normal">· holnap</span>
        )}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] font-medium text-amber-600">
      <span className="h-1.5 w-1.5 rounded-full bg-amber-400 flex-shrink-0" />
      Rendelhető · {store.deliveryDays} nap
    </span>
  );
}

// =============================================================================
// Best Price Hero — elevated card for the cheapest store
// =============================================================================

interface BestPriceProps {
  store: StorePricing;
  savings: number;
}

function BestPriceHero({ store, savings }: BestPriceProps) {
  return (
    <div className="mx-4 sm:mx-5 -mt-4 relative z-10">
      <div className="rounded-xl bg-gradient-to-br from-emerald-50 to-emerald-50/50 border border-emerald-200/80 p-4 sm:p-5 shadow-sm shadow-emerald-100/50">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-2.5 min-w-0">
            <div
              className="w-2 h-2 rounded-full flex-shrink-0 ring-2 ring-white shadow-sm"
              style={{ backgroundColor: store.storeLogoColor }}
            />
            <span className="text-sm font-bold text-slate-900 truncate">{store.storeName}</span>
            <span className="text-[10px] text-slate-400 font-mono truncate hidden sm:inline">{store.brand}</span>
          </div>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-600 text-white flex-shrink-0">
            <MaterialIcon name="arrow_downward" className="text-[10px]" />
            Legjobb ár
          </span>
        </div>

        <div className="flex items-end justify-between gap-4">
          <div>
            <span className="text-2xl sm:text-3xl font-black text-emerald-800 tabular-nums font-['Space_Grotesk',sans-serif] tracking-tight">
              {formatHUF(store.price)}
            </span>
            <span className="text-sm font-bold text-emerald-600 ml-1">Ft</span>
          </div>
          <div className="flex flex-col items-end gap-1">
            {savings > 0 && (
              <span className="text-[11px] font-bold text-emerald-700 tabular-nums">
                -{formatHUF(savings)} Ft
              </span>
            )}
            <StockBadge store={store} />
          </div>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Secondary Store Row — compact row for non-best stores
// =============================================================================

function SecondaryStoreRow({ store }: { store: StorePricing }) {
  return (
    <div className="flex items-center justify-between gap-3 py-3 group/row hover:bg-slate-50/80 transition-colors px-4 sm:px-5">
      <div className="flex items-center gap-2.5 min-w-0 flex-1">
        <div
          className="w-2 h-2 rounded-full flex-shrink-0 ring-2 ring-white shadow-sm"
          style={{ backgroundColor: store.storeLogoColor }}
        />
        <span className="text-[13px] font-semibold text-slate-700 flex-shrink-0">{store.storeName}</span>
        <span className="text-[10px] text-slate-400 font-mono truncate hidden sm:inline">{store.brand}</span>
      </div>
      <div className="flex items-center gap-4 flex-shrink-0">
        <StockBadge store={store} />
        <span className="text-[15px] font-bold text-slate-800 tabular-nums font-['Space_Grotesk',sans-serif] min-w-[80px] text-right">
          {formatHUF(store.price)}
          <span className="text-[10px] font-medium text-slate-400 ml-0.5">Ft</span>
        </span>
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
  const bestPrice = useMemo(() => getBestPrice(part.stores), [part.stores]);
  const maxPrice = useMemo(() => getMaxPrice(part.stores), [part.stores]);
  const savings = bestPrice ? maxPrice - bestPrice.price : 0;

  const secondaryStores = part.stores.filter(
    (s) => !bestPrice || s.storeName !== bestPrice.storeName
  );

  return (
    <article className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden transition-all duration-300 hover:shadow-lg hover:border-slate-300 group">
      {/* ═══ ZONE A: Navy Header — Part Identity ═══ */}
      <div className="bg-[#0D1B2A] px-4 sm:px-5 py-4 sm:py-5 pb-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-40 h-40 bg-blue-500/5 rounded-full blur-[40px] -translate-y-1/2 translate-x-1/3" />

        <div className="relative z-10">
          <div className="flex items-start justify-between gap-3 mb-2.5">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="inline-flex items-center justify-center w-6 h-6 rounded-lg bg-white/10 text-white text-[10px] font-bold border border-white/10">
                {index + 1}
              </span>
              <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider bg-white/10 text-blue-200 border border-white/5">
                {part.category}
              </span>
              {part.isOem && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider bg-amber-500/20 text-amber-300 border border-amber-500/10">
                  OEM
                </span>
              )}
            </div>

            <div className="flex items-center gap-1.5 flex-shrink-0" title={`${part.qualityRating} / 5`}>
              <div className="flex items-center gap-0">{getQualityStars(part.qualityRating)}</div>
              <span className="text-[10px] text-white/40 font-medium tabular-nums ml-0.5">
                {part.qualityRating.toFixed(1)}
              </span>
            </div>
          </div>

          <h3 className="text-base sm:text-lg font-bold text-white group-hover:text-blue-100 transition-colors font-['Space_Grotesk',sans-serif] leading-snug">
            {part.name}
          </h3>
          {part.name_en && (
            <p className="text-[11px] text-white/30 mt-0.5">{part.name_en}</p>
          )}

          <div className="flex items-center gap-3 mt-2.5 text-[10px] text-white/40 font-mono">
            <span>
              <MaterialIcon name="tag" className="text-xs inline-block mr-0.5 align-middle text-white/25" />
              {part.partNumber}
            </span>
            <span>
              <MaterialIcon name="verified" className="text-xs inline-block mr-0.5 align-middle text-blue-400/50" />
              OEM: {part.oemNumber}
            </span>
          </div>
        </div>
      </div>

      {/* ═══ ZONE B: Best Price Hero ═══ */}
      {bestPrice && <BestPriceHero store={bestPrice} savings={savings} />}

      {/* ═══ ZONE C: Secondary Stores ═══ */}
      <div className="mt-2 divide-y divide-slate-100">
        {secondaryStores.map((store) => (
          <SecondaryStoreRow key={store.storeName} store={store} />
        ))}
      </div>

      {/* ═══ ZONE D: Footer — meta info ═══ */}
      <div className="px-4 sm:px-5 py-3 border-t border-slate-100 bg-slate-50/50 mt-1">
        <div className="flex items-center justify-between text-[11px] gap-3 flex-wrap">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-slate-500">
              <MaterialIcon name="schedule" className="text-sm text-slate-400" />
              ~{part.labor_hours} óra
            </span>
            <span className="flex items-center gap-1.5 text-slate-400">
              <MaterialIcon name="payments" className="text-sm" />
              {formatHUF(part.price_range_min)}–{formatHUF(part.price_range_max)} Ft
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
                  className="w-2.5 h-2.5 rounded-full ring-2 ring-white shadow-sm"
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
