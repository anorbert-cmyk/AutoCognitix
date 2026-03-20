/**
 * ShopCard Component
 *
 * Displays a single service shop listing with rating, price level,
 * specializations, services, and contact info.
 */

import { MapPin, Phone, Shield, Wrench } from 'lucide-react'
import type { ServiceShop } from '../../../services/serviceShopService'
import { formatPriceLevel } from '../../../services/serviceShopService'
import { RatingStars } from './RatingStars'

interface ShopCardProps {
  shop: ServiceShop
  onClick: () => void
}

export function ShopCard({ shop, onClick }: ShopCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full text-left bg-white rounded-xl border border-slate-200 p-4 transition-all duration-200 hover:shadow-lg hover:border-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500"
    >
      {/* Header: Name + Rating */}
      <div className="flex items-start justify-between gap-3 mb-2">
        <h3 className="text-sm font-bold text-slate-900 leading-tight">
          {shop.name}
        </h3>
        <div className="flex-shrink-0">
          <RatingStars rating={shop.rating} reviewCount={shop.review_count} />
        </div>
      </div>

      {/* Address */}
      <div className="flex items-start gap-1.5 mb-2.5">
        <MapPin className="w-3.5 h-3.5 text-slate-400 mt-0.5 flex-shrink-0" />
        <span className="text-xs text-slate-500 leading-relaxed">
          {shop.address}, {shop.city}
        </span>
      </div>

      {/* Price level + Inspection badge */}
      <div className="flex items-center gap-3 mb-3">
        <span
          className="text-xs font-bold text-emerald-700 tracking-wide"
          title={`\u00C1rszint: ${shop.price_level}/3`}
        >
          {formatPriceLevel(shop.price_level)}
        </span>
        {shop.has_inspection && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-blue-50 text-blue-700 border border-blue-100">
            <Shield className="w-3 h-3" />
            M\u0171szaki vizsga
          </span>
        )}
      </div>

      {/* Specializations */}
      {shop.specializations.length > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap mb-2.5">
          {shop.specializations.map((spec) => (
            <span
              key={spec}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-semibold bg-amber-50 text-amber-800 border border-amber-100"
            >
              <Wrench className="w-2.5 h-2.5" />
              {spec}
            </span>
          ))}
        </div>
      )}

      {/* Services tags */}
      {shop.services.length > 0 && (
        <div className="flex items-center gap-1 flex-wrap mb-3">
          {shop.services.slice(0, 5).map((service) => (
            <span
              key={service}
              className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600"
            >
              {service}
            </span>
          ))}
          {shop.services.length > 5 && (
            <span className="text-[10px] text-slate-400 font-medium">
              +{shop.services.length - 5}
            </span>
          )}
        </div>
      )}

      {/* Phone */}
      {shop.phone && (
        <div className="flex items-center gap-1.5 pt-2.5 border-t border-slate-100">
          <Phone className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-xs text-slate-600 font-medium">{shop.phone}</span>
        </div>
      )}
    </button>
  )
}

export default ShopCard
