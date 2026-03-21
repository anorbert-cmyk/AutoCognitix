/**
 * RatingStars Component
 *
 * Displays a star rating (0-5) with filled, half, and empty stars.
 * Optionally shows numeric rating and review count.
 */

import { Star } from 'lucide-react'

interface RatingStarsProps {
  rating: number
  reviewCount?: number
  size?: 'sm' | 'md'
}

export function RatingStars({ rating, reviewCount, size = 'sm' }: RatingStarsProps) {
  const clampedRating = Math.max(0, Math.min(5, rating))
  const fullStars = Math.floor(clampedRating)
  const hasHalf = clampedRating - fullStars >= 0.25 && clampedRating - fullStars < 0.75
  const emptyStart = hasHalf ? fullStars + 1 : fullStars

  const iconSize = size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4'

  const stars: React.ReactElement[] = []

  // Full stars
  for (let i = 0; i < fullStars; i++) {
    stars.push(
      <Star
        key={`full-${i}`}
        className={`${iconSize} text-amber-400 fill-amber-400`}
      />
    )
  }

  // Half star
  if (hasHalf) {
    stars.push(
      <span key="half" className={`relative inline-flex ${iconSize}`}>
        <Star className={`${iconSize} text-slate-200 fill-slate-200 absolute`} />
        <span className="overflow-hidden w-1/2 absolute">
          <Star className={`${iconSize} text-amber-400 fill-amber-400`} />
        </span>
      </span>
    )
  }

  // Empty stars
  for (let i = emptyStart; i < 5; i++) {
    stars.push(
      <Star
        key={`empty-${i}`}
        className={`${iconSize} text-slate-200 fill-slate-200`}
      />
    )
  }

  return (
    <div className="flex items-center gap-1">
      <div className="flex items-center gap-0.5">{stars}</div>
      <span className="text-xs font-semibold text-slate-700 tabular-nums">
        {clampedRating.toFixed(1)}
      </span>
      {reviewCount !== undefined && (
        <span className="text-xs text-slate-400">
          ({reviewCount})
        </span>
      )}
    </div>
  )
}

export default RatingStars
