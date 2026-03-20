/**
 * RiskGauge - Circular risk gauge component for inspection results
 *
 * Displays a circular SVG gauge with color-coded risk level:
 * - Green (<0.3): Low risk
 * - Yellow (0.3-0.6): Medium risk
 * - Red (>0.6): High risk
 */

interface RiskGaugeProps {
  /** Risk score from 0 to 1 */
  score: number
  /** Risk level category */
  risk: 'high' | 'medium' | 'low'
}

function getRiskColor(score: number): { stroke: string; text: string; bg: string; label: string } {
  if (score > 0.6) {
    return {
      stroke: 'stroke-red-500',
      text: 'text-red-600',
      bg: 'bg-red-50',
      label: 'Magas kockazat',
    }
  }
  if (score >= 0.3) {
    return {
      stroke: 'stroke-yellow-500',
      text: 'text-yellow-600',
      bg: 'bg-yellow-50',
      label: 'Kozepes kockazat',
    }
  }
  return {
    stroke: 'stroke-green-500',
    text: 'text-green-600',
    bg: 'bg-green-50',
    label: 'Alacsony kockazat',
  }
}

export default function RiskGauge({ score, risk }: RiskGaugeProps) {
  const percentage = Math.round(score * 100)
  const colors = getRiskColor(score)

  // SVG circle parameters
  const size = 180
  const strokeWidth = 12
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const dashOffset = circumference - (score * circumference)

  return (
    <div className={`flex flex-col items-center p-6 rounded-2xl ${colors.bg}`}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="transform -rotate-90"
        >
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            className="stroke-gray-200"
            strokeWidth={strokeWidth}
          />
          {/* Progress arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            className={colors.stroke}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{ transition: 'stroke-dashoffset 0.8s ease-in-out' }}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-4xl font-black ${colors.text}`}>
            {percentage}%
          </span>
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider mt-1">
            Kockazat
          </span>
        </div>
      </div>
      <p className={`mt-4 text-lg font-bold ${colors.text}`}>
        {colors.label}
      </p>
      <p className="text-sm text-slate-500 mt-1 capitalize">
        {risk === 'high' && 'Vizsga megbukas valoszinu'}
        {risk === 'medium' && 'Vizsga kockazatos, javitas ajanlott'}
        {risk === 'low' && 'Vizsga atmenese valoszinu'}
      </p>
    </div>
  )
}
