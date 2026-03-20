/**
 * Password strength meter component with visual indicator and requirement checklist.
 * Pure client-side validation - no API calls.
 */

import { useMemo } from 'react'
import { Check, X } from 'lucide-react'

interface PasswordStrengthMeterProps {
  password: string
}

interface Requirement {
  id: string
  label: string
  test: (password: string) => boolean
}

const REQUIREMENTS: Requirement[] = [
  { id: 'length', label: 'Legalább 8 karakter', test: (p) => p.length >= 8 },
  { id: 'uppercase', label: 'Nagybetű (A-Z)', test: (p) => /[A-Z]/.test(p) },
  { id: 'lowercase', label: 'Kisbetű (a-z)', test: (p) => /[a-z]/.test(p) },
  { id: 'digit', label: 'Szám (0-9)', test: (p) => /\d/.test(p) },
  {
    id: 'special',
    label: 'Speciális karakter (!@#$%...)',
    // eslint-disable-next-line no-useless-escape
    test: (p) => /[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]/.test(p),
  },
]

const SCORE_CONFIG: Record<number, { label: string; color: string; bgColor: string }> = {
  0: { label: '', color: 'bg-gray-200', bgColor: 'bg-gray-200' },
  1: { label: 'Gyenge', color: 'bg-red-500', bgColor: 'bg-red-100' },
  2: { label: 'Gyenge', color: 'bg-orange-500', bgColor: 'bg-orange-100' },
  3: { label: 'Közepes', color: 'bg-yellow-500', bgColor: 'bg-yellow-100' },
  4: { label: 'Erős', color: 'bg-green-500', bgColor: 'bg-green-100' },
  5: { label: 'Nagyon erős', color: 'bg-green-600', bgColor: 'bg-green-100' },
}

export default function PasswordStrengthMeter({ password }: PasswordStrengthMeterProps) {
  const { score, results } = useMemo(() => {
    const results = REQUIREMENTS.map((req) => ({
      ...req,
      passed: req.test(password),
    }))
    const score = results.filter((r) => r.passed).length
    return { score, results }
  }, [password])

  if (!password) {
    return null
  }

  const config = SCORE_CONFIG[score]
  const widthPercent = (score / 5) * 100

  return (
    <div className="mt-2 space-y-2">
      {/* Progress bar */}
      <div className="flex items-center gap-2">
        <div className="flex-1 h-2 rounded-full bg-gray-200 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-300 ${config.color}`}
            style={{ width: `${widthPercent}%` }}
          />
        </div>
        {config.label && (
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${config.bgColor} ${
              score <= 2
                ? 'text-red-700'
                : score === 3
                  ? 'text-yellow-700'
                  : 'text-green-700'
            }`}
          >
            {config.label}
          </span>
        )}
      </div>

      {/* Requirements checklist */}
      <ul className="space-y-1">
        {results.map((req) => (
          <li
            key={req.id}
            className={`flex items-center text-xs ${
              req.passed ? 'text-green-600' : 'text-gray-500'
            }`}
          >
            {req.passed ? (
              <Check className="h-3 w-3 mr-1.5 flex-shrink-0" />
            ) : (
              <X className="h-3 w-3 mr-1.5 flex-shrink-0" />
            )}
            {req.label}
          </li>
        ))}
      </ul>
    </div>
  )
}
