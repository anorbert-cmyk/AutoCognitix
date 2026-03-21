import { CheckCircle } from 'lucide-react'

interface ConditionSliderProps {
  value: string
  onChange: (value: string) => void
}

const CONDITIONS = [
  { value: 'excellent', label: 'Kiváló', color: 'bg-green-500', description: 'Szinte új állapot' },
  { value: 'good', label: 'Jó', color: 'bg-blue-500', description: 'Normál kopás' },
  { value: 'fair', label: 'Elfogadható', color: 'bg-yellow-500', description: 'Látható kopás' },
  { value: 'poor', label: 'Gyenge', color: 'bg-red-500', description: 'Jelentős kopás' },
]

export default function ConditionSlider({ value, onChange }: ConditionSliderProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {CONDITIONS.map((condition) => {
        const isSelected = value === condition.value
        return (
          <button
            key={condition.value}
            type="button"
            onClick={() => onChange(condition.value)}
            className={`relative flex flex-col items-center gap-2 rounded-lg border-2 p-4 transition-all ${
              isSelected
                ? 'border-indigo-500 bg-indigo-50 shadow-md'
                : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-sm'
            }`}
          >
            {isSelected && (
              <CheckCircle className="absolute right-2 top-2 h-5 w-5 text-indigo-600" />
            )}
            <span className={`h-3 w-3 rounded-full ${condition.color}`} />
            <span
              className={`text-sm font-semibold ${
                isSelected ? 'text-indigo-700' : 'text-gray-800'
              }`}
            >
              {condition.label}
            </span>
            <span className="text-xs text-gray-500">{condition.description}</span>
          </button>
        )
      })}
    </div>
  )
}
