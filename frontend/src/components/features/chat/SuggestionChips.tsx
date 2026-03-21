/**
 * SuggestionChips Component
 * Horizontal scrollable row of quick-reply suggestion chips.
 */

interface SuggestionChipsProps {
  suggestions: string[]
  onSelect: (suggestion: string) => void
}

export function SuggestionChips({ suggestions, onSelect }: SuggestionChipsProps) {
  if (suggestions.length === 0) return null

  return (
    <div className="flex gap-2 overflow-x-auto px-4 py-2 scrollbar-hide">
      {suggestions.map((suggestion) => (
        <button
          key={suggestion}
          type="button"
          onClick={() => onSelect(suggestion)}
          className="flex-shrink-0 rounded-full border border-primary-300 bg-white px-4 py-2 text-sm text-primary-700 font-medium hover:bg-primary-50 hover:border-primary-400 active:bg-primary-100 transition-colors whitespace-nowrap"
        >
          {suggestion}
        </button>
      ))}
    </div>
  )
}

export default SuggestionChips
