/**
 * Shared Tailwind class tokens.
 *
 * These are *style* constants, not components — which is why they live in
 * `src/lib/` next to `cn()` rather than in `src/components/ui/`, whose barrel
 * deliberately exports React components only.
 *
 * Nothing here may encode layout that depends on a call site's content (icons,
 * count badges, flex gaps). Those stay at the call site; only the decisions that
 * are genuinely one decision live here.
 */

/**
 * Keyboard focus ring for interactive elements on LIGHT surfaces.
 *
 * This is the same utility chain `index.css` already `@apply`s into `.btn`,
 * `.input` and `.textarea` — this constant is for elements that cannot use those
 * classes (bespoke links, tab buttons, chip buttons). It was previously copied
 * verbatim into five modules, which is exactly how a focus style drifts.
 */
export const focusRing =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'

/**
 * Focus ring counterpart for elements sitting on a DARK surface (the HomePage
 * hero gradient). Deliberately NOT merged into `focusRing`: `--ring` is a dark
 * accent and would be nearly invisible against the gradient, so this variant
 * switches to a white ring with a matching dark offset colour. Two surfaces,
 * two decisions — kept adjacent so the pairing stays discoverable.
 */
export const focusRingDark =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-offset-2 focus-visible:ring-offset-primary-800'

/**
 * The "selected pill on a slate-100 track" segmented control item.
 *
 * Scope note — this shares the STYLE, not the markup. The three call sites are
 * not the same component: the page tab bars (`SettingsPage`, `VehicleDetailPage`)
 * are navigation over a static tab list, while `CommonIssuesPanel`'s year toggle
 * is a two-option data filter with its own `role="group"` wrapper and a
 * prop-derived value. Their ARIA contracts and item payloads differ; only the
 * class list was ever duplicated, so only the class list is shared.
 *
 * `VehicleDetailPage` records what happens without this: two of its five tabs
 * had drifted to `px-4 py-2 font-medium` and the tab bar rendered visually
 * uneven. `VehicleDetailPage.test.tsx > VehicleDetailPage — fülsáv` pins the
 * `md` output (`h-9` / `px-5` / `font-semibold`).
 *
 * The emitted string is byte-identical to the three literals it replaces, so
 * adopting it is a provably pixel-neutral change.
 */
const SEGMENTED_SIZES = {
  /** Page-level tab bars. */
  md: 'h-9 px-5 rounded-lg text-sm font-semibold',
  /** Compact in-panel filters. */
  sm: 'h-8 px-3 rounded-md text-xs font-semibold',
} as const

export type SegmentedSize = keyof typeof SEGMENTED_SIZES

export function segmentedItemClass(isActive: boolean, size: SegmentedSize = 'md'): string {
  return [
    SEGMENTED_SIZES[size],
    'transition-colors',
    focusRing,
    isActive ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-800',
  ].join(' ')
}
