import { describe, it, expect } from 'vitest'
import { focusRing, focusRingDark, segmentedItemClass } from '../styles'

/**
 * These tests exist to make the consolidation claim FALSIFIABLE.
 *
 * `focusRing` and the segmented-control class list were previously copied
 * verbatim into five and three modules respectively. The literals below are the
 * exact strings those modules used to build before the consolidation, so if the
 * shared helper ever emits something different the diff is no longer a
 * pixel-neutral refactor and these fail.
 */

// Byte-for-byte, as it appeared in HomePage / SettingsPage / VehicleDetailPage /
// CommonIssuesPanel / EmptyState before consolidation.
const ORIGINAL_FOCUS_RING =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'

describe('focusRing', () => {
  it('matches the focus ring the five call sites used to duplicate', () => {
    expect(focusRing).toBe(ORIGINAL_FOCUS_RING)
  })

  it('is the same utility chain index.css @applies to .btn/.input/.textarea', () => {
    // Keeps a bespoke focusable element visually identical to a real .btn.
    for (const utility of [
      'focus-visible:outline-none',
      'focus-visible:ring-2',
      'focus-visible:ring-ring',
      'focus-visible:ring-offset-2',
    ]) {
      expect(focusRing.split(' ')).toContain(utility)
    }
  })

  it('keeps the dark-surface variant distinct from the light one', () => {
    // `--ring` is a dark accent and would be nearly invisible on the HomePage
    // hero gradient, so the dark variant switches to a white ring. Merging the
    // two would silently break focus visibility on the hero.
    expect(focusRingDark).not.toBe(focusRing)
    expect(focusRingDark).toContain('focus-visible:ring-white')
    expect(focusRingDark).toContain('focus-visible:ring-offset-primary-800')
  })
})

describe('segmentedItemClass', () => {
  // Exactly what SettingsPage's template literal produced.
  const originalMd = (isActive: boolean) =>
    `h-9 px-5 rounded-lg text-sm font-semibold transition-colors ${ORIGINAL_FOCUS_RING} ${
      isActive ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-800'
    }`

  // Exactly what CommonIssuesPanel's year-toggle template literal produced.
  const originalSm = (isActive: boolean) =>
    `h-8 px-3 rounded-md text-xs font-semibold transition-colors ${ORIGINAL_FOCUS_RING} ${
      isActive ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-800'
    }`

  it('reproduces the md tab-bar string byte for byte', () => {
    expect(segmentedItemClass(true)).toBe(originalMd(true))
    expect(segmentedItemClass(false)).toBe(originalMd(false))
  })

  it('reproduces the sm in-panel filter string byte for byte', () => {
    expect(segmentedItemClass(true, 'sm')).toBe(originalSm(true))
    expect(segmentedItemClass(false, 'sm')).toBe(originalSm(false))
  })

  it('defaults to the md size', () => {
    expect(segmentedItemClass(true)).toBe(segmentedItemClass(true, 'md'))
  })

  it('keeps the size tokens VehicleDetailPage.test.tsx pins', () => {
    // That page's fülsáv test asserts h-9 / px-5 / font-semibold on all five
    // tabs, after two of them had drifted to px-4 py-2 font-medium.
    const cls = segmentedItemClass(false)
    expect(cls).toContain('h-9')
    expect(cls).toContain('px-5')
    expect(cls).toContain('font-semibold')
  })

  it('gives the selected item the raised-pill treatment and the rest a hover', () => {
    expect(segmentedItemClass(true)).toContain('bg-white text-slate-900 shadow-sm')
    expect(segmentedItemClass(true)).not.toContain('hover:text-slate-800')

    expect(segmentedItemClass(false)).toContain('text-slate-600 hover:text-slate-800')
    expect(segmentedItemClass(false)).not.toContain('shadow-sm')
  })

  it('carries the shared focus ring at every size', () => {
    expect(segmentedItemClass(true, 'md')).toContain(focusRing)
    expect(segmentedItemClass(false, 'sm')).toContain(focusRing)
  })
})
