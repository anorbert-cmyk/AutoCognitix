# Accessibility Audit: AutoCognitix Landing Page
**Standard:** WCAG 2.1 AA | **Date:** 2026-03-21

## Summary
**Issues found:** 16 | **Critical:** 5 | **Major:** 6 | **Minor:** 5

---

## Findings

### Perceivable

| # | Issue | WCAG Criterion | Severity | Recommendation |
|---|-------|---------------|----------|----------------|
| 1 | Muted text (#8D96A3) on light bg (#F2F2F4) = 2.67:1 | 1.4.3 Contrast | 🔴 Critical | Darken to #6B7280 (4.56:1) |
| 2 | Muted text (#8D96A3) on gray section (#E4E4E7) = 2.36:1 | 1.4.3 Contrast | 🔴 Critical | Darken to #5F6673 (5.0:1) |
| 3 | Accent color (#D97757) on light bg (#F2F2F4) = 2.79:1 | 1.4.3 Contrast | 🔴 Critical | Darken to #B85A3A (4.62:1) for text use |
| 4 | Active lang switcher (white on #D97757) = 3.12:1 | 1.4.3 Contrast | 🟡 Major | Darken bg to #B85A3A or use dark text |
| 5 | Image grid items lack alt text / aria-label | 1.1.1 Non-text Content | 🟡 Major | Add aria-label to each .image-grid-item |
| 6 | Blog card images (background-image) not accessible | 1.1.1 Non-text Content | 🟢 Minor | Use role="img" with aria-label |

### Operable

| # | Issue | WCAG Criterion | Severity | Recommendation |
|---|-------|---------------|----------|----------------|
| 7 | No skip navigation link | 2.4.1 Bypass Blocks | 🔴 Critical | Add "Skip to content" link |
| 8 | FAQ items use div, not button — not keyboard operable | 2.1.1 Keyboard | 🔴 Critical | Change to button with aria-expanded |
| 9 | No visible focus indicators on links/buttons | 2.4.7 Focus Visible | 🟡 Major | Add :focus-visible outline styles |
| 10 | Pricing toggle buttons lack keyboard feedback | 2.1.1 Keyboard | 🟡 Major | Add role="tablist" and aria-selected |
| 11 | Newsletter input missing visible label | 2.4.6 Headings and Labels | 🟢 Minor | Add aria-label or visually hidden label |

### Understandable

| # | Issue | WCAG Criterion | Severity | Recommendation |
|---|-------|---------------|----------|----------------|
| 12 | Newsletter form has no error state for invalid email | 3.3.1 Error Identification | 🟢 Minor | Already handled via JS, but add aria-live |
| 13 | FAQ answers lack aria-hidden when closed | 3.2.1 On Focus | 🟢 Minor | Toggle aria-hidden on FAQ open/close |

### Robust

| # | Issue | WCAG Criterion | Severity | Recommendation |
|---|-------|---------------|----------|----------------|
| 14 | No `<main>` landmark on page | 4.1.2 Name, Role, Value | 🟡 Major | Wrap content in `<main>` |
| 15 | SVG icons missing role="img" or aria-hidden | 4.1.2 Name, Role, Value | 🟢 Minor | Add aria-hidden="true" to decorative SVGs |
| 16 | Nav links missing aria-current="page" for active link | 4.1.2 Name, Role, Value | 🟡 Major | Add aria-current to active nav item |

---

## Color Contrast Check

| Element | Foreground | Background | Ratio | Required | Pass? |
|---------|-----------|------------|-------|----------|-------|
| Body text | #12110E | #F2F2F4 | 16.89:1 | 4.5:1 | ✅ |
| Muted text | #8D96A3 | #F2F2F4 | 2.67:1 | 4.5:1 | ❌ |
| Muted on white | #8D96A3 | #FFFFFF | 2.99:1 | 4.5:1 | ❌ |
| Accent on light | #D97757 | #F2F2F4 | 2.79:1 | 4.5:1 | ❌ |
| Accent on dark | #D97757 | #12110E | 6.05:1 | 4.5:1 | ✅ |
| White on dark | #FFFFFF | #12110E | 18.88:1 | 4.5:1 | ✅ |
| White on btn hover | #FFFFFF | #2F3640 | 12.19:1 | 4.5:1 | ✅ |
| White on overlay | #FFFFFF | ~#474543 | 9.54:1 | 4.5:1 | ✅ |
| Muted on gray | #8D96A3 | #E4E4E7 | 2.36:1 | 4.5:1 | ❌ |
| White on accent | #FFFFFF | #D97757 | 3.12:1 | 4.5:1 | ❌ |

---

## Priority Fixes

1. **Darken muted text color** — Affects all subtitles, descriptions, excerpts across entire page. Change #8D96A3 → #6B7280
2. **Add skip-to-content link** — Blocks keyboard-only users from reaching content
3. **Make FAQ keyboard-accessible** — Change div.faq-question to button with aria-expanded
4. **Add focus-visible outlines** — Keyboard users cannot see where focus is
5. **Add `<main>` landmark** — Screen readers cannot identify main content area
6. **Fix accent text color** — Hero subtitle accent fails contrast; use darker variant for text
