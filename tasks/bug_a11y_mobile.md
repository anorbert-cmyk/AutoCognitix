# A11y & Mobile Audit

## A) Aria labels

- **severity: HIGH** — 10/21 pages (48%) have **zero** aria-label / aria-describedby / role / htmlFor. Total 65 hits across 21 pages → ~3/page average, heavily skewed (GaragePage 13, VehicleDetailPage 20 carry the load; many pages 0).
- **CRITICAL gap pages (0 coverage):**
  - `frontend/src/pages/ResultPage.tsx:0` — main output of the product, no aria on any button/icon. e.g. `:95` `<button className="hidden md:flex...">` icon-only button with no label.
  - `frontend/src/pages/DemoResultPage.tsx:0` — public demo, no labels.
  - `frontend/src/pages/HomePage.tsx:0`, `ChatPage.tsx:0`, `NewDiagnosisPage.tsx:0`, `BlogPage.tsx:0`, `ChangelogPage.tsx:0`, `DTCDetailPage.tsx:0`, `NotFoundPage.tsx:0`, `ServiceComparisonPage.tsx:0`.
- **findings (good examples to replicate):**
  - `frontend/src/pages/GaragePage.tsx:79` — `aria-label={"${displayName} törlése"}` icon-only delete button correctly labeled.
  - `frontend/src/pages/GaragePage.tsx:280-293` — modal uses `role="dialog"` + `aria-label="Új jármű hozzáadása"` + close button `aria-label="Bezárás"`. Reference pattern.
  - `frontend/src/pages/LoginPage.tsx:84,107` — only `htmlFor` on `email`/`password`. Password-show toggle button at `:123` has no `aria-label` → screen reader announces nothing.
  - `frontend/src/pages/RegisterPage.tsx` — 4 `htmlFor` for inputs, but no aria-describedby on validation hints (PasswordStrengthMeter not linked).
- **Snippet — LoginPage password toggle (missing aria-label):**
  ```tsx
  // LoginPage.tsx:123
  <button type="button" onClick={() => setShowPassword(...)} className="...">
    <MaterialIcon name={showPassword ? 'visibility_off' : 'visibility'} />
  </button>  // ← needs aria-label="Jelszó megjelenítése"
  ```

## B) Mobile breakpoint

- **severity: MEDIUM** — ResultPage uses `sm:` 5x, `md:` 10x, `lg:` 6x (21 total / ~735 LOC). Skews to `md:` (tablet), thin `sm:` coverage → 360-639px phones get default styles only.
- **risk points:**
  1. `frontend/src/pages/ResultPage.tsx:95` — `hidden md:flex` action button is invisible on mobile (<768px). Mobile users lose functionality silently.
  2. `frontend/src/pages/ServiceComparisonPage.tsx:158` — `w-full lg:w-[400px]` fixed 400px sidebar on `lg:`. OK, but adjacent map column has no min-w → squishes on tablets 1024-1100px.
  3. `frontend/src/pages/ResultPage.tsx:235` and `DemoResultPage.tsx:244` — `w-80 h-80` (320px) decorative blur circles with no responsive scale; overflow on 360px viewports.
  4. `frontend/src/pages/ChangelogPage.tsx:105` — fixed `w-[32px] h-[32px] md:w-[40px]` timeline dots: fine, but the page has 0 sm: breakpoint usage at all.
  5. `frontend/src/components/features/diagnosis/DiagnosisForm.tsx:269` — `min-w-[280px]` button forces horizontal scroll on 320-360px iPhone SE.
  6. `frontend/src/components/features/diagnosis/PartStoreCard.tsx:140` — `min-w-[80px] text-right` price column inside flex row with long brand names → wraps badly on mobile (no `flex-wrap` parent observed).

## C) Contrast/color-only

- **severity: MEDIUM** — Recall badge uses **icon + color + text** (good), but lacks ARIA role for assistive tech. No standalone `RecallBadge.tsx` component — inlined in `ResultPage.tsx:418-470` and `VehicleDetailPage.tsx` (tab content).
- **Recall snippet — `frontend/src/pages/ResultPage.tsx:419-432`:**
  ```tsx
  <div className="bg-red-50 border-2 border-red-200 rounded-2xl p-6">
    <div className="flex items-center gap-3 mb-4">
      <MaterialIcon name="warning" className="text-xl text-red-600" />  // ✓ icon present
      <h3 className="text-lg font-bold text-red-900">NHTSA Visszahívások</h3>
      <span className="...bg-red-600 text-white">{n} aktív</span>
    </div>
  ```
  - Color-only risk: **none** (icon + "Visszahívások" text + count badge). Color-blind users still get signal.
  - Missing: no `role="alert"` or `aria-live="polite"` — recall is critical safety info but not announced.
- **Color-only findings elsewhere:**
  - `frontend/src/components/features/calculator/ValueComparison.tsx:42` — `ratio <= 40 ? text-green-600 : ratio <= 70 ? text-yellow-600 : text-red-600` on a number. **No icon, no text label.** Color-blind users cannot distinguish good/bad value.
  - `frontend/src/components/ui/PasswordStrengthMeter.tsx:89` — `req.passed ? 'text-green-600' : 'text-gray-500'`. Need check/x icon for color-blind users.
  - `frontend/src/components/VehicleSelector.tsx:407,413` — error/success message uses color only on small `<p>`; needs icon prefix (uses `flex items-center gap-1` but icon child not always rendered).
  - `frontend/src/components/features/chat/ChatInput.tsx:118,151` — recording mic state and char-limit warning rely on `text-red-500` alone. The `animate-pulse` is motion-only (failed by `prefers-reduced-motion`).

## Olvasott fájlok

- `frontend/src/pages/LoginPage.tsx` (lines 73-160)
- `frontend/src/pages/GaragePage.tsx` (lines 76-430)
- `frontend/src/pages/ResultPage.tsx` (lines 95, 235, 355, 415-470)
- `frontend/src/pages/DiagnosisPage.tsx` (lines 261-465)
- `frontend/src/pages/RegisterPage.tsx` (lines 113-228)
- `frontend/src/pages/VehicleDetailPage.tsx` (header — recall tab)
- `frontend/src/components/features/diagnosis/PartStoreCard.tsx` (line 140)
- `frontend/src/components/features/calculator/ValueComparison.tsx` (line 42)
