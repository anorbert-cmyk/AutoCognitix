## Frontend Logic Audit
**Auditor:** Frontend-Logic Specialist
**Dátum:** 2026-03-29
**Branch:** `claude/ralph-loop-global-memory-lFEhR`

### Érintett fájlok

| Fájl | Státusz |
|------|---------|
| `frontend/src/hooks/useStreamingDiagnosis.ts` | Auditálva |
| `frontend/src/contexts/AuthContext.tsx` | Auditálva |
| `frontend/src/services/diagnosisService.ts` | Auditálva |
| `frontend/src/services/garageService.ts` | Auditálva |
| `frontend/src/pages/DiagnosisPage.tsx` | Auditálva |
| `frontend/src/pages/ResultPage.tsx` | Auditálva |
| `frontend/src/components/features/diagnosis/AnalysisProgress.tsx` | Auditálva |
| `frontend/src/services/authService.ts` | Auditálva |
| `frontend/src/services/api.ts` | Auditálva |

---

### Talált hibák — Osztályozva

---

#### [HIGH-1] `useStreamingDiagnosis` — `streamDiagnosisGenerator` generator nem hívja meg az `onStart`/`onProgress` callback-eket

**Fájl:** `frontend/src/hooks/useStreamingDiagnosis.ts` (sor 175–195)

**Leírás:**
A `streamDiagnosisGenerator` async generator a `streamDiagnosis()` service-t hívja meg, de csak `onAnalysis`, `onComplete` és `onError` callback-eket ad át. Az `onStart`, `onContext`, `onCause`, `onRepair`, `onWarning` és `onProgress` callback-ek hiányoznak. Ha a generator-t fogyasztó kód ezekre a közbülső eseményekre támaszkodik (pl. haladásjelző UI), soha nem fogja megkapni azokat.

**Súlyosság:** HIGH — A generator API hiányos; progress állapot sosem frissül a generator variansnál.

---

#### [HIGH-2] `AuthContext` — Token lejárat után nincs automatikus logout/redirect a React context szintjén

**Fájl:** `frontend/src/contexts/AuthContext.tsx` (sor 66–85), `frontend/src/services/api.ts` (sor 191–196)

**Leírás:**
Az `initAuth` csak egyszer fut mountkor. Ha az access token lejár a session közben, az axios interceptor (`api.ts` sor 161–198) megpróbálja a refresh token-nel megújítani. Ha a refresh is meghiúsul, az interceptor `window.location.href = '/login'`-ra redirect-el — ez azonban **bypass-olja a React router-t** és a teljes alkalmazás állapotot elveszíti. Nem hív `clearTokens()`-t az AuthContext-en belül, így az `isAuthenticated` / `user` állapot inkonzisztens marad a redirect előtt.

Ezen felül az `isAuthenticated` értéke `!!user` a context-ben (sor 178), de az `authService.isAuthenticated()` egy különálló in-memory flag (`authenticated`). Ha az axios interceptor az `authenticated` flaget `false`-ra állítja (`setCsrfToken(null)` → `clearTokens()`), a context `user` state még mindig nem-null értékű marad egészen a page reload-ig.

**Súlyosság:** HIGH — Az auth állapot felhasználói felületen kívüli meghajtású invalidáció esetén inkonzisztens.

---

#### [HIGH-3] `diagnosisService.ts` — SSE parser figyelmen kívül hagyja az SSE protokoll szintű `event:` sort

**Fájl:** `frontend/src/services/diagnosisService.ts` (sor 279–309)

**Leírás:**
A `parseSSEEvents()` függvény az SSE eseményeket a `data:` sor alapján parsing-olja, és a JSON-ból olvassa az `event_type`-ot. Ez azt feltételezi, hogy a backend az event type-ot kizárólag a JSON payloadban (`event.event_type`) küldi, nem az SSE `event:` header sorban. A parser a `event:` sort a for-ciklusban csendesen elveti (nincs `if (line.startsWith('event: '))` ág). Ha a backend a standard SSE formátumot használja (`event: analysis\ndata: {...}`) de nem teszi bele az `event_type`-ot a JSON-ba is, akkor a dispatch silent-fail-t szenved.

**Súlyosság:** HIGH — Nem-teljes SSE protokoll implementáció; backend SSE `event:` header sor elvész.

---

#### [MEDIUM-1] `DiagnosisPage` — Frontend validáció nem fut a submit előtt

**Fájl:** `frontend/src/pages/DiagnosisPage.tsx` (sor 121–140)

**Leírás:**
Az űrlap submit (`handleSubmit`) csak a `dtcCode.trim()` üresség-ellenőrzést végzi el, majd azonnal `setCurrentStep('analysis')` hívással átvált az `AnalysisProgress` komponensre. A `validateDiagnosisRequest()` (`diagnosisService.ts` sor 88–126) csak a service layer-ben fut le, miután az `AnalysisProgress` már megjelent és a streaming elindult. Ha a felhasználó szóközzel elválasztott kódokat gépel be (`P0300 P0301`), az teljes egészében egyetlen érvénytelen DTC kódként kerül elküldésre, a streaming indul, majd az error callback hívódik — ahelyett, hogy az input mezőnél azonnal hibát kapna.

**Súlyosság:** MEDIUM — Késői validáció rontja a UX-et; az érvénytelen adat a streaming start-ig nem derül ki.

---

#### [MEDIUM-2] `DiagnosisPage` — Submit gomb disabled állapota streaming módban soha nem aktiválódik

**Fájl:** `frontend/src/pages/DiagnosisPage.tsx` (sor 416–425)

**Leírás:**
A submit gomb `disabled={analyzeDiagnosis.isPending}` állapotban van. Az `analyzeDiagnosis.isPending` azonban soha nem lesz `true` a normál streamelt útvonalnál, mert `handleSubmit` közvetlenül `setCurrentStep('analysis')`-t hív, nem indít `analyzeDiagnosis.mutateAsync()`-t. Ennek következménye, hogy a gomb sosem kerül disabled állapotba streaming módban — ha az `AnalysisProgress` komponens visszanavigál (pl. hiba esetén), a felhasználó újra beküldheti az űrlapot a gomb vizuális visszajelzése nélkül.

**Súlyosság:** MEDIUM — Dupla submit lehetséges; hiányzó loading feedback streaming módban.

---

#### [MEDIUM-3] `AnalysisProgress` — `isMockMode` useMemo dependency array hiányos

**Fájl:** `frontend/src/components/features/diagnosis/AnalysisProgress.tsx` (sor 412–415)

**Leírás:**
```tsx
const isMockMode = useMemo(
  () => !streamingEnabled || !getStreamDiagnosisFn() || !diagnosisRequest,
  [streamingEnabled, diagnosisRequest]
);
```
A `getStreamDiagnosisFn()` egy runtime modul-lookup, amely potenciálisan eltérő értéket adhat vissza újrarenderelések között (pl. lazy import esetén). Mivel a `useMemo` dependency array-ben nem szerepel a függvény referenciája, a memo cache-elt értéket ad vissza és nem reagál a modul betöltési állapot változásaira. Ez azt okozhatja, hogy mock mode-ban ragad az oldal, miközben a streaming már elérhető.

**Súlyosság:** MEDIUM — Lazy-loaded modulok esetén az oldal mock mode-ban ragadhat.

---

#### [MEDIUM-4] `AnalysisProgress` — `handleRetry` manuálisan `mountedRef.current = true`-ra állít, de ez megkerüli a cleanup logikát

**Fájl:** `frontend/src/components/features/diagnosis/AnalysisProgress.tsx` (sor 493–513)

**Leírás:**
A `handleRetry` függvény (sor 509) `mountedRef.current = true`-ra állítja a ref-et. Ha viszont a useEffect cleanup már lefutott (pl. React Strict Mode kétszeres mount/unmount), a `mountedRef` visszaállítása megkerüli azt a védelmet, amelyet a ref nyújt az unmountolt komponensen végrehajtott setState hívások ellen. Strict Mode-ban ez "state update on unmounted component" warningot okozhat.

**Súlyosság:** MEDIUM — React Strict Mode / fejlesztési környezetben bugos viselkedés.

---

#### [MEDIUM-5] `ResultPage` — `handleSavePDF` és `handlePrintWorksheet` azonos `window.print()` hívást tartalmaz

**Fájl:** `frontend/src/pages/ResultPage.tsx` (sor 65–74)

**Leírás:**
Mindkét handler (`handleSavePDF` és `handlePrintWorksheet`) csak `window.print()`-et hív, más logika nélkül. A két gomb eltérő label-lel jelenik meg ("Nyomtatás / PDF" vs "Munkalap nyomtatása"), de viselkedésük azonos. Ha a felhasználó a "Munkalap nyomtatása" gombra kattint azzal a várakozással, hogy különálló munkalapformátumot kap, megtévesztő élményt tapasztal.

**Súlyosság:** MEDIUM — Félrevezető UX; a kettős gomb megtéveszti a felhasználót.

---

#### [LOW-1] `useStreamingDiagnosis` — `stopStreaming` nem nullázza a `progress` értékét

**Fájl:** `frontend/src/hooks/useStreamingDiagnosis.ts` (sor 108–112)

**Leírás:**
Ha a stream befejezés előtt `stopStreaming()`-et hívunk, az `isStreaming` `false`-ra kerül, de `progress` értéke megőrzi az utolsó értéket (pl. 0.82). Ha az UI ezt az értéket progress bar megjelenítéséhez használja, a következő `startStreaming()` hívásnál csak akkor kerül nullázásra, amikor az `INITIAL_STATE` spread lefut (sor 64) — ami elvárt viselkedés. Azonban ha a komponens a `progress`-t a `stopStreaming()` után is rendereli (pl. visszaküldi a parent-nek), vizuálisan félrevezető marad.

**Súlyosság:** LOW — Vizuális inkonzisztencia szélsőséges esetben.

---

#### [LOW-2] `AuthContext` — `refreshUser` csendes hibát produkál lejárt token esetén

**Fájl:** `frontend/src/contexts/AuthContext.tsx` (sor 165–175)

**Leírás:**
A `refreshUser` callback az `authService.isAuthenticated()` értékét ellenőrzi, ami egy in-memory flag, nem a valódi token érvényességét. Ha a token lejárt, de a flag még `true`, `refreshUser()` API hívást indít, ami 401-es hibát kaphat. A catch ágban `clearTokens()` és `setUser(null)` hívódik, de az `error` state nem kerül set-elésre. A hívó kód tehát nem tudja megkülönböztetni a sikeres és a silent-fail `refreshUser()` hívást.

**Súlyosság:** LOW — Csendes hiba, nehezen debuggolható.

---

#### [LOW-3] `diagnosisService.ts` — `formatCostRange` hamis negatívot ad `0` értéknél

**Fájl:** `frontend/src/services/diagnosisService.ts` (sor 609–629)

**Leírás:**
```ts
if (!min && !max) { return 'Nincs becslés' }
```
Ha `min = 0` (érvényes ár), az `!min` feltétel `true`, és ha `max` is `0` vagy undefined, a függvény `'Nincs becslés'`-t ad vissza. Ez JavaScript falsy értékkezelési antipattern — a `0` mint érvényes nullás ár nem kezelt. A helyes ellenőrzés: `if (min == null && max == null)`.

**Súlyosság:** LOW — Nullás/ingyenes árak esetén helytelen megjelenítés.

---

#### [LOW-4] `DiagnosisPage` — Szükségtelen `diagnosisId` state a navigáció előtt

**Fájl:** `frontend/src/pages/DiagnosisPage.tsx` (sor 82, 178–184)

**Leírás:**
A `diagnosisId` state (sor 82) `setDiagnosisId(resultId)` hívással frissül a `handleAnalysisComplete`-ben (sor 182), de ezt közvetlenül a `navigate()` hívás követi (sor 183). Az állapotfrissítés felesleges, mivel semmilyen render nem függ ettől az értéktől a navigáció előtt — az `AnalysisProgress` az `undefined`-ot kapja egészen az ID megérkeztéig. Ez szükségtelen setState → re-render ciklust idéz elő.

**Súlyosság:** LOW — Felesleges re-render, nincs funkcionális hatás.

---

### Összefoglalás

| Szint | Darab |
|-------|-------|
| HIGH | 3 |
| MEDIUM | 5 |
| LOW | 4 |
| **Összesen** | **12** |

### Kritikus figyelmet igénylő területek (prioritás sorrendben)

1. **Auth állapot szinkronizáció** (HIGH-2): A token lejárat kezelése két különböző rendszeren van elosztva (axios interceptor + AuthContext) és ezek nincsenek szinkronizálva. A `window.location.href` redirect az AuthContext állapotát inkonzisztensen hagyja.

2. **SSE parser robusztussága** (HIGH-3): A `parseSSEEvents` csak a JSON payload `event_type` mezőjét nézi, az SSE protokoll-szintű `event:` sort figyelmen kívül hagyja — nem-teljes SSE implementáció.

3. **Generator API hiányossága** (HIGH-1): A `streamDiagnosisGenerator` csak 3 callback-et ad tovább a lehetséges 8-ból, ezért a haladás és közbülső lépések sosem jelennek meg generátor alapú fogyasztóknál.
