# UX Error States Audit

Sprint 10 utáni gyors audit: 8 fájl olvasva, 3 tengely (empty / loading / error+retry). User-facing magyar copy ellenőrzése.

## A) Empty state hiány

- **severity: LOW** — három auditált oldal mind kezeli az empty state-et, de a minőség inkonzisztens.

### Találatok

- `frontend/src/pages/GaragePage.tsx:244-259` — **JÓ**: 0 jármű esetén dedikált empty card emojival ("🚗"), magyar szöveggel ("Még nincs jármű hozzáadva") + CTA gombbal ("Jármű hozzáadása"). Példa-snippet:
  ```tsx
  {!isLoading && !error && vehicles.length === 0 && (
    <h2>Még nincs jármű hozzáadva</h2>
    <button onClick={handleOpenModal}>Jármű hozzáadása</button>
  )}
  ```
- `frontend/src/pages/HistoryPage.tsx:371-383` — **JÓ**: empty állapot CTA-val (`<Link to="/diagnosis">Új diagnózis indítása</Link>`), de **44-105: mock data fallback** — ha API üres, a felhasználó valójában 5 hamis Toyota/Ford/Honda rekordot lát, NEM az empty state-et. Ez UX-csapda, súlyos demo-bug:
  ```tsx
  const historyData = apiHistoryData?.items ? apiHistoryData.items.map(...) : mockHistoryData;
  ```
  → **severity: HIGH** (külön finding) — éles üzem: a user soha nem éri el az empty state-et, ha az API empty `items: []` helyett `undefined`-ot ad vissza.
- `frontend/src/pages/ResultPage.tsx:284-289` — **JÓ**: 0 javítási lépés esetén "Nincs elérhető javítási javaslat ehhez a diagnosztikához." üzenet ikonnal. Sor 405-411: 0 alkatrész esetén kék info-kártya "Alkatrész árinformáció nem elérhető".
- `frontend/src/pages/VehicleDetailPage.tsx:508-511` — **GYENGE**: recalls empty `<p>Nincs ismert visszahívás ehhez a járműhöz</p>` — csak szöveg, **nincs ikon, nincs CTA**. Egyébként inkonzisztens a GaragePage gazdag empty cardjával.

## B) Loading inkonzisztencia

- **severity: MEDIUM** — 4 különböző loading mintát használ a kódbázis ugyanarra a feladatra.

### Találatok (5 minta a 8 fájlból)

1. `GaragePage.tsx:229-233` — `<Loader2 className="h-8 w-8 animate-spin">` centered, py-24, NINCS szöveg ("Betöltés..." felirat hiányzik a vizuális mellől, csak aria-label).
2. `ResultPage.tsx:544-552` — fullscreen `Loader2 h-12 w-12` + felirat ("Diagnózis betöltése..."). **Más méret, van szöveg** → inkonzisztens a GaragePage-dzsel.
3. `HistoryPage.tsx:347-352` — táblázat sorba `colSpan={7}` "Betöltés..." **csak szöveg**, NINCS spinner. Stats-okhoz külön `statsLoading ? '...' : value` (sor 488, 507, 523) — három pont! Ez harmadik minta.
4. `VehicleDetailPage.tsx:108-114` — `Loader2 h-8 w-8`, min-h-[50vh], NINCS szöveg.
5. `ServiceComparisonPage.tsx:161-166` — `Loader2 w-8 h-8` + "Szervizek betöltése..." felirat. **Mojibake**: a hex escape (`ö` stb.) látszólag nem renderelődik megfelelően a forrásban — ellenőrizendő, hogy a vágólapon helyesen jelenik-e meg ("betöltése").

**Sehol nincs skeleton loader** — minden oldal spinner-only. Tab-váltáskor (VehicleDetailPage health/reminders/costs) **nincs** loading state, hanem `?? []` üres fallback → user villog/azt hiszi üres, közben tölt.

## C) Error retry

- **severity: MEDIUM** — retry gombok inkonzisztensek és hiányosak.

### Találatok

- `GaragePage.tsx:236-241` — error banner van ("Nem sikerült betölteni a járműveket. Próbáld újratölteni az oldalt."), de **NINCS retry gomb** — csak natural language utasítás (oldal-frissítés manuálisan). `useVehicles` `refetch` nincs kihasználva.
- `HistoryPage.tsx:353-364` — **JÓ**: `<button onClick={() => refetch()}>Újrapróbálás</button>` magyar címke, csak a táblázat törzsében. Stats fail esetén (sor 488/507/523) viszont végtelenségig `...` látszik, **statsError** nincs kezelve.
- `ResultPage.tsx:555-584` — **JÓ**: dedikált full-page error oldal két CTA-val: "Újrapróbálás" (`refetch()`) + "Új diagnózis készítése" (navigate). Magyar copy. Nincs hibakód kiírva, ami fejlesztői debughoz hiányos, viszont user-friendly.
- `ServiceComparisonPage.tsx:169-176` — **NINCS retry gomb**, csak hibaüzenet (`error.message`) — ez Vite/JS Error message-t mutat az usernek (pl. "Network Error"), ami nem magyar és nem felhasználóbarát.
- `VehicleDetailPage.tsx:116-133` — **NINCS retry**: 404-szerű empty page CTA-val a garázs listához ("Vissza a garázs listához"), de tranziens hálózati hibán is örökre 404-ot mutat.
- `VehicleDetailPage` recalls/health/costs query-k (sor 94-97) — `useVehicleHealth`, `useReminders`, `useCosts`, `useVehicleRecalls` **nem ellenőrzi** az `isError`-t. Hiba esetén user-nek úgy tűnik, mintha minden 0/üres lenne (`?? []`, `?? null` fallback) — silent fail.

## Olvasott fájlok

- `/home/user/AutoCognitix/frontend/src/pages/GaragePage.tsx`
- `/home/user/AutoCognitix/frontend/src/pages/HistoryPage.tsx`
- `/home/user/AutoCognitix/frontend/src/pages/ResultPage.tsx`
- `/home/user/AutoCognitix/frontend/src/pages/VehicleDetailPage.tsx`
- `/home/user/AutoCognitix/frontend/src/pages/ServiceComparisonPage.tsx`
- `/home/user/AutoCognitix/frontend/src/pages/DiagnosisPage.tsx` (grep only)
- `/home/user/AutoCognitix/frontend/src/pages/ChatPage.tsx` (grep only)
- pages/ directory listing (ls)
