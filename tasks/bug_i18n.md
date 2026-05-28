# i18n & Magyar Konzisztencia

## A) Hard-coded angol UI
- severity: MEDIUM (kevés direkt angol UI string maradt — fő gond a magyar ékezet-hiány)
- találatok:
  - `frontend/src/pages/LoginPage.tsx:146` — test mock: `new Error('Login failed')` (csak teszt)
  - `frontend/src/pages/InspectionPage.tsx:179` — `<option value="">Valasszon gyartot</option>` (ékezet nélkül, nem angol, de nem helyesírású magyar)
  - `frontend/src/components/VehicleSelector.tsx:399` — `<span>Dekodolas...</span>` (ékezet nélkül; helyesen "Dekódolás...")
  - `frontend/src/pages/ServiceComparisonPage.tsx:173` — fallback: `'Ismeretlen hiba'` OK; de `error.message` natívan ANGOL lehet (Axios/fetch hibák)
  - `frontend/src/pages/CalculatorPage.tsx:278` — `<Badge>Diagnozis alapjan</Badge>` (ékezet nélkül)
  - **Konzisztencia hiba**: a kódbázis kevert — `'Emlékeztető törölve'` (ékezettel, GaragePage:195) vs. `'Vizsga kockazat elemzes kesz!'` (ékezet nélkül, InspectionPage:122). Nincs egységes szabály.
  - Test file: `frontend/src/pages/__tests__/LoginPage.test.tsx:146` — `'Login failed'` (csak teszt, OK)

## B) Backend error msgs
- severity: HIGH (2 hely angolul maradt prod auth flow-ban)
- konzisztencia: `auth.py` 19 HTTPException-ből 17 magyar, 2 angol
- találatok:
  - `backend/app/api/v1/endpoints/auth.py:959` — `detail="Password reset service temporarily unavailable"` ANGOL
  - `backend/app/api/v1/endpoints/auth.py:1057` — `detail="Password reset service temporarily unavailable"` ANGOL (duplikálva!)
  - `backend/app/api/v1/endpoints/dtc_codes.py:636` — `detail="Invalid DTC code format. Expected format: P0101, B1234..."` ANGOL
  - `backend/app/api/v1/endpoints/dtc_codes.py:943` — `detail="Bulk import failed, all changes rolled back."` ANGOL (admin, kevésbé kritikus)
  - pozitív minta: `services.py:110,150,193,204` mind magyar és konzisztens
  - jó architektúra: `error_handlers.py:91-120` `build_error_response()` támogat `message` (en) + `message_hu` dual-output-ot, és `exceptions.py:83` `ERROR_MESSAGES_HU` map tartalmaz fordításokat — de a direkt `HTTPException(detail=...)` hívások KIKERÜLIK ezt.

## C) i18n lib gap
- severity: CRITICAL (skálázhatósági blokker)
- `frontend/package.json` — NINCS i18next / react-intl / formatjs / lingui dep. Grep eredmény: `NO_I18N_LIB`.
- Jelenleg ~100+ hard-coded magyar string szétszórva komponensek között (toast, label, placeholder, aria-label, Badge szöveg).
- Következmények:
  - angol/német piacra terjeszkedés = manuális string-vadászat minden komponensben
  - SSR / locale formatting (dátum, szám, pénz) inkonzisztens (`toLocaleDateString('hu-HU')` szétszórva)
  - copywriter / fordító nem tud közvetlenül szerkeszteni, fejlesztő kell minden szövegmódosításhoz
- Javaslat:
  1. **rövid táv**: `react-i18next` + `i18next` telepítése, `locales/hu.json` + `locales/en.json`, lazy load
  2. **migráció lépcsőzetes**: új komponensek azonnal `t('key')`-vel; meglévők sprint-enként 1-2 oldal
  3. **backend párhuzamosan**: minden új HTTPException-nek `AutoCognitixException(code=..., message=...)` formát kell használnia (a meglévő `ERROR_MESSAGES_HU`-t kihasználva), NEM direkt `HTTPException(detail="...")`
  4. **konzisztencia szabály**: a 4-5 ékezet nélküli string (`Valasszon`, `Dekodolas`, `Diagnozis`) javítása ékezetes verzióra — vagy a teljes kódbázis transliterate-elése (egyik vagy másik, nem keverve)

## Olvasott fájlok
- `frontend/package.json` (grep)
- `backend/app/core/error_handlers.py` (1-120)
- `backend/app/api/v1/endpoints/auth.py` (128-140, 955-960, grep teljes)
- `backend/app/api/v1/endpoints/dtc_codes.py` (grep)
- `backend/app/api/v1/endpoints/services.py` (grep)
- `backend/app/core/exceptions.py` (grep)
