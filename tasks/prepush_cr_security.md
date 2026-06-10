# Pre-Push Security Lead Audit — claude/code-review-fixes

Scope: `git diff main` (working tree, 20 fájl + 2 új: `pii.py`, `vehicle_makes.py`).
Fókusz: A) PII redaction / Sentry before_send, B) vehicle_make validator, C) cd.yml rollback script.

## Összefoglaló

| # | Súlyosság | Fókusz | Találat |
|---|-----------|--------|---------|
| 1 | HIGH | A | `exception.values[].value` nincs redaktálva a before_send-ben |
| 2 | MEDIUM | A | breadcrumbs message/data nincs redaktálva |
| 3 | MEDIUM | B | "after" validator megkerüli a min_length=1-et (whitespace input → "") |
| 4 | LOW | C | rollback job: hiányzó explicit `permissions: issues: write` |
| 5 | LOW | C | version/tag név markdown-injekció az issue body-ba (env-en át, nem kód-injekció) |
| 6 | INFO | C (out-of-scope) | prepare job: `${{ github.event.release.tag_name }}` bash-be interpolálva (pre-existing) |

## A) pii.py + `_sentry_before_send` (backend/app/core/logging.py:583-619)

### 1. HIGH — Exception üzenetek nem redaktálódnak
`_sentry_before_send` csak `request.url/query_string`, `extra`, `message`, `logentry.message` mezőket redaktálja. Az `event["exception"]["values"][i]["value"]` (exception message string) kimarad — pl. `httpx.HTTPStatusError` URL-lel, vagy `ValueError(f"Invalid VIN {vin}")` változatlanul kimegy Sentry-be. Mivel `LoggingIntegration(event_level=ERROR)` + `attach_stacktrace=True` mellett az exception event a leggyakoribb event-típus, ez a redaction fő kiskapuja. A modul docstring GDPR-igénye ("must be stripped from any text that leaves the system") így nem teljesül.
- Fix: `for exc in event.get("exception", {}).get("values", []): exc["value"] = redact_pii(exc["value"])` (isinstance-guardokkal).

### 2. MEDIUM — Breadcrumbs kimaradnak
`event["breadcrumbs"]["values"][i]["message"]` és `["data"]` (pl. httpx/logging breadcrumb-ok korábbi log sorokkal, URL-ekkel) nem redaktált. Egy error event 100 megelőző breadcrumb-ot vihet ki UUID-s/VIN-es URL-ekkel.
- Megj.: `request.headers` (Referer) és `request.data` szintén kimarad, de `send_default_pii=False` + `FastApiIntegration` default mellett ezek korlátozottak — a breadcrumb a reálisabb szivárgási út.

### Pozitívumok (A)
- Exception-safe: teljes try/except, mindig `return event` — event sosem vész el, sosem dob. OK.
- ReDoS: mindkét regex (pii.py:14-20) fix hosszú karakterosztály, nincs nested/ambiguous kvantor → lineáris, nincs ReDoS. OK.
- VIN regex `\b...{17}\b` IGNORECASE: 17 hosszú hex/random ID-kre false positive lehet, de redaction-nél a false positive biztonságos irányú. OK.
- UUID→VIN sorrend determinisztikus, dokumentált. OK.

## B) vehicle_make validator (schemas/diagnosis.py:34-44, 484-489 + core/vehicle_makes.py)

### 3. MEDIUM — "after" validator visszaadhat üres stringet
A `field_validator` default módja "after": a `min_length=1` ELŐBB fut, tehát `None`/hiányzó értéket Pydantic elutasít, a validator mindig `str`-t kap — exceptiont nem dob (`.strip()`, `.lower()`, dict `.get()` bármilyen unicode-ra biztonságos). DE: `vehicle_make=" "` átmegy a `min_length=1`-en, majd `normalize_make` `""`-t ad vissza, és az "after" validator visszatérési értékét Pydantic már NEM ellenőrzi újra a constraint-ek ellen → üres make jut a pipeline-ba (Qdrant filter, LLM prompt, NHTSA hívás üres make-kel).
- Fix: a validatorban `if not cleaned: raise ValueError("vehicle_make cannot be blank")`, vagy `normalize_make` után ellenőrzés.

### Pozitívumok (B)
- Idempotens: minden canonical érték lowercase kulcsként önmagára képződik (kézzel ellenőrizve: McLaren, RAM, DS, Citroën, smart, FCA, AM General mind OK). Determinisztikus, nincs locale-függés (`str.lower()` locale-független).
- Nincs Title-case fallback → ismeretlen input változatlan, nincs adatkorrupció. OK.
- Injection-szempontból semleges: nem épül SQL-be/Cypher-be stringként a diffben.

## C) cd.yml rollback issue script (.github/workflows/cd.yml:362-450)

### 4. LOW — Hiányzó least-privilege permissions
A `rollback` jobnak nincs `permissions` blokkja (a workflow-nak sincs top-level), így a default token-jogokat örökli. Issue-create-hez `permissions: { issues: write, contents: read }` lenne a minimális. Ha a repo default-ja restricted (read-only), a step csendben elbukik (`continue-on-error: true` el is nyeli).

### 5. LOW — Markdown injection az issue body-ba (nem kód-injekció)
`DEPLOY_VERSION` (= release tag név) és `DEPLOY_ENVIRONMENT` env-változón keresztül kerül a scriptbe — ez a HELYES minta: nincs `${{ }}` interpoláció a script body-ban, script injection NINCS. A tag név tartalma viszont nyersen kerül az issue markdown body-ba; tag-et csak push-joggal rendelkező hozhat létre, ezért csak LOW (kozmetikai markdown-torzítás).
- Ellenőrizve: a script body kizárólag `process.env.*` és `context.*` (trusted) értékeket használ. Secret nem kerül se env-be, se body-ba.

### 6. INFO (out-of-scope, pre-existing) — prepare job tag interpoláció
cd.yml:42-44: `${{ github.event.release.tag_name }}` közvetlenül bash `run` scriptbe interpolálva. Git ref-név nem tartalmazhat szóközt, de `$`, `` ` ``, `;`, `()` igen → `v1$(...)` alakú tag elvi command injection. Nem a mostani diff része; külön issue-ként javasolt (`${TAG}` env-átadással javítandó).

## Verdikt

**Nincs CRITICAL.** 1 HIGH (#1, exception values redaction) — a Pre-Push Protocol szerint **push BLOKKOLVA** a javításáig (pár soros fix a `_sentry_before_send`-ben). #2-#3 MEDIUM: push engedélyezett TODO-val, de #3 olcsón javítható ugyanabban a körben. #4-#6 nem blokkoló.

