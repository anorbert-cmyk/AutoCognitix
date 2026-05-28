# Wave2 R3 — Operational

Audit scope: HEAD = `4258ddc` on `claude/sprint13-bugfixes-wave2`.
Files: `.github/workflows/cd.yml`, `backend/app/core/error_handlers.py`.

---

## A) cd.yml rollback semantics

- **severity:** LOW (observation, not a blocker)
- **finding:**
  Az új ordering `run-migrations` → `deploy-railway` helyes (app sosem indul
  el régi schema-val). A failure-path is rendben: ha `run-migrations` FAIL-el,
  `deploy-railway` `needs: [..., run-migrations]` miatt automatikusan skipped,
  a régi backend production-ben marad — ez a kívánt viselkedés.

  **Dangling tag analízis (build-backend outputs):**
  - A `build-backend` job 5 tag-et push-ol a GHCR-ba: `{{version}}`,
    `{{major}}.{{minor}}`, `sha-<short>`, `latest` (csak release-kor),
    és `<environment>` (`staging` / `production`).
  - `outputs.image` és `outputs.digest` definiálva van, **DE a workflow-ban
    SEMMI nem fogyasztja** őket (`grep needs.build-backend.outputs` → 0 hit).
    A `deploy-railway` `railway up`-pal source-ból build-el, nem a registry
    image-ből → nincs runtime hatás.
  - **Következmény:** ha a migration FAIL, a registry-ben marad egy
    `:staging` / `:latest` / `:<version>` tag, ami olyan kódra mutat, ami
    **soha nem futott migration-nel** sikeresen. Mivel Railway nem ezt
    a tag-et használja, a deployment NEM lesz korrupt — de a tag-ek
    mutable módon megmaradnak, és ha valaki manuálisan `docker pull
    ghcr.io/.../backend:staging`-ot tesz (pl. lokális repro, vagy
    jövőbeli registry-alapú deploy), inkonzisztens kódot kap.
  - Tényleges manual cleanup most NEM szükséges (mert nincs fogyasztó).
    Jövőbeli kockázat: ha bárki bekapcsolja a registry-alapú Railway
    deploy-t (`railway service --image ghcr.io/...:staging`), a fenti
    pattern egyből éles probléma lesz.

  **Javaslat (nem blokkoló):**
  Vagy (a) töröld a használatlan `outputs.image` / `outputs.digest`
  blokkokat a `build-backend` / `build-frontend` job-okból (clean
  signal: "ez a job csak push-ol, nem konzumálnak belőle"); vagy (b)
  ha jövőbeli registry-alapú deploy várható, mozgasd a `build-backend`-et
  `run-migrations` UTÁN (drága: párhuzamosság elveszik), vagy adj hozzá
  egy `cleanup-failed-tags` job-ot `if: failure() && needs.run-migrations
  .result == 'failure'` feltétellel, ami `gh api -X DELETE` paranccsal
  törli a friss `sha-<x>` tag-et.

---

## B) `_redact_pii` overhead

- **severity:** INFO (observation, not actionable)
- **finding:**
  `_redact_pii(path)` két `re.sub` hívást futtat (`_UUID_RE`, `_VIN_RE`)
  minden 5xx exception-ön (mind az `autocognitix_exception_handler`,
  mind a `sqlalchemy_exception_handler`, mind a `generic_exception_handler`
  hívja a `_capture_to_sentry`-t, ami a `_redact_pii`-t).

  **Költségbecslés:**
  - A két pattern **modul-szinten compile-olt** (lines 314–318) — pontosan
    ezt akarjuk perf szempontból, nem per-call compile.
  - URL path tipikusan < 200 char. Compiled regex `re.sub` ekkora input-on
    ~5–15 mikroszekundum / call → 10–30 µs total / 5xx.
  - 100 req/sec 5xx (irreális worst case prod-ban): ~3 ms CPU / sec
    = 0.3% egy core-ból. Mérhetetlen.
  - Reális prod 5xx rate (1/sec): ~30 µs / sec = teljesen elhanyagolható.

  **Nincs javaslat.** A regex compile pattern már optimális, a redaction
  helye (csak 5xx ágban) helyes (nem fut a 4xx / 2xx hot path-on). Csak
  rögzítjük, hogy a kód-review során észrevettük és tudatosan elfogadjuk.
