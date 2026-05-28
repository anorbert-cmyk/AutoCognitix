# Wave2 Push — Logic/Correctness Lead

Branch: `claude/sprint13-bugfixes-wave2` (commit `4f84293`)
Scope: 3 fókusz pont — NHTSA alias normalizáció, embedding cache verziózás, `_thread_pool` megosztás.

---

## A) NHTSA alias lookup edge cases

**severity:** HIGH (silent zero-recall results for valid US makes)

### A1. Casing fallback bug — leggyakoribb felhasználói bemenet kiesik
- **file:line:** `backend/app/services/nhtsa_service.py:310-313`
- `_normalize_make("VOLKSWAGEN")` → `key="volkswagen"` → nincs a `BRAND_ALIASES`-ban (csak `"vw"` és `"volkswagen ag"` van) → fallback ágon `return make.strip()` → **"VOLKSWAGEN"** (eredeti casing megmarad).
- Ugyanez: `"Volkswagen"`, `"VolksWagen"`, `"Toyota"` mind a fallback ágba esik (mert maguk a kanonikus alakok NEM kulcsok a dictben), és **a normalizálás soha nem érinti őket**, ami szerencsére itt OK — DE ha valaki "TOYOTA" all-caps-szel jön, az pont úgy ahogy van megy a NHTSA-hoz. NHTSA `recallsByVehicle` enyhe case-érzékenységű egyes mezőkre, az inkonzisztencia cache-fragmentációt is okoz: `_generate_cache_key("recalls", "VOLKSWAGEN", ...)` vs `("Volkswagen", ...)` → két külön cache slot.
- **Fix javaslat:** fallback ágban is `return make.strip().title()` (vagy egy whitelist-alapú kanonizálás). Még jobb: `_normalize_make` mindig normalizált formát (pl. Title Case) adjon vissza, és a `EU_ONLY_MAKES` is normalizált alakkal hasonlítson (jelenleg `.lower()`-rel — működik, csak inkonzisztens).

### A2. Whitespace / üres input
- **file:line:** `backend/app/services/nhtsa_service.py:312-313`, `:562`, `:567`
- `_normalize_make("")` → `""`. Aztán `"" in EU_ONLY_MAKES` False → NHTSA hívást kap üres `make=` paraméterrel → 400 vagy értelmetlen 200. Nincs guard a hívó oldalon sem (`get_recalls` nem validál üres make-et a `.strip()` után).
- `_normalize_make("  vw  ")` → `key="vw"` → "Volkswagen". OK, működik.
- `_normalize_make("Mercedes ")` → `key="mercedes"` → "Mercedes-Benz". OK.
- **Fix javaslat:** `if not make.strip(): raise ValueError("make is required")` a `get_recalls` / `get_complaints` elején.

### A3. Aliasok hiányoznak / következetlenek
- **file:line:** `backend/app/services/nhtsa_service.py:287-307`
- `"alfa romeo"` (space, no dash) NEM kulcs — csak `"alfa-romeo"` van. Szerencsére `EU_ONLY_MAKES` tartalmazza a `"alfa romeo"` lowercase formát (`:276`), tehát kiszűrődik mint EU-only. De ha a kanonikus alakra rakták volna (és Alfa Romeo amúgy adott évjáratoknál létezett US-ban — pl. Stelvio, Giulia 2017–), akkor mindkét casing/separator variánst kellene kezelni.
- `"mercedes-benz"` (lowercase, kötőjellel) NEM kulcs → fallback → `"Mercedes-Benz"` ha kis kezdőbetű volt; `"mercedes-benz"` ha kis volt. Inkonzisztens.
- `"land rover"` (space, no dash, lowercase) NEM kulcs (csak `"land-rover"`, `"landrover"`). User-tipikus bevitel kiesik.
- `"mini cooper"` → "MINI" — DE Mini Cooper egy modell, nem márka. Ha user `make="Mini Cooper" model="Hardtop"`-ot ír, a make jó lesz de inkonzisztens szemantikailag. Apró.
- Hiányzó gyakori aliasok: `"mb"` van de `"benz"` nincs; `"chevrolet"` nincs külön (`"chevy"` van), `"vauxhall"` EU-only de `"opel"` is — magyar user gyakran ír Opel-t, OK.

### A4. `_normalize_make` nem `@staticmethod` viselkedés
- **file:line:** `backend/app/services/nhtsa_service.py:309-313`
- `@classmethod` de a `cls` használata egyetlen helyen van: `cls.BRAND_ALIASES`. Mivel a dict osztály-attribútum, ez OK; csak megjegyzés, nem hiba.

**Bizonyítható mismatch-ek** (3 db):
1. User ír `"VOLKSWAGEN"` → kimegy `"VOLKSWAGEN"`-ként a NHTSA-ba (nem `"Volkswagen"`-ként, amit a komment ígér).
2. User ír `"mercedes-benz"` → kimegy `"mercedes-benz"`-ként (nem `"Mercedes-Benz"`-ként).
3. User ír `"land rover"` (space, lowercase) → kimegy `"land rover"`-ként; csak a `"land-rover"` / `"landrover"` / `"range rover"` triggerel.

---

## B) Cache version migration

**severity:** LOW (termék-szempontból elfogadható, csak dokumentálandó)

- **file:line:** `backend/app/db/redis_cache.py:565-575`
- Új cache-kulcs séma: `sha256(f"{HUBERT_MODEL}@{HUBERT_REVISION}|" + text)`. Helyes: izoláció model + revision váltáskor.
- **Migration hiánya:** Nincs back-fill, nincs alias lookup a régi (csak `sha256(text)`) kulcsokhoz. `HUBERT_REVISION` `"main"` → `"abc1234"` váltás esetén:
  - Redis cache: 35k+ kulcs hidegre vált. TTL `CacheTTL.EMBEDDINGS` szerint magától lejár, addig holt súly → memória + eviction nyomás.
  - **Qdrant: NEM érinti** ez a változás (a 35k vektor a Qdrant kollekcióban van, nem Redis-ben). A Redis embedding cache csak egy gyors lookup réteg ismételt query stringekre — Qdrant a tényleges keresés.
- **Tehát:** "35k vektor elveszik" tévedés lenne — csak a Redis warm cache invalidálódik. A Qdrant vektorok stabilak (külön reindex kell ha a modell ténylegesen vált, ami egy másik, súlyosabb művelet — lásd alább).
- **Külön (és súlyosabb) kockázat:** ha `HUBERT_REVISION` változik DE a Qdrant kollekciót NEM reindexeljük, akkor a query embedding (új revision) és a stored embedding (régi revision) közti cosine similarity zajos lesz — recall-vesztés diagnosztikai találatoknál. Erre se kulcs-séma, se reindex hook nincs. **Új sprint backlog item javaslat:** revision-bump előtt kötelező reindex playbook.
- **Mai fix érdeme:** helyes, csak a Redis-szinkronizáció kérdését rendezi, és pontosan azt a hibát előzi meg, hogy régi-revision-vektort visszakap a kliens.

### Dokumentálandó (CLAUDE.md / lessons.md javaslat):
- "HUBERT_REVISION váltás → Redis embedding cache cold restart; Qdrant kollekció kézi reindex szükséges, különben recall-vesztés."

---

## C) `_thread_pool` kontamináció

**severity:** MEDIUM (head-of-line blocking kockázat, de a default pool kontamináció rosszabb lenne)

- **file:line:** `backend/app/services/diagnosis_service.py:425-429`, `backend/app/services/embedding_service.py:57`, `:683-684`, `:754-755`
- A pool: `ThreadPoolExecutor(max_workers=4)` — modul-szintű globális.
- Új használók: `diagnosis_service.preprocess_hungarian` (per request, 1 task). Régiek: `embed_text` (per request, 1 task), `embed_text_batch` (per batch, 1 task).
- **Worst-case forgatókönyv:** 4 párhuzamos diagnosis request, mindegyik először `preprocess_hungarian`-t hív (4 worker mind elfoglalva), majd `embed_text`-et (sorba állnak). Két szekvenciális szakasz a request-en belül serializálódik — de mivel egy request soha nem indít 2-t párhuzamosan, ez NEM deadlock, csak head-of-line.
- **Trade-off elemzés:**
  - **Régi (default executor) baj:** asyncio default executor min(32, cpu_count+4) worker, OSZTOZIK `loop.getaddrinfo`-val (DNS), `aiofiles`-szal, és minden más `run_in_executor(None, ...)` hívóval (lásd `rag_service.py:97`, `:766`, `email_service.py:401`, `:591`). Egy lassú HuBERT preprocess (HuSpaCy ~50-200ms CPU-bound) blokkolhat DNS lookup-ot → cascade lassulás.
  - **Új (_thread_pool, max=4) baj:** ML-bound munka egy szűk poolban — saturation könnyebben elérhető (4 worker), de izolált. DNS, fájl I/O, email SMTP védve.
  - **Egyensúly:** új megoldás összességében jobb (izoláció > kapacitás ebben az esetben), DE a 4-es méret szűk lehet, ha a `preprocess_hungarian` + `embed_text` szekvenciális CPU munkája fő bottleneck.
- **Javaslat:**
  1. Méret felfelé: `max_workers=8` (vagy `min(8, (os.cpu_count() or 2) * 2)`) — még mindig izolált, de nagyobb párhuzam.
  2. Megnevezett pool: `ThreadPoolExecutor(max_workers=..., thread_name_prefix="nlp")` — log/profiling kedvéért.
  3. **Public API:** a `_thread_pool` egy aláhúzással kezdődik (private konvenció), de most már külső modul (`diagnosis_service`) importálja. Vagy nevezd át `nlp_thread_pool`-ra, vagy adj `get_nlp_executor()` getter-t — explicit szerződés.
  4. **App shutdown:** sehol nincs `_thread_pool.shutdown(wait=True)` — process exit-kor leaked workerek. Add hozzá FastAPI `lifespan` shutdown handler-hez.

---

## Olvasott fájlok
- `/home/user/AutoCognitix/backend/app/services/nhtsa_service.py` (255-665)
- `/home/user/AutoCognitix/backend/app/services/embedding_service.py` (1-60, 680-760)
- `/home/user/AutoCognitix/backend/app/services/diagnosis_service.py` (420-435)
- `/home/user/AutoCognitix/backend/app/db/redis_cache.py` (560-595)
- `/home/user/AutoCognitix/backend/app/core/config.py` (145-160)
- `/home/user/AutoCognitix/backend/app/services/rag_service.py` (grep, 97 + 766)
- `/home/user/AutoCognitix/backend/app/services/email_service.py` (grep, 401 + 591)
- git diff `main...HEAD` (commit `4f84293`)
