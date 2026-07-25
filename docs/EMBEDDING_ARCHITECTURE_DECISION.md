# ADR — HuBERT embedding a production runtime-ban: ONNX Runtime fp32

> **Státusz:** ELFOGADVA és SZÁLLÍTVA
> **Döntés dátuma:** 2026-07-25 · **Implementáció:** `050965e` (`feat(embedding): run huBERT through ONNX Runtime so production can embed`)
> **Kapcsolódó szállítás:** `c64dcf1` (RAG collection-drift javítás), `c700f1f` (`common-issues` outage ≠ no-data)
> **Hatókör:** `backend/app/services/embedding_service.py`, `backend/app/core/exceptions.py`, `backend/app/core/config.py`, `backend/Dockerfile.prod`, `backend/requirements.prod.txt`, `backend/app/db/qdrant_client.py`, `backend/app/services/rag_service.py`, `backend/app/api/v1/endpoints/health.py`, Qdrant `autocognitix` collection

> **Ez a dokumentum korábban egy döntési javaslat volt.** Az implementáció azóta landolt, és több ponton **eltért** a tervtől (más fájlszerkezet, más kivétel-hierarchia, más beállítás-készlet, újramért paritás-számok). Ez a változat azt írja le, ami **ténylegesen a kódban van**. A mérési bizonyítékok (3. fejezet) megmaradtak — azok az érvelés alapjai, és ma is érvényesek.

---

## 1. A probléma

A magyar nyelvű szemantikus keresés a production környezetben **nem működött**. A `/api/v1/dtc/search?use_semantic=true` szemantikus ága és a zászlóshajó `/api/v1/diagnosis/analyze` RAG pipeline-ja **nulla Qdrant kontextust** kapott.

A hiba **csendes** volt — nem dobott hibát, nem logolt errort, nem jelent meg a válaszban. Ezért maradhatott hónapokig rejtve.

### 1.1 Első root cause — a nullvektor

```
requirements.prod.txt (torch/transformers kikommentelve)
   └─> Dockerfile.prod build: torch NINCS az image-ben
        └─> embedding_service.py  ImportError -> TORCH_AVAILABLE = False
             └─> embed_text()     return [0.0] * 768      # NULLVEKTOR
                  └─> Qdrant cosine search nullvektorral
                       ├─> rag_service score_threshold=0.5 -> GARANTÁLTAN []
                       └─> dtc_codes.py threshold nélkül -> értelmetlen, score=0 találatok
```

**Kritikus megfigyelés:** a nullvektor nem "rossz" találatot ad, hanem **matematikailag értelmetlen** eredményt. Cosine távolságnál a Qdrant normalizálja a query vektort; egy nulla normájú vektor normalizálása nulla vektort ad, így minden dot product 0.0. A `score_threshold=0.5` szűrő ezt garantáltan üres listává alakítja — ami **megkülönböztethetetlen** attól, hogy tényleg nincs találat.

Ráadásul a `[0.0]*768` **típushelyes**: 768 elemű float lista. Minden downstream ellenőrzés (dimenzió, típus) átengedte. Nem létezett olyan invariáns a rendszerben, ami kimondta volna: *"egy similarity search query vektorának egységhosszúnak kell lennie"*.

### 1.2 Második, független root cause — collection drift

A torch hiánya **nem az egyetlen** ok volt. A `/diagnosis/analyze` RAG pipeline **rossz Qdrant collection-öket** kérdezett:

| Hely | Collection | Tartalom |
|------|-----------|----------|
| `rag_service.py` DTC-ág | `QdrantService.DTC_COLLECTION` = `dtc_embeddings_hu` | **üres** |
| `rag_service.py` symptom-ág | `QdrantService.SYMPTOM_COLLECTION` = `symptom_embeddings_hu` | **üres** |
| `dtc_codes.py` → `search_dtc()` → `search_unified()` | `settings.QDRANT_UNIFIED_COLLECTION` = **`autocognitix`** | **itt vannak a HuBERT vektorok** |

Az `autocognitix` collection **csak `dtc` / `complaint` / `recall` payload-típust tartalmaz** (`scripts/index_qdrant_hubert.py`, `"type"` mezők) — **`symptom` típus nem létezik benne.** A "symptom search" ág tehát nem egyszerűen rossz collectiont kérdezett; a keresett entitástípus **egyáltalán nincs indexelve**.

A `vehicle_make` exact-match filter a symptom-ágon egy harmadik réteg volt: a unified payload kulcsa `"make"`, nem `"vehicle_make"`, és **nyers NHTSA all-caps** ("VOLKSWAGEN") — tehát garantáltan 0 találat.

### 1.3 Miért maradhatott hónapokig rejtve — a valódi tanulság

Három egymásra rakódó **csendes** fallback:

1. `embedding_service.py` — nincs torch → `[0.0]*768`, `logger.warning` szinten, **kivétel nélkül**
2. `rag_service.py` — `except Exception` → `query_embedding = None` → `return []`, `logger.warning`
3. `diagnosis_service.py` — `except Exception` → `_fallback_diagnosis()`, ami a felhasználónak **ugyanúgy néz ki**, mint egy valódi AI-jelentés

Mindhárom réteg a hibát **üres eredménnyé** alakította. Egy üres eredmény pedig egy diagnosztikai keresőben teljesen legitim válasz. **Nem volt olyan pont a rendszerben, ahol a "nem tudtam megkérdezni" megkülönböztethető lett volna a "megkérdeztem, nincs válasz"-tól.**

> **Ez a dokumentum legfontosabb mondata:** a javítás értékének nagyobb része nem az embedding visszahozása, hanem annak garantálása, hogy **legközelebb hangosan hasítson**.

---

## 2. Kényszerek — bizonyítékkal

### 2.1 A prod image nem tartalmazott ML stacket

A `backend/requirements.prod.txt` egy kommentblokkal zárta ki a `torch`-ot ("~2GB"), a `transformers`-t és a `sentence-transformers`-t; a `Dockerfile.prod` builder stage kizárólag ezt a fájlt telepítette.

### 2.2 A runtime embedding spec — amit bármely backendnek reprodukálnia kell

| Lépés | Pontos viselkedés |
|-------|-------------------|
| Modell | `SZTAKI-HLT/hubert-base-cc`, BERT-base architektúra, 110 618 112 paraméter |
| Tokenizálás | `padding=True, truncation=True, max_length=512` |
| **Pooling** | attention-mask súlyozott **mean pooling** a `last_hidden_state`-en: `sum(emb * mask) / clamp(mask.sum(), min=1e-9)` |
| **Normalizálás** | L2 (`p=2`, `dim=1`) → egységhosszú vektor |
| Kimenet | `List[float]`, 768 dim |

**A döntés szempontjából a legfontosabb tény: a pooling és az L2-normalizálás a modellen KÍVÜL, tiszta tensor-aritmetikával történik.** Bármely inference backend, ami ugyanazt a `last_hidden_state`-et adja vissza, ugyanezt a poolingot numpy-ban elvégezve **azonos embedding-térben** marad. Ez tette az ONNX opciót reálissá.

### 2.3 Preprocessing (`preprocess_hungarian`) — jelenleg NO-OP

A `spacy`/`huspacy` **sem** a dev, **sem** a prod requirements-ben nincs telepítve → `SPACY_AVAILABLE = False` → a `preprocess_hungarian()` az azonosság-függvény (`return text.strip()`).

Ez **szerencsés véletlen**: az indexelő script `preprocess=False`-szal futott, a runtime viszont `preprocess=True`-val hívott. Mivel a preprocess mindkét oldalon no-op, nem volt query/index eltérés.

> **FIGYELMEZTETÉS (ma is érvényes):** ha valaki később hozzáadja a `spacy`+`hu_core_news_lg`-t a prod image-hez, a query szövegek lemmatizálódnak, az indexelt vektorok viszont nyers szövegből készültek → **azonnali, csendes minőségromlás**. A `spacy` hozzáadása tehát külön, tudatos döntés, ami **teljes reindexet igényel**.

### 2.4 Railway / deployment kényszerek

| Tény | Forrás | Érték |
|------|--------|-------|
| Builder | `backend/railway.toml` | `DOCKERFILE`, `Dockerfile.prod` |
| Healthcheck path / timeout | `backend/railway.toml` | `/health` / **100 s** |
| Restart policy | `backend/railway.toml` | `ON_FAILURE`, max 3 retry |
| `/health` költsége | `app/main.py` | statikus JSON, **nem** érint adatbázist, nem érint embeddinget |
| `warmup()` a boot-on | `app/main.py` lifespan | **NINCS meghívva** — a modell lazy, az első embed hívásnál töltődik |

**Következtetés:** a modell betöltése **nem** a healthcheck kritikus útján van. Ez a mai ONNX-képnél is így maradt (lásd 5.4).

---

## 3. Mért bizonyíték

> Minden szám ténylegesen mérve (PyPI JSON API, `download.pytorch.org/whl/cpu`, illetve izolált venv-ben lefuttatott export + összehasonlítás, 2026-07-25). Ahol nem sikerült mérni, ott **BECSLÉS** jelölés van.

### 3.1 Csomagméretek

**A torch kérdés — az eredeti kizárás egy wheel-index félreértésen alapult:**

| Csomag | Wheel | Megjegyzés |
|--------|-------|-----------|
| `torch==2.2.0` (PyPI default, cp311 manylinux1_x86_64) | 720,5 MB | CUDA-bundled |
| + 11 db `nvidia-*` wheel (cudnn 697,8 / cublas 391,6 / cusparse 186,9 / nccl 158,3 / cusolver 118,4 / cufft 116,0 / curand 53,9 / …) | 1 759,8 MB | `platform_system=="Linux" and platform_machine=="x86_64"` esetén **feltétel nélkül** |
| + `triton==2.2.0` | 160,2 MB | szintén Linux-x86_64-re pinelve |
| **PyPI default ÖSSZESEN** | **2 640,4 MB (2,58 GiB)** | ← ez volt a `requirements.prod.txt` "~2GB" megjegyzése |
| **`torch==2.2.0+cpu`** (pytorch.org/whl/cpu, cp311) | **178,1 MB** | **nulla nvidia/triton dep** |

> **Megtakarítás CUDA → CPU wheel: 2 462 MB (2,40 GiB).** Az eredeti kizárási döntés tehát a helyes wheel-index ismeretében nem állt meg — de a torch még CPU-ban sem "kicsi".

**Telepített méret (venv site-packages, `du`-val mérve):**

| Stack | Méret |
|-------|-------|
| `torch(+cpu)` egyedül | **894 MB** (torch 754 + sympy 80 + networkx 19 + mpmath 5,1) |
| `onnxruntime` egyedül | **58 MB** |
| minimális deployolható stack (`onnxruntime`+`tokenizers`+`numpy`) | **176 MB** (pip/setuptools nélkül) |

**Modell-artifactok:** `model.safetensors` **445,0 MB** · `hubert_fp32.onnx` **440,36 MB — egyetlen fájl**, nincs külső `.onnx_data`.

**Teljes image-delta összevetés (runtime + modell):**
- **torch út:** 894 MB + 445 MB ≈ **1 339 MB**
- **ONNX út:** 176 MB + 440 MB ≈ **616 MB**
- **Különbség: ~720 MB (−54%)**

### 3.2 Numerikus ekvivalencia — torch ↔ ONNX fp32

**(a) Feltáró mérés** — Python 3.11.15, `torch 2.13.0+cpu`, `onnxruntime 1.28.0`, `onnx 1.22.0`, **`transformers 5.14.1`**, numpy 2.4.6, 4 vCPU / 16 GB. 13 magyar tesztszöveg, 2–448 token:

| Metrika | Mért érték | Küszöb | |
|---------|-----------|--------|---|
| cosine **min** | **0,9999997616** | > 0,9999 | ✅ |
| cosine átlag | 0,9999999908 | — | ✅ |
| max abs elemenkénti eltérés | **1,788e-07** | < 1e-4 | ✅ |
| `cos > 0.9999` arány | **13 / 13** | 13/13 | ✅ |
| determinizmus (30 futás) | **1 / 30** különböző kimenet | 1 | ✅ |

**(b) Újramérés a projekt PINELT környezetében** — `torch 2.2.0+cpu`, `transformers 4.37.2`, `onnxruntime 1.28.0` (ez az, amit a `Dockerfile.prod` `onnx-export` stage ténylegesen telepít):

| Metrika | Mért érték |
|---------|-----------|
| cosine **min** | **≥ 0,9999991** |
| max abs elemenkénti eltérés | **1,5e-07** |
| token ID-k (`tokenizers` vs `AutoTokenizer`) | **azonos minden próbaszövegen** |

> Forrás: `backend/app/services/embedding_service.py` modul-docstring ("Inference backends" szekció) és `backend/requirements.prod.txt` fejléc-kommentje.
>
> **A két mérés nem mond ellent egymásnak** — más `transformers`/`torch` verzión futottak. A **(b)** az irányadó, mert az a szállított környezet. A különbség (0,9999997616 → 0,9999991) a nagyságrend szintjén jelentéktelen, és mindkettő nagyságrendekkel a 0,9999-es küszöb felett van.

### 3.3 Rangsor-ekvivalencia — *ez dönt, nem a nyers cosine*

40 dokumentumos magyar autós korpusz torch-csal indexelve, 6 query:

| Query encoder | top-5 sorrend azonos | top-1 azonos | átlag top-5 átfedés | max score-delta |
|---------------|:--------------------:|:------------:|:-------------------:|-----------------|
| **onnx-fp32** | **6 / 6** ✅ | 6 / 6 | **1,00** | **2,384e-07** |
| onnx-int8 (QInt8) | 0 / 6 ❌ | 4 / 6 | 0,47 | 4,372e-02 |
| onnx-uint8 (QUInt8) | 0 / 6 ❌ | 6 / 6 | 0,70 | 2,261e-02 |

> **Kontextus, ami ezt élessé teszi:** a mért korpuszban a torch top1–top2 margók **0,0004–0,0127** között vannak. Az fp32 eltérése (2,4e-07) ennek **1/1600 – 1/50 000-e** → biztonságos. Az int8 eltérése (4,4e-02) a margó **3–100×-osa** → garantáltan átrendezi a sorrendet.

### 3.4 Az int8 KIZÁRVA — két, egymástól független okból

1. **Pontatlanság:** `hubert_int8.onnx` = 110,83 MB (3,97× kisebb), de cos **min 0,9437 / átlag 0,9652**, max abs diff 5,02e-02, **0/13** a küszöb felett. Pl. *"kék füst jön a kipufogóból hidegindításkor"*: torch/fp32 helyesen a turbó-olajfogyasztást hozza (0,9737), az int8 a lambda-szondát (0,4899).
2. **Nemdeterminizmus (önmagában is kizáró ok):** alapértelmezett szálszámmal az int8 modell **ugyanarra a bemenetre futásonként MÁS eredményt ad.** Ugyanaz a session, 30 ismétlés: egy 21-tokenes szövegre 11 különböző kimenet, egy 448-tokenesre **29 különböző kimenet 30-ból**. Az fp32 ugyanezen a teszten **1/30** (teljesen determinisztikus).

> **Verdikt: az int8 nem "kicsit pontatlanabb" — futásonként más választ adna ugyanarra a kérdésre. NEM SZÁLLÍTHATÓ.** Ha valaki később mégis felveti, ez a szekció a válasz.

### 3.5 Latencia (medián 20 futásból, 3 warmup)

| Bemenet | Szálak | torch | onnx-fp32 |
|---------|--------|-------|-----------|
| 21 token (tipikus query) | 4 (default) | 30,3 ms | **17,6 ms** |
| 448 token | 4 (default) | **203 ms** | 274 ms |
| 21 token | 1 | 75,6 ms | **57,1 ms** |
| 448 token | 1 | **634 ms** | 755 ms |

> **Fontos aszimmetria:** az ONNX-fp32 **~1,7× gyorsabb rövid (query méretű) szövegre**, de 10–30%-kal lassabb hosszú inputon. A runtime query-út tehát nyer; a tömeges hosszú-dokumentum indexelés nem. Ez az AutoCognitix profiljához illik (a query rövid tünetszöveg, az indexelés offline, torch-csal fut).

**Cold load:** `AutoModel.from_pretrained` 0,35–0,62 s vs `ort.InferenceSession(fp32)` **0,87–0,99 s**. Mindkettő elhanyagolható a 100 s-os healthcheck timeouthoz képest.

### 3.6 Memória (peak RSS) — és a `transformers` csapda

| Folyamat | Peak RSS |
|----------|----------|
| torch + transformers + hubert | 802,5 MB |
| onnxruntime + fp32, **de `transformers` importálva** | **937,7 MB** ⚠️ |
| **onnxruntime + fp32, torch-mentes (`tokenizers` a `vocab.txt`-ből)** | **594,5 MB** |

> **KRITIKUS IMPLEMENTÁCIÓS FELTÉTEL:** a `transformers` import **behúzza a torchot** (+367 MB, még mielőtt bármi betöltődne). **Az ONNX csak akkor spórol memóriát, ha a `transformers` teljesen kimarad**, és a tokenizálás közvetlenül `tokenizers.BertWordPieceTokenizer(vocab.txt, lowercase=False)`-szal történik. Ha bent marad, az ONNX út **rosszabb**, mint a tiszta torch.
>
> A `lowercase=False` **kritikus** (`config.json`: `do_lower_case: false`). Ha valaki elrontja, az embedding csendben romlik — ezért van a befagyasztott fixture-ök között kötelezően magyar nagybetűs/ékezetes szöveg.

### 3.7 A mérés korlátai (őszintén)

- **Recall@k valódi relevancia-címkékkel: NEM MÉRVE** (nincs címkézett halmaz — lásd 8.4). Csak a torch-rangsorral való egyezés lett mérve, ami viszont **pontosan a migrációs kérdés helyes metrikája**.
- A teljes indexelt korpusz viselkedése **nem lett mérve** (40 dokumentumos proxy). Nagyobb korpusznál a margók szűkebbek → az fp32 (2,4e-07 delta) továbbra is bőven biztonságos.
- Batch-throughput és a default feletti ORT graph-optimalizációs szintek: **nem mérve**.
- A CUDA wheelek **kicsomagolt** mérete nem lett megmérve (~6–9 GB BECSLÉS).

---

## 4. A döntés és miért

**Választott megoldás: ONNX Runtime fp32, a `.onnx` az image-be sütve, `transformers` és `torch` NÉLKÜL.**

Az érvelés útja két megdőlt premisszán át vezetett — ezt szándékosan meghagyjuk, mert magyarázza, miért nem a "legegyszerűbb" opció nyert:

1. **"a torch 2 GB, ezért ki kell hagyni"** → **RÉSZBEN CÁFOLVA** (3.1). A 2,58 GiB a CUDA wheelre igaz; a `+cpu` wheel 178 MB, telepítve 894 MB.
2. **"akkor vigyük vissza a torchot, az a legegyszerűbb"** → **CÁFOLVA a numerikus méréssel** (3.2/3.3). Az ONNX-fp32 ekvivalenciája nem "valószínű", hanem **bizonyított** — ezzel az ONNX egyetlen valódi kockázata (hogy elrontja az embedding-teret) megszűnt, miközben minden előnye megmaradt.
3. **"az int8 még kisebb és gyorsabb lenne"** → **HATÁROZOTTAN CÁFOLVA**, két független okból (3.4).

**A döntés négy indoka, mind mért adaton:**

1. **Az embedding-tér kompatibilitás bizonyított.** A már indexelt vektorok **egyetlen bitjét sem kell újraszámolni.**
2. **~720 MB kisebb image, ~208 MB kisebb RSS workerenként** (802,5 → 594,5 MB).
3. **~1,7× gyorsabb a query-úton** (17,6 ms vs 30,3 ms egy tipikus tünetszövegre).
4. **A `torch` teljesen kikerül a prod image-ből** — vele a `sympy`/`networkx`/`mpmath` farok és a torch teljes autograd/JIT/distributed felülete, amiből a projekt semmit nem használ.

### 4.1 Amit NEM választottunk

| Opció | Miért nem |
|-------|-----------|
| **torch `+cpu` + transformers** (fallback jelölt) | Teljesen működőképes, csak drágább (image, RAM) és lassabb a query-úton. **Nem került szállításra**, de a `torch` backend-út **megmaradt a kódban** (`EMBEDDING_BACKEND=torch`) — a dev gép és az offline indexelő scriptek ezen futnak, és **az az embedding-tér horgonya.** |
| **int8/uint8 kvantálás** | 3.4 — rangsor megtörik + nemdeterminisztikus. |
| **Külön embedding microservice** | +1 Railway service, +RTT, új hibamód (timeout/502), két helyen driftelő modell-revízió. A mai terhelésre over-engineering. |
| **Hosted embedding API** (OpenAI / Cohere / Voyage / Jina) | **Nem csere, hanem adatmigrációs projekt**: más dimenzió → **teljes reindex**, plusz új futó költség és külső függés minden query-re. A pénzköltség elhanyagolható (~$0,27 BECSLÉS), az idő- és kockázatköltség nem (2–4 nap BECSLÉS). **Nincs olyan hosted provider, ami `hubert-base-cc`-t szolgálna ki.** |
| **HF Inference Endpoint** | Funkcionálisan a microservice, csak drágábban (~$45–90/hó BECSLÉS) és vendor lock-kal. |
| **Nincs query-embedding (lexikai only)** | A teljes indexelt Qdrant-befektetés halottá válik, és a magyar morfológia miatt a lexikai keresés gyengén működik ("rángat" / "rángatás" / "rángatva"). Félúton maradni — fizetni a Qdrantért, miközben nullvektort küldünk bele — volt a **kiinduló állapot**, és az a legrosszabb. |
| **Modellváltás (E5 / MiniLM)** | Külön projekt. A `hubert-base-cc` egy nyers MLM-BERT, nem sentence-transformer, tehát elméletileg egy retrieval-re finomhangolt modell jobb lehetne — de ez **mérendő hipotézis**, és nincs hozzá magyar kiértékelő halmaz (8.4). Backlog. |

---

## 5. Ami TÉNYLEGESEN implementálva lett — és hol

> Ez a fejezet az implementáció térképe. Az itt szereplő állítások mind ellenőrizhetők a hivatkozott fájlokban.

### 5.1 A modell-revízió pinelve van

`backend/app/core/config.py`:

```python
HUBERT_MODEL: str = "SZTAKI-HLT/hubert-base-cc"
HUBERT_REVISION: str = "028baac7feb87a7b2f042bbdaa5deec6513c6060"
```

A korábbi `"main"` **nincs többé**. Ugyanez a SHA a `Dockerfile.prod` **mindkét** stage-ének `ARG HUBERT_REVISION` default értéke (`onnx-export` és `production`), így az exportált gráf és a futó app **nem tud szétcsúszni** arról, melyik súlyok készítették az indexelt vektorokat.

### 5.2 Négy új beállítás (nem egy)

`backend/app/core/config.py`, az "Embedding backend selection" blokk:

| Beállítás | Default | Szerep |
|-----------|---------|--------|
| `EMBEDDING_BACKEND` | `"auto"` | `"auto"` \| `"onnx"` \| `"torch"` \| **`"disabled"`** |
| `HUBERT_ONNX_PATH` | `/app/models/hubert_fp32.onnx` | env-overridable, hogy egy rossz útvonal Railway-változó javítás legyen, ne redeploy |
| `HUBERT_VOCAB_PATH` | `/app/models/vocab.txt` | ugyanaz |
| `EMBEDDING_ORT_THREADS` | **`1`** | ORT intra-op szálak **session-önként** |

A `"disabled"` mód **teljes értékű** üzemmód, nem csak egy rollback-címke: `_select_backend_name()` `None`-t ad vissza rá, és minden embed hívás `EmbeddingUnavailableError`-t dob. **Egyetlen mód sem ad vissza nullvektort.**

Az `EMBEDDING_ORT_THREADS=1` default indoklása a szálszám **szorzat**, nem összeg:
`WEB_CONCURRENCY (2) × embedding pool slot (2) × EMBEDDING_ORT_THREADS (1) = 4` szál egy 2 vCPU-s Railway konténeren. A pool mérete `_thread_pool = ThreadPoolExecutor(max_workers=2)` az `embedding_service.py`-ban. Nagyobb plan-en az `EMBEDDING_ORT_THREADS` Railway-változóként emelhető.

A production image ezeket a `Dockerfile.prod` `production` stage `ENV` blokkjában rögzíti: `EMBEDDING_BACKEND=onnx`, `EMBEDDING_ORT_THREADS=1`, plusz a két útvonal.

### 5.3 A backend-implementáció INLINE, nincs `embedding_backends/` csomag

> **Eltérés a tervtől.** A terv egy `app/services/embedding_backends/` package-et és egy `scripts/export_hubert_onnx.py` scriptet ígért. **Egyik sem létezik.** Ne keresd őket.

Ténylegesen:

| Elem | Hol van |
|------|---------|
| ONNX backend | `_OnnxEmbeddingBackend` osztály **`backend/app/services/embedding_service.py`-ban**, inline |
| Backend-választás | `HungarianEmbeddingService._select_backend_name()` ugyanott |
| Közös pooling | `_mean_pool_l2_numpy()` modul-szintű függvény ugyanott — **ez a numpy-változat a torch és az ONNX út közös nevezője**, és ez tartja őket ugyanabban az embedding-térben |
| torch út | ugyanabban a fájlban maradt (`_load_hubert_model()`), dev + offline indexelés |
| ONNX export | **`backend/Dockerfile.prod`, `onnx-export` stage, egyetlen `RUN python -c` blokk** — nincs külön script |

Az `_OnnxEmbeddingBackend.__init__` beállításai: `intra_op_num_threads = max(1, threads)`, `inter_op_num_threads = 1`, `graph_optimization_level = ORT_ENABLE_ALL`, `providers=["CPUExecutionProvider"]`. A tokenizálás `BertWordPieceTokenizer(vocab_path, lowercase=False)` + `enable_truncation(512)` + `enable_padding()`. A `forward()` csak azokat az inputokat adja át, amiket az exportált gráf ténylegesen deklarál (`self._input_names` szűrő).

**A `transformers` sehol nem szerepel az ONNX úton** — ez a 3.6 mérés miatt kötelező feltétel, nem stílus.

### 5.4 A modell betöltése lazy maradt

Az ONNX session **double-checked lockinggal, az első embed hívásnál** épül fel (`_load_onnx_backend()`), **nem** importkor és **nem** a FastAPI lifespanben. A `warmup()` szándékosan nincs bekötve a bootba. A boot-hoz hozzáadott költség csak az `import onnxruntime` (~0,2–0,5 s) — két nagyságrenddel olcsóbb a kiváltott torch importnál. A boot továbbra is az `alembic upgrade head`-től dominált.

### 5.5 `EmbeddingUnavailableError` — más helyen és más ősosztállyal, mint a terv

> **Eltérés a tervtől.** A terv az `embedding_service.py`-ba tette volna, `RuntimeError` leszármazottként. Ténylegesen:

`backend/app/core/exceptions.py`:

```
AutoCognitixException
  └─ EmbeddingException            (ErrorCode.EMBEDDING_ERROR, HTTP 500)
       └─ EmbeddingUnavailableError  (status_code felülírva: HTTP 503)
```

Ez fontos gyakorlati különbség: az `EmbeddingUnavailableError` így beleilleszkedik a projekt **strukturált, magyar üzenetű** hibahierarchiájába (`ErrorCode`, `details`, `message`), és **503**-at hordoz, nem 500-at.

A szerződés a docstringben ki van mondva: **a hívók elkapják és a lexikai/gráf ágra degradálnak (ERROR szintű logolással), soha nem 500-aznak és soha nem helyettesítenek nullvektorral.**

### 5.6 Három rétegű "soha többé csendes nullvektor" védelem

| Réteg | Mi történik | Hol |
|-------|-------------|-----|
| **1. Fail loudly** | nincs backend → `EmbeddingUnavailableError`, nem `[0.0]*768` | `embedding_service.py` `_require_backend()` |
| **2. Kapu a keresés előtt** | üres vagy `norm < 1e-6` query vektor → `ValueError`, a keresés el sem indul | `qdrant_client.py` `_validate_query_vector()` + `MIN_QUERY_VECTOR_NORM` |
| **3. Kívülről látható próba** | `self_test()` egy fix magyar mondatot embedel és ellenőrzi, hogy `abs(norm − 1.0) < 1e-4` és `dim == 768` | `embedding_service.py` `self_test()` → `health.py` `check_embedding_health()` |

A 2. réteg szándékosan `ValueError`-t dob, nem `QdrantException`-t: nem a Qdranttal van baj, a hívó adott át érvénytelen vektort. Minden hívóoldal amúgy is széles `except Exception`-nel degradál a lexikai/gráf útra, tehát ez **hangosan logol anélkül, hogy endpointot 500-azna**.

**Üres szöveg:** az `embed_text()` üres bemenetre **továbbra is** `[0.0]*768`-at ad. Ez az **indexelési** oldalon védhető (üres rekord ne kapjon random vektort), a query oldalon pedig a 2. réteg fogja meg. A kódban ez explicit kommenttel van dokumentálva. *(A tervezett `allow_empty` paraméter nem került be — lásd 8.)*

### 5.7 Redis cache: verziózott névtér a manuális flush HELYETT

> **Eltérés a tervtől.** A terv **kötelező** operátori lépésként írta elő az `embed:*` kulcsok `SCAN`+`DEL`-jét deploy után. **Erre nincs szükség, és nem is szabad csinálni.**

Ténylegesen: az `embedding_service.py` egy verziózott névteret tesz a cache-kulcs elé:

```python
EMBEDDING_CACHE_VERSION = "v2"
# _cache_key_material(text) ->  f"{EMBEDDING_CACHE_VERSION}|{backend_name}|{text}"
```

A `redis_cache.py::_embedding_cache_key()` ezt sózza tovább `HUBERT_MODEL@HUBERT_REVISION`-nel és hasheli. Következmény:

- a régi build által beírt **mérgezett `[0.0]*768` bejegyzések egyszerűen elérhetetlenné válnak** abban a pillanatban, amikor az új image bebootol — nincs operátori lépés, és nincs egyórás TTL-ablak, amiben a javítás törötten néz ki;
- a **backend neve** is a kulcsban van, így egy torch- és egy ONNX-készítésű vektor soha nem szolgálható ki egymás helyett.

**Szabály a jövőre:** ha a produkált vektorok bármi miatt megváltozhatnak (backend-csere, modell/revízió bump, pooling-változás), **`EMBEDDING_CACHE_VERSION`-t kell bumpolni** — nem Redist takarítani.

### 5.8 R2 — a RAG collection-drift javítás LESZÁLLÍTVA

> **Eltérés a tervtől.** A terv R1/R2/R3-at függő feladatként sorolta fel. **Az R2 ugyanebben a PR-ben szállításra került** (`c64dcf1`).

`backend/app/services/rag_service.py::retrieve_from_qdrant()` kapott egy `type_` paramétert. Ha meg van adva, a keresés a `settings.QDRANT_UNIFIED_COLLECTION`-re megy a `QdrantService.search_unified()`-en keresztül, ami a `{"type": type_}` diszkriminátort **utoljára** injektálja (így egy hívó által átadott `type` sosem tudja felülírni). A `collection` paraméter csak explicit legacy lookupra maradt.

A két retrieval-láb ma:

| Láb | Route |
|-----|-------|
| DTC | `type_="dtc"` → unified `autocognitix` |
| "symptom" | **`type_="complaint"`** → unified `autocognitix` |

A symptom-ág tudatosan a `complaint` típusra képez: az `autocognitix` collectionben **nincs `symptom` payload-típus**, és az indexelt NHTSA panasz-narratívák **maguk a tünetleírások**. A `vehicle_make` exact-match filter **eltávolítva** — a payload kulcsa `make`, nyers all-caps NHTSA értékkel, tehát a régi filter garantáltan 0 találatot adott.

Két további részlet, ami könnyen elromlana:
- **A cache-kulcs tartalmazza a `type_`-ot** — a unified collection alatt két láb egyébként ugyanazon a collection+query+filter kulcson osztozna, és egymás eredményét szolgálná ki.
- **`model_version` NEM adható át** a unified úton: az `autocognitix` pontok nem hordoznak `_embedding_model_version` payloadot, tehát a szűrő mindent kizárna.

Telemetria minden retrievalnél (`rag qdrant retrieval: collection=%s type=%s hits=%d`), hogy egy jövőbeli drift **redeploy nélkül** diagnosztizálható legyen.

### 5.9 Amit a legacy collectionökről tudni kell

**A legacy per-típus collectionök NEM lettek megszüntetve.** A `QdrantService` konstansai (`DTC_COLLECTION = "dtc_embeddings_hu"` stb.) megvannak, és az `initialize_collections()` — amit az `app/main.py` lifespan hív — **továbbra is létrehozza mind az ötöt, üresen**, ha nem léteznek.

Következmény, amit egy operátor lát: a Qdrant Cloudon ott lesz öt üres `*_hu` collection **plusz** az `autocognitix`, amiben a vektorok vannak. Ez **nem hiba, de félrevezető** — lásd 8.

A `search_similar_symptoms()`, `search_components()`, `search_repairs()` metódusok **még mindig a legacy (üres) collectionökre mutatnak**, de a javítás óta **nincs hívójuk** az `app/`-ban és a `scripts/`-ben. Halott kód, ami újra használatba véve azonnal visszahozná a driftet.

### 5.10 requirements.prod.txt

`torch` és `transformers` **továbbra sincs** benne — most már indoklással. Helyettük:

```
onnxruntime==1.28.0
tokenizers==0.15.2
```

Mindkét pin **exact**. A `tokenizers` pin az egyetlen forrása az igazságnak: a `Dockerfile.prod` `onnx-export` stage-e **kigrepeli ezt a sort** és azt telepíti, hogy a build-időben bizonyított token-ID egyezés arról a tokenizerről szóljon, amit a production ténylegesen futtat. Ha a sor eltűnik, a build elhasad (`test -n "${TOKENIZERS_PIN}"`).

---

## 6. Hogyan van verifikálva

### 6.1 A build-time kapu — ez a legfontosabb

A `Dockerfile.prod` `onnx-export` stage-e **exportál ÉS ellenőriz egy lépésben**. Ez minden buildnél újra bizonyítja:

**(a) Tokenizer-egyezés.** A `BertWordPieceTokenizer(vocab.txt, lowercase=False)` **pontosan ugyanazokat a token ID-kat** adja, mint az `AutoTokenizer` — beleértve egy **512 tokennél hosszabb** próbaszöveget, azaz a truncation is egyezik. A build külön assertáli, hogy ez a próbaszöveg valóban 512 token felett van (`'the truncation probe is no longer over 512 tokens - it proves nothing'`) — különben a teszt észrevétlenül elveszítené a jelentését.

**(b) Gráf ↔ torch paritás.** Az ONNX gráf reprodukálja az ugyanabból a revízióból letöltött torch modell embeddingjét `cos > 0.9999`-re, **egyedi szövegeken ÉS egy vegyes hosszúságú batchen** (ez utóbbi az, ami a paddinget és a mask-súlyozott poolingot együtt teszteli).

**(c) Gráf ↔ BEFAGYASZTOTT referenciavektorok.** Ez az, amit **nem lehet meghamisítani.** A (b) az exportot egy *ugyanazon a revízión* újraletöltött torch modellhez méri — tehát egy **rossz revízió-pin tökéletesen átmegy rajta**, miközben olyan súlyokat szállít, amik soha nem készítették az indexelt vektorokat. A fixture viszont az **indexben ténylegesen benne lévő** vektorokhoz van horgonyozva, tehát egy rossz pin a **buildet** bukja el, nem csendben üríti ki a szemantikus keresést a production-ben.

A build ezen felül ellenőrzi, hogy a fixture `model` és `revision` mezője **egyezik-e a build pinjeivel**, és beszédes hibaüzenettel áll meg, ha nem:
`'frozen fixture was produced by %s@%s but this build pins %s@%s - regenerate the fixture AND reindex Qdrant, or fix the pin'`.

Opcionálisan `--build-arg HUBERT_ONNX_SHA256=...` átadható, ekkor egy váratlan export `sha256sum -c` hibán bukik el. Enélkül a build csak kiírja a digestet.

**Miért a build és nem a teszt?** Mert ugyanez az assertion `backend/tests/unit/test_embedding_backends.py::test_frozen_reference_vectors_match_backend`-ben is ott van — de `@pytest.mark.skipif`-fel, ami **minden környezetben skippel, ahol nincs `.onnx` fájl**. Azaz: CI-ben skippel, dev gépen skippel, mindenhol skippel. **Egy őr, ami soha nem fut le, nem őr.** A Docker build az egyetlen hely, ahol a `.onnx` létezik, ezért az ellenőrzésnek ott a helye.

### 6.2 Befagyasztott fixture-ök

`backend/tests/fixtures/`:

| Fájl | Tartalom | Méret |
|------|----------|-------|
| `hubert_reference_vectors.json` | **6** (text → 768-dim vektor) pár + `n_tokens`, plusz `model` / `revision` / `pooling` / `tokenizer` / `generated_with` metaadat | ~71 KB |
| `pooling_reference.json` | **3** befagyasztott pooling+L2 eset — a modellgráfon KÍVÜLI aritmetikát pinelí, ami a torch és az ONNX backendet ugyanabban a térben tartja | ~8 KB |
| `.gitignore` | `!*.json` / `!*.txt` negáció | — |

> A `.gitignore` **nem kozmetika**: a repo-gyökér `.gitignore` kizárja a `*.json`-t mint "nagy adatfájl", és e negáció nélkül a fixture-ök **sosem kerülnének commitba** — a teszt egy friss klónon elhasadna, a build-kapu pedig a legfontosabb assertionjét veszítené el.

A 6 szöveg lefedi: rövid tünet, DTC-formátumú kód, összetett mondat, **magyar nagybetűs + ékezetes** (`ÁRAMSZÜNET! … ŐRÜLT ŰRHAJÓ, öt szép szűzlány.` — ez a `lowercase=False` regressziós őre), egytokenes (`fék`), és egy hosszú panaszszöveg.

> **A fixture regenerálása CSAK teljes Qdrant reindex-szel együtt megengedett.** Ezt a fájl `_comment` mezője is kimondja.

### 6.3 Egység- és integrációs tesztek

| Fájl | Mit véd |
|------|---------|
| `backend/tests/unit/test_embedding_backends.py` | backend-választás minden `EMBEDDING_BACKEND` módra, ONNX encode/forward/pooling, a fixture-assertion (skippelt, lásd 6.1) |
| `backend/tests/unit/test_embedding_service.py` | a régi "nincs torch → nullvektor" tesztek **átírva** "→ raises `EmbeddingUnavailableError`"-ra |
| `backend/tests/api/test_embedding_degradation.py` | az endpointok degradálnak, nem 500-aznak, ha az embedding nem elérhető |
| `backend/tests/unit/test_qdrant_client.py` | a nullvektor-guard, és hogy a unified collection az alapértelmezett útvonal |
| `backend/tests/integration/test_service_rag.py` | a RAG a unified collectionre megy, `type` diszkriminátorral |

### 6.4 Health endpoint — figyelem a státusz-leképezésre

`GET /api/v1/health/detailed` (auth kell) → `services.Embedding`.

A belső `self_test()` **három** értéket adhat, a health endpoint pedig **átképezi** őket:

| `self_test()` belső státusz | `services.Embedding.status` a válaszban |
|---|---|
| `"ok"` | **`"healthy"`** |
| `"degraded"` | `"degraded"` |
| `"unavailable"` | **`"degraded"`** (nem `"unhealthy"`) |

> **Ne `status == "ok"`-ra ellenőrizz.** A helyes assertion: `services.Embedding.status == "healthy"`, és `services.Embedding.details.self_test_norm ≈ 1.0`.

Az `"unavailable" → "degraded"` leképezés szándékos: az API ilyenkor is kiszolgál (lexikai + gráf út), és `"unhealthy"`-ra állítva egy embedding-kiesés úgy nézne ki, mint egy teljes adattár-leállás.

Az embedding-próbának **saját, 5 s-os timeoutja** van (`EMBEDDING_HEALTH_TIMEOUT_SECONDS`), szigorúan a `detailed_health_check()` közös 10 s-os budgetje alatt. Enélkül egy lassú, hideg modellbetöltés kiütné a **közös** timeoutot, aminek a kezelője a postgres/neo4j/qdrant/redis **mindegyikét** `unhealthy`-nak jelentené — teljes adattár-kiesést jelentve akkor, amikor csak az embedding backend lassú.

---

## 7. Üzemeltetési runbook

### 7.1 Build

A `Dockerfile.prod` három stage-e:

| Stage | Szerep |
|-------|--------|
| `onnx-export` | **Eldobott.** Itt és csak itt él torch + transformers + onnx. Exportálja a gráfot és lefuttatja a 6.1 ellenőrzéseket. Semmi nem jut belőle a runtime image-be a `.onnx` és a `vocab.txt` kivételével. Amd64-hez kötött (a `torch==2.2.0+cpu` local-version wheel nincs aarch64-re publikálva). |
| `builder` | venv + `requirements.prod.txt` (torch/transformers **nélkül**). |
| `production` | A venv + az app + `COPY --from=onnx-export /models ./models` + az embedding `ENV` blokk. Non-root `appuser`. |

Az export után a build **törli** a `tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` fájlokat — a runtime csak a `.onnx`-et és a `vocab.txt`-t olvassa.

Opcionális build-arg: `--build-arg HUBERT_ONNX_SHA256=<digest>` a bit-pontos reprodukálhatóság kikényszerítésére.

### 7.2 Deploy / rollout

1. **Redis-takarítás NEM kell.** A verziózott cache-névtér (5.7) magától érvényteleníti a régi bejegyzéseket. Ha bármi miatt mégis kétséges: `EMBEDDING_CACHE_VERSION` bump a kódban — ne kézi `SCAN`+`DEL`.
2. Deploy alacsony forgalmú időablakban (nincs staging környezet).
3. Élő ellenőrzés (7.4) **kézzel, azonnal** — lásd a figyelmeztetést lent.

> **FIGYELEM — a CD nem véd meg:** a `cd.yml` smoke-tesztjei `continue-on-error: true`-val futnak, és teljesen kimaradnak, ha a `vars.API_URL` nincs beállítva. A "rollback" job pedig **csak egy GitHub issue-t nyit**. Azaz **egy törött deploy nem bukik el automatikusan, és nem áll vissza magától.**

### 7.3 RAM és worker-szám

Mért **594,5 MB peak RSS per worker** az ONNX úton (vs 802,5 MB torch-csal) → ~1,19 GB steady 2 workerrel, és **~2 GB tranziensen**, amíg mindkettő hidegen betölti a gráfot.

A worker-szám ezért **Railway-változó (`WEB_CONCURRENCY`)**, nem beégetett literál. Ennek konkrét oka van: a `railway.toml` `restartPolicyMaxRetries = 3`, tehát **három OOM-kill után a restartok végleg leállnak** — és abban a pillanatban a "rebuildeljük kevesebb workerrel" **nem** helyreállítási lehetőség. `WEB_CONCURRENCY=1` a dashboardon + restart az.

### 7.4 Élő ellenőrzés deploy után

1. **Health:** `GET /api/v1/health/detailed` → `services.Embedding.status == "healthy"`, `details.self_test_norm ≈ 1.0`, `details.backend == "onnx"`.
2. **Szemantikus DTC keresés** magyar tünettel (ne DTC-kód alakú, hogy a `is_code_query` ág ne vigye el):
   `GET /api/v1/dtc/search?query=rángat a motor gyorsításkor&use_semantic=true`
   → a logban `semantic dtc search: collection=autocognitix type=dtc hits=N pg_matched=M`, ahol **`hits` > 0**.
3. **Teljes diagnózis:** `POST /api/v1/diagnosis/analyze` valós tünetszöveggel → a logban `rag qdrant retrieval: collection=autocognitix type=dtc hits=N` **és** `type=complaint hits=N`; a válasz `sources` / `similar_complaints` mezői nem üresek, `used_fallback == false`.
4. **Negatív kontroll:** értelmetlen query (pl. `"qqqq zzzz"`) → kevés vagy nulla találat a `score_threshold` felett. Ha erre is tele van a lista, a szűréssel van baj.
5. **Log-alapú:** deploy után 24 óra alatt **nulla** `EmbeddingUnavailableError` és nulla `Rejected degenerate query vector` üzenet.

### 7.5 Rollback

| Kiváltó | Lépés | Idő |
|---------|-------|-----|
| OOM / restart loop | `WEB_CONCURRENCY=1` Railway-változó + restart; ha 3 retry már elfogyott, Railway rollback az előző deploymentre | ~1–2 perc |
| Az embedding rossz eredményt ad, de a service él | `EMBEDDING_BACKEND=disabled` env → a szemantikus ág **explicit hibát dob** (a hívók degradálnak), a lexikai + Neo4j út tovább szolgál | ~30 s |
| Boot lassulás | `WEB_CONCURRENCY=1` | ~30 s |

**Kulcs:** az `EMBEDDING_BACKEND=disabled` út **NEM** a régi csendes nullvektor. A rendszer ilyenkor is *tudja és logolja*, hogy degradált, és a health endpoint is mutatja. Ez a különbség.

### 7.6 Ha a modell-revíziót valaha bumpolni kell

Ez **nem** egy config-változtatás, hanem egy migráció. A sorrend kötelező:

1. `HUBERT_REVISION` frissítése `config.py`-ban **és** a `Dockerfile.prod` **mindkét** `ARG` default értékében.
2. `backend/tests/fixtures/hubert_reference_vectors.json` regenerálása az új revízióval (`model` + `revision` mezőkkel együtt).
3. `EMBEDDING_CACHE_VERSION` bump.
4. **Teljes Qdrant reindex** az `autocognitix` collectionbe.

Ha a 2. lépés kimarad, a **build fog elhasadni** — pontosan úgy, ahogy kell.

---

## 8. Nyitott kérdések és follow-upok

> Ez a szekció **csak azt tartalmazza, ami ténylegesen nyitva van.** Amit a `050965e` / `c64dcf1` leszállított, az az 5–6. fejezetben van.

### 8.1 Az R3 safety-guard csomag részben szállított

| Elem | Állapot |
|------|---------|
| `EmbeddingUnavailableError` nullvektor helyett | ✅ szállítva (5.5) |
| Qdrant query-vektor norm guard | ✅ szállítva (5.6) |
| `/health/detailed` embedding-próba | ✅ szállítva (6.4) |
| **Boot-time embedding önteszt** az `app/main.py` lifespanben | ❌ **nincs** — a lifespan csak a thread poolok leállítását kezeli. A modell szándékosan lazy (5.4), így egy induláskori próba egy nemkívánt cold loadot kényszerítene. Ha kell, külön, nem-blokkoló taskként. |
| **`semantic_search_available` / `degraded_reason`** a `/diagnosis/analyze` válaszában | ❌ **nincs** — a degradáció ma csak logban és a health endpointon látszik, a **válaszban nem**. A felhasználó továbbra sem tudja megkülönböztetni a "nincs szemantikus kontextus"-t a "nincs releváns találat"-tól. **Ez a legértékesebb megmaradt follow-up.** |
| `embed_text(..., allow_empty=...)` | ❌ **nincs** — üres szöveg továbbra is `[0.0]*768`, a query oldalon a Qdrant-guard fogja meg (5.6). |

### 8.2 A legacy collection-modell takarítása

Ma egyszerre igaz:
- az `initialize_collections()` **létrehoz öt üres `*_hu` collectiont** minden app-induláskor (5.9);
- a `search_similar_symptoms()` / `search_components()` / `search_repairs()` metódusok **ezekre az üres collectionökre mutatnak**, és **nincs hívójuk**;
- a `delete_by_user()` (GDPR) **szintén csak az öt legacy collectionből** töröl, az `autocognitix`-ból nem — ma ez nem hiba, mert a unified collection nem tárol `user_id`-t, de **ez egy nem dokumentált feltevés**, ami egy jövőbeli felhasználói vektor-indexeléssel csendben elromlana;
- a `get_storage_stats()` / `check_storage_alerts()` **csak a legacy collectionöket** nézi, azaz **a valódi vektortároló méretét nem monitorozza**.

**Javasolt:** vagy vezessük ki a legacy konstansokat és az auto-create-et, vagy vegyük fel az `autocognitix`-ot a stats/alert/GDPR listákba. A jelenlegi félút bármikor újra drift forrása lehet.

### 8.3 A `type="symptom"` hiánya

Az `autocognitix` collectionben nincs `symptom` payload-típus (1.2). A symptom-ág ma `complaint`-re képez (5.8), ami **védhető proxy**, de nem ugyanaz. Ha valaha lesz kurált magyar tünet-korpusz, azt külön kell indexelni és a leképezést újragondolni.

### 8.4 Nincs magyar retrieval kiértékelő halmaz

Nincs 50–100 elemű `magyar query → várt DTC` halmaz. Enélkül:
- semmilyen **retrieval-minőség** állítás nem bizonyítható (csak backend-ekvivalencia, ami más kérdés);
- a modellváltás (4.1 utolsó sora) nem értékelhető felelősen.

### 8.5 Az `_embedding_model_version` payload aszimmetria

Az `upsert_vectors()` minden payloadba beleírja az `_embedding_model_version`-t, de az `autocognitix` collection pontjai (más indexelő úton készültek) **nem hordozzák**. Ezért tilos a `model_version` szűrő a unified úton (5.8). Ez ma helyesen kezelt, de **implicit** — érdemes lenne a reindexnél egységesíteni.

### 8.6 Nem tisztázott operatív adatok

| # | Kérdés | Miért számít |
|---|--------|--------------|
| 8.6.1 | **Railway backend service RAM limit / futó plan** | Sehol nincs dokumentálva a repóban. A `WEB_CONCURRENCY` helyes értékéhez kell (7.3). Egy pillantás a Railway Metrics fülre megválaszolja. |
| 8.6.2 | **Hány pont van ténylegesen az `autocognitix` collectionben?** | A repó ellentmond magának — lásd `CLAUDE.md` "Aktuális Adatbázis Állapot" tábláját, ami a forrásokat és az ellentmondást is felsorolja. Egyetlen `GET /collections` a Qdrant Cloudon lezárná. |
| 8.6.3 | **A CI által épített image-et a Railway nem használja** | A `cd.yml` felépíti a `Dockerfile.prod`-ot és GHCR-be tolja, de a deploy `railway up`-ot futtat, ami a **Railway oldalán újraépít**. A build ideje így duplán fizetendő. Az `onnx-export` stage ezt még hangsúlyosabbá teszi. |
| 8.6.4 | **`docs/RAILWAY_DEPLOYMENT.md` reindex-lépése** | `railway run python scripts/index_qdrant.py` — a script torchot igényel, ami a prod image-ben **nincs és nem is lesz**. Ez a lépés ma megtévesztő: az indexelés **dev gépen / offline** futtatandó. |

---

## 9. Egy bekezdésben

A szemantikus keresés azért volt néma, mert a prod image-ből kihagyott torch helyett a kód **csendben nullvektort adott**, a RAG pedig ráadásul **üres collectionöket kérdezett**. A megoldás az **ONNX Runtime fp32**, ami mérésileg bit-közeli azonos embeddinget ad (a pinelt környezetben `cos ≥ 0,9999991`, rangsor 6/6), így az indexelt vektorok érintetlenek maradtak, miközben az image ~720 MB-tal, a memória ~208 MB/worker-rel kisebb és a query ~1,7× gyorsabb. A gráfot egy eldobott Docker build-stage exportálja a **SHA-ra pinelt** revízióból, és minden build újra bebizonyítja, hogy a gráf reprodukálja a repóban befagyasztott referenciavektorokat. A legfontosabb rész azonban **nem** az embedding visszahozása, hanem hogy egy hiányzó backend ma **hangosan hasít**, egy nullvektor pedig el sem jut a keresésig.
