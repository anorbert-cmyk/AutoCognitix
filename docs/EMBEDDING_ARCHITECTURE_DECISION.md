# Embedding Architektúra Döntés — HuBERT a production runtime-ban

> **Státusz:** DÖNTÉSRE VÁR
> **Készült:** 2026-07-25
> **Hatókör:** `backend/app/services/embedding_service.py`, `backend/Dockerfile.prod`, `backend/requirements.prod.txt`, Qdrant `autocognitix` collection
> **Döntéshozó:** projekt tulajdonos

---

## 1. Probléma

A magyar nyelvű szemantikus keresés a production környezetben **nem működik**. A `/api/v1/dtc/search?use_semantic=true` szemantikus ága és a zászlóshajó `/api/v1/diagnosis/analyze` RAG pipeline-ja **nulla Qdrant kontextust** kap.

A hiba **csendes** — nem dob hibát, nem logol errort, nem jelenik meg a válaszban. Ez az oka, hogy hónapokig rejtve maradt.

### 1.1 A hibalánc (megerősítve)

```
requirements.prod.txt (torch/transformers kikommentelve)
   └─> Dockerfile.prod build: torch NINCS az image-ben
        └─> embedding_service.py:34-38  ImportError -> TORCH_AVAILABLE = False
             └─> embed_text():362-364   return [0.0] * 768      # NULLVEKTOR
                  └─> Qdrant cosine search nullvektorral
                       ├─> rag_service.py:470 score_threshold=0.5 -> GARANTÁLTAN []
                       └─> dtc_codes.py:481 threshold nélkül -> értelmetlen, score=0 találatok
```

**Kritikus megfigyelés:** a nullvektor nem "rossz" találatot ad, hanem **matematikailag értelmetlen** eredményt. Cosine távolságnál a Qdrant a query vektort normalizálja; egy nulla normájú vektor normalizálása nulla vektort ad, így minden dot product 0.0. A `score_threshold=0.5` szűrő ezt garantáltan üres listává alakítja.

### 1.2 MÁSODIK, FÜGGETLEN ROOT CAUSE (új felfedezés)

A torch hiánya **nem az egyetlen** ok. A `/diagnosis/analyze` RAG pipeline **rossz Qdrant collection-öket** kérdez:

| Hely | Collection | Tartalom |
|------|-----------|----------|
| `rag_service.py:794` | `QdrantService.DTC_COLLECTION` = `dtc_embeddings_hu` | régi/külön collection |
| `rag_service.py:801` | `QdrantService.SYMPTOM_COLLECTION` = `symptom_embeddings_hu` | régi/külön collection |
| `dtc_codes.py:481` → `search_dtc()` → `search_unified()` | `settings.QDRANT_UNIFIED_COLLECTION` = **`autocognitix`** | **a ~54k HuBERT vektor itt van** |

A saját kódbázis dokumentálja is a driftet — `qdrant_client.py:309-312`:

> "The huBERT DTC vectors live in the unified `autocognitix` collection with a `{"type": "dtc", "code", ...}` payload — **NOT in the (empty) `dtc_embeddings_hu` collection**"

**Következmény: még ha a torch-ot vissza is tesszük a prod image-be, a `/diagnosis/analyze` RAG-ja továbbra is 0 Qdrant kontextust kapna**, mert nem a feltöltött collection-t kérdezi. Ezt a javítást kötelezően **együtt** kell szállítani az embedding-javítással, különben a fő fájdalompont nem szűnik meg.

Ráadásul a `rag_service.py:803` `vehicle_make` exact-match filtert tesz a symptom collection-re, amit a korábbi audit már HIGH-ként azonosított mint "MINDIG 0 találat" (`tasks/code_review_fixes_2c28099.md:42-47`).

És egy harmadik réteg: az `autocognitix` collection **csak `dtc` / `complaint` / `recall` payload-típust tartalmaz** (`scripts/index_qdrant_hubert.py:210, :313, :419`) — **`symptom` típus nem létezik benne.** A "symptom search" ág tehát nem egyszerűen rossz collectiont kérdez; a keresett entitástípus **egyáltalán nincs indexelve**. Ezt a leképezést az R2-ben tudatosan meg kell hozni (részletek: 10. fejezet, 3. pont).

---

## 2. Kényszerek — bizonyítékkal (file:line)

### 2.1 A prod image nem tartalmaz ML stacket

`backend/requirements.prod.txt:46-51`:
```
# ============================================
# EXCLUDED for production (local embedding only):
# - torch==2.2.0 (~2GB)
# - transformers==4.37.2
# - sentence-transformers==2.3.1
# ============================================
```

`backend/Dockerfile.prod:25-31` — a builder stage kizárólag a `requirements.prod.txt`-t telepíti:
```
# Use pre-built requirements.prod.txt (without torch/ML libraries)
# Embeddings are pre-computed and stored in Qdrant Cloud
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt
```

### 2.2 Diff: `requirements.txt` vs `requirements.prod.txt`

Amit a prod **nélkülöz** (a dev requirements-hez képest), és ami az embeddinghez lényeges:

| Csomag | dev (`requirements.txt`) | prod | Kell az embeddinghez? |
|--------|--------------------------|------|------------------------|
| `torch` | `==2.2.0` (:47) | **hiányzik** | IGEN (A opció) |
| `transformers` | `==4.37.2` (:46) | **hiányzik** | IGEN (tokenizer mindenképp) |
| `sentence-transformers` | `==2.3.1` (:45) | **hiányzik** | NEM — a kód nem használja, saját mean-pooling van |
| `pandas` | `==2.2.0` (:52) | hiányzik | nem |
| `beautifulsoup4`, `lxml`, `playwright` | :93-95 | hiányzik | nem |
| `pytest`, `black`, `mypy`, stb. | :71-81 | hiányzik | nem |
| `huspacy` / `spacy` | **kikommentelve** (:48-49) | hiányzik | lásd 2.4 |
| `numpy` | `==1.26.4` | `==1.26.4` (:54) | megvan |
| `qdrant-client` | `>=1.7.0` | `>=1.7.0` (:25) | megvan |

**Fontos:** a `sentence-transformers` **nem szükséges**. A projekt saját mean-poolingot implementál (`embedding_service.py:291-312`), így csak `transformers` (tokenizer + modell) vagy annak helyettesítője kell.

### 2.3 A runtime embedding spec — amit bármely helyettesítőnek BIT-SZINTEN reprodukálnia kell

`backend/app/services/embedding_service.py`:

| Lépés | Sor | Pontos viselkedés |
|-------|-----|-------------------|
| Modell | `settings.HUBERT_MODEL` = `SZTAKI-HLT/hubert-base-cc` (`config.py:177`) | BERT-base architektúra, ~110M paraméter |
| Revízió | `settings.HUBERT_REVISION` = `"main"` (`config.py:181`) | **NINCS commit SHA-ra pinelve** — lásd 2.7 |
| Tokenizer | `:226-230` | `AutoTokenizer.from_pretrained(MODEL, revision=REV, use_fast=True)` |
| Modell betöltés | `:233-241` | `AutoModel.from_pretrained(...)`, `torch_dtype=float32` (CPU-n; FP16 csak CUDA-n, `:120`), `.eval()` |
| Tokenizálás | `:380-382` | `padding=True, truncation=True, max_length=512, return_tensors="pt"` |
| Forward | `:388-389` | `torch.no_grad()` |
| **Pooling** | `:291-312` | attention-mask súlyozott **mean pooling** a `last_hidden_state`-en: `sum(emb * mask) / clamp(mask.sum(), min=1e-9)` |
| **Normalizálás** | `:395` | `torch.nn.functional.normalize(embedding, p=2, dim=1)` → **L2, egységhosszú** |
| Kimenet | `:398` | `embedding.squeeze().cpu().tolist()` → `List[float]`, 768 dim |
| Üres szöveg | `:371-373` | `[0.0] * 768` (jogos, de lásd 5. — a similarity search-be így sem szabadna bejutnia) |
| **Nincs torch** | `:362-364` | `[0.0] * 768` ← **A HIBA** |
| Batch nincs torch | `:425-427` | `[[0.0] * 768 for _ in texts]` ← ugyanaz batch úton |

**Ez a döntés szempontjából a legfontosabb tény: a pooling és az L2-normalizálás a modellen KÍVÜL, tiszta tensor-aritmetikával történik.** Bármely inference backend (ONNX Runtime, külön szolgáltatás), ami ugyanazt a `last_hidden_state`-et adja vissza, ugyanezt a poolingot numpy-ban elvégezve **azonos embedding-térben** marad. Ez teszi a B opciót reálissá.

### 2.4 Preprocessing (`preprocess_hungarian`) — jelenleg NO-OP

`embedding_service.py:314-349`. HuSpaCy-vel lemmatizál és stopword/punctuation-t szűr. **DE:**

- `spacy` nincs sem a `requirements.txt`-ben (`:48-49` kikommentelve), sem a `requirements.prod.txt`-ben
- `embedding_service.py:45-51` → `SPACY_AVAILABLE = False`
- `:330-332` → `return text.strip()`, azaz **azonosság-függvény**

Ez **szerencsés véletlen**: az indexelő script `preprocess=False`-szal futott (`scripts/index_qdrant_hubert.py:134-138`), a runtime viszont `preprocess=True`-val hív (`rag_service.py:499` a `retrieve_from_qdrant` `preprocess=True` default-ján keresztül, `dtc_codes.py:478` explicit `preprocess=True`). Mivel a preprocess mindkét oldalon no-op, jelenleg **nincs** query/index eltérés.

> **FIGYELMEZTETÉS:** ha valaki később hozzáadja a `spacy`+`hu_core_news_lg`-t a prod image-hez, a query szövegek lemmatizálódnak, az indexelt vektorok viszont nyers szövegből készültek → **azonnali, csendes minőségromlás**. A `spacy` hozzáadása tehát egy külön, tudatos döntés, ami **teljes reindexet igényel**. A jelen javítással NE kerüljön be.

### 2.5 A tárolt vektorok előállítása — egyezik-e a runtime úttal?

`scripts/index_qdrant_hubert.py`:
- `:126-128` — **ugyanazt a `get_embedding_service()`-t** használja, ugyanazt az osztályt
- `:132-138` — `embed_batch(texts, preprocess=False, use_cache=False)`
- `:117` — collection: `VectorParams(size=768, distance=Distance.COSINE)`, név `autocognitix` (`:45`)

**VERDIKT: a pooling és normalizálás azonos** (ugyanaz a kódút, `_embed_batch_internal` → `_mean_pooling` → `normalize` `:559-562`). Két eltérés viszont van, amit ki kell mondani:

1. **Eltérő batch-padding.** Batch módban `padding=True` a batch leghosszabb szekvenciájára paddel. A mask-súlyozott mean pooling ezt matematikailag semlegesíti — a padding tokenek 0 maszkkal kiesnek. Elméletileg a BERT self-attention a padding tokeneket az attention maskkal kizárja, tehát a nem-padding tokenek reprezentációja változatlan. **Gyakorlatban float-szintű eltérés (~1e-6) lehet.** Elhanyagolható.
2. **Szöveg-truncation az indexelésnél:** `texts.append(text[:8000])` (`:286`, `:391`) — karakter szintű vágás, majd a tokenizer 512 tokenre vág. A query oldalon nincs 8000-es vágás, csak az 512 token limit. Ez **nem** embedding-tér probléma, csak azt jelenti, hogy hosszú panaszoknak az eleje van indexelve.
3. **A modell nincs commit SHA-ra pinelve** (`HUBERT_REVISION="main"`). Ha a HF-en a `main` időközben mozdult, a mostani letöltés más súlyokat adhat, mint amivel a 54k vektor készült. Ezt a korábbi audit is jelezte (`tasks/audit_sprint13_master.md:43`, `tasks/wave2_refix_ops.md:28`). **A 6. fejezet verifikációs eljárása ezt is leteszteli.**

### 2.6 Railway / deployment kényszerek

| Tény | Forrás | Érték |
|------|--------|-------|
| Builder | `backend/railway.toml:1-3` | `DOCKERFILE`, `Dockerfile.prod` |
| Healthcheck path | `backend/railway.toml:7` | `/health` |
| Healthcheck timeout | `backend/railway.toml:8` | **100 s** |
| Restart policy | `backend/railway.toml:9-10` | `ON_FAILURE`, max 3 retry |
| Boot parancs | `Dockerfile.prod:93-105` | `alembic upgrade head && exec gunicorn ... --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120` |
| Base image | `Dockerfile.prod:9, 36` | `python:3.11-slim-bookworm`, multi-stage (builder + production) |
| `/health` költsége | `main.py:404-412` | statikus JSON, **nem** érint adatbázist, nem érint embeddinget |
| `warmup()` a boot-on | `main.py:109-148` (lifespan) | **NINCS meghívva** — a modell lazy, az első kérésnél töltődik |

**Fontos következtetés:** mivel a `warmup()` nincs a lifespan-ben és a `/health` triviális, a modell betöltése **nem** a healthcheck kritikus útján van. A boot-időhöz csak az `import torch` (modulszintű, `embedding_service.py:30`, amit a `services/__init__.py:19` behúz) adódik hozzá — ez CPU-n tipikusan pár másodperc, bőven belefér a 100 s-be.

**Ugyanakkor 2 gunicorn worker fut** (`Dockerfile.prod:96`). Ha mindkét worker betölti a modellt, a RAM-igény **duplázódik** (2 × ~440 MB súly + runtime). Ez a RAM-tervezés kulcsa.

#### Amit a repó NEM dokumentál (és ezért nem szabad feltételezni)

| Hiányzó adat | Következmény |
|--------------|--------------|
| **Railway backend service RAM limit / futó plan** | Ez a nyitott 9.1 kérdés. `docs/RAILWAY_DEPLOYMENT.md:247-257` csak annyit mond: Free Tier "$5 kredit/hó", Hobby "$5/hó (500 óra)". A `docs/DEPLOYMENT.md:227-231` 2G limit / 1G reservation értékei **docker-compose limitek, nem Railway-limitek.** |
| **Railway image size limit** | Sehol. Az egyetlen méret-szám a repóban a `requirements.prod.txt:49` "~2GB" megjegyzése. |
| **Railway build timeout / mért build idő** | Sehol. |
| **`HF_HOME` / `TRANSFORMERS_CACHE`** | **Nulla előfordulás repó-szerte.** Nincs build-time model prefetch, nincs cache-dir pinelés → ha az A/B opció nem süti be a modellt az image-be, futásidőben tölt a HuggingFace-ről. |
| Bármilyen embedding feature flag (`ENABLE_EMBEDDINGS` stb.) | Nincs. Az elérhetőséget kizárólag a `TORCH_AVAILABLE` import-próba dönti el, implicit módon. |

#### Két deployment-ellentmondás, amit a javítással együtt rendezni kell

1. **A CI által épített image-et a Railway nem használja.** `.github/workflows/cd.yml:107-121` felépíti a `Dockerfile.prod`-ot `linux/amd64,linux/arm64`-re és feltölti a GHCR-be — de a `cd.yml:208-224` deploy lépés csak `railway up`-ot futtat, ami a **Railway oldalán újraépíti** a forrásból. A GHCR image tehát halott súly, **de a build ideje duplán fizetendő** — a torch/ONNX beemelése így két helyen növeli a build-időt, ráadásul a **multi-arch (arm64) build a torch-nál lényegesen drágább**. Javasolt: vagy a GHCR build kivezetése, vagy a platformlista `linux/amd64`-re szűkítése.
2. **A `docs/RAILWAY_DEPLOYMENT.md:204-210` egy nem működő lépést dokumentál:** `railway run python scripts/index_qdrant.py` — a script `AutoTokenizer/AutoModel.from_pretrained`-et hív, de a Railway image-ben nincs torch. Ez a doksi ma megtévesztő; az A/B opcióval viszont **működővé válna**.

### 2.7 Redis embedding cache — rollout-kockázat

`backend/app/db/redis_cache.py:509-527`:
- kulcs: `sha256(f"{HUBERT_MODEL}@{HUBERT_REVISION}|{text}")`, prefix `embed:`
- TTL: `CacheTTL.EMBEDDINGS = 3600` (1 óra)

Két következmény:
1. **A nullvektorok jelenleg cache-elődnek.** `embed_text_async` (`:706-714`) feltétel nélkül elmenti, amit az `embed_text` visszaad — beleértve a `[0.0]*768`-at. Deploy után a mérgezett kulcsok **max. 1 órán át** élhetnek. → a rollout-ban **flush kell** (7.4).
2. **A cache kulcs nem tartalmazza a `preprocess` flaget.** Ma nem okoz gondot (2.4 — a preprocess no-op), de a `spacy` bevezetése azonnali cache-collisiont okozna. Külön TODO.

---

## 3. Opciók összehasonlítása

> A méret- és latencia-számok forrása és mérési módja a 4. fejezetben opciónként szerepel. Ahol nem sikerült ténylegesen megmérni, ott **BECSLÉS** jelölés van.

| # | Opció | Image (runtime + modell) | Peak RSS / worker | Query latencia (21 tok) | Költség/hó | Embedding-tér kompat. | Komplexitás |
|---|-------|--------------------------|-------------------|-------------------------|-----------|------------------------|-------------|
| A0 | torch==2.2.0 **PyPI default** (CUDA) | ~2,6 GB wheel / ~6–9 GB telepítve (BECSLÉS) | – | – | $0 | 100% | **KIZÁRVA (méret)** |
| **A** | **torch `+cpu`** + transformers | **1 339 MB** (894 + 445) | **802,5 MB** | 30,3 ms | $0 extra | **100% (referencia)** | **Alacsony** |
| **B** ⭐ | **ONNX Runtime fp32**, transformers nélkül | **616 MB** (176 + 440) | **594,5 MB** | **17,6 ms** | $0 extra | **BIZONYÍTOTT: cos ≥ 0,9999997, rangsor 6/6** | Közepes (build-lépés) |
| ~~B2~~ | ~~ONNX int8/uint8~~ | 176 + 111 MB | 228,5 MB | 6,8 ms | $0 extra | **MEGBUKOTT — rangsor 0/6, ÉS nemdeterminisztikus** | **KIZÁRVA** |
| **C** | Külön embedding microservice (torch) | 0 (backend nem nő) | 0 a backendben | +RTT | +$5–20 (BECSLÉS) | 100% | **Magas** |
| **D** | Hosted embedding API (OpenAI/Cohere/Jina/Voyage) | ~0 | ~0 | +100–400 ms (BECSLÉS) | ~$1 reembed + query-díj | **0% — teljes reindex kell** | Közepes + adatmigráció |
| D2 | HF Inference Endpoint hubert-base-cc-vel | ~0 | ~0 | +50–300 ms (BECSLÉS) | ~$45–90 (BECSLÉS) | 100% | Közepes, vendor lock |
| **E** | Nincs query-embedding (lexikai + kurált mapping) | 0 | 0 | ~0 | $0 | n/a — a 54k vektor **halott** | Alacsony, de funkcióvesztés |
| F | Kisebb többnyelvű modell (E5-small / MiniLM) | kisebb | kisebb | gyorsabb | $0 | **0% — teljes reindex** | Magas (reindex + minőségvizsgálat) |

> Minden szám **ténylegesen mérve** (PyPI JSON API, `download.pytorch.org/whl/cpu`, illetve egy izolált venv-ben lefuttatott export + összehasonlítás, 2026-07-25). Részletes bontás és a mérés korlátai: **4.7**.

---

## 4. Opciók mélyelemzése

### 4.1 A opció — torch + transformers a prod image-ben

**Hogyan működik:** a `requirements.prod.txt`-be visszakerül a `torch` és a `transformers`; a `TORCH_AVAILABLE` igaz lesz; a meglévő kódút változatlanul fut.

**Embedding-tér kompatibilitás: 100%.** Ez a referencia implementáció — pontosan ugyanaz a kód készítette a 54k vektort.

**A méret kérdése — itt van a döntés súlypontja, és itt derül ki, hogy a projekt egy félreértésre épült.**

A `requirements.prod.txt:49` "~2GB" megjegyzése **igaz — de csak a default PyPI wheelre**, ami `platform_system=="Linux" and platform_machine=="x86_64"` esetén **feltétel nélkül** behúzza a teljes CUDA stacket. Mért adatok (4.7):

- `torch==2.2.0` PyPI default + 11 db `nvidia-*` + `triton` = **2 640 MB wheel** (≈2,58 GiB). A megjegyzés tehát nemcsak igaz, hanem konzervatív.
- `torch==2.2.0+cpu` a `https://download.pytorch.org/whl/cpu` indexről = **178 MB wheel / 648 MB kicsomagolva**, és **nulla `nvidia-*` / `triton` függőség**.

**A különbség 2,40 GiB. Ugyanaz a torch, ugyanaz a numerika, ugyanaz az embedding-tér.** A `+cpu` variáns egyetlen dolgot nem tud: GPU-n futni — amire a Railway CPU-only konténerében amúgy sincs szükség (`_detect_device()` `embedding_service.py:151` amúgy is `cpu`-ra esne).

**Ez azt jelenti, hogy az eredeti kizárási döntés — bármilyen ésszerű is volt a 2,6 GB-os számmal — a helyes wheel-index ismeretében nem áll meg.**

Ehhez jön még:
- `transformers==4.37.2` + a ténylegesen ÚJ függőségei (tokenizers, safetensors, huggingface-hub, regex, filelock — a pyyaml/requests/packaging/numpy/tqdm már bent van) = **13,7 MB**. Elhanyagolható.
- `sentence-transformers` **nem kell** (0,13 MB önmagában, de ~56 MB scipy/sklearn/Pillow farkat és egy `torch>=1.11` hard requirementet hoz). **Ne kerüljön vissza.**
- A modell súlyai: ~110M paraméter fp32 ≈ **440 MB** (a repó dokumentációja ~500 MB-ot mond: `docs/INSTALLATION.md:24`, `docs/ONBOARDING.md:157`). Ez **minden** opciónál felmerül (A, B, C, D2), kivéve D-t és E-t.

**Reális image-delta az A opcióra: ≈ 648 MB (torch) + 14 MB (transformers) + ~440 MB (modell az image-be sütve) ≈ 1,1 GB.**

**Kód-változás:** minimális.
- `requirements.prod.txt`: `torch==2.2.0+cpu` (vagy újabb) + `transformers==4.37.2`
- `Dockerfile.prod`: extra index URL a CPU wheelhez (`--index-url https://download.pytorch.org/whl/cpu`), + a modell **build-időben** letöltése az image-be (különben az első query letölti a ~500 MB súlyt futásidőben)
- `embedding_service.py`: a nullvektor-fallback cseréje (5. fejezet) — ez minden opciónál kell

**Kockázatok / hibamódok:**
- RAM: 2 gunicorn worker × (torch runtime + 110M param fp32 ≈ 440 MB súly + aktivációk). Ha a Railway plan RAM-ja szűk, OOM-kill → restart loop (max 3 retry, `railway.toml:10`).
  - **Mitigáció:** `--workers 1` + több uvicorn thread, VAGY `--preload` gunicorn flag (fork előtt tölt, copy-on-write-tal osztozik a súlyokon). Utóbbi a `torch` lazy-loadja miatt csak akkor segít, ha a `warmup()`-ot a preload fázisban hívjuk.
- Build idő: a nagy wheel letöltése + telepítése minden deploynál (Railway build cache-elhet, de nem garantált).
- HF modell letöltés futásidőben, ha nem sütjük be az image-be → első query 10–60 s, plusz kimenő hálózati függés a HuggingFace felé production runtime-ban. **Ezt mindenképp el kell kerülni.**

**Verdikt:** a legegyszerűbb, legalacsonyabb kockázatú út, HA a CPU-wheel mérete és a RAM belefér.

---

### 4.2 B opció — ONNX Runtime ⭐ *(AJÁNLOTT — a numerikus ekvivalencia mérésileg bizonyítva)*

**Hogyan működik:**
1. **Build- vagy CI-időben** (nem a prod image-ben) exportáljuk a `SZTAKI-HLT/hubert-base-cc` BertModel-t ONNX-be (`optimum-cli export onnx`, vagy `torch.onnx.export` opset 14, dinamikus batch/seq tengelyekkel). Kimenet: `hubert-base-cc.onnx`, output = `last_hidden_state`.
2. A prod image tartalmazza: `onnxruntime` + `tokenizers` (a HF fast tokenizer Rust magja, **torch nélkül**) + a `.onnx` fájl.
3. Runtime: `ort.InferenceSession(...).run()` → `last_hidden_state` (numpy) → **ugyanaz a mean pooling és L2 normalizálás numpy-ban** → `List[float]`.

**Embedding-tér kompatibilitás — a kulcskérdés.**

Ez azért működik, mert (2.3) **a pooling és a normalizálás a modellen kívül van**. Az ONNX graph csak a transformer forward passt tartalmazza; a `sum(emb*mask)/clamp(mask.sum(),1e-9)` és az L2 normalizálás numpy-ban bit-szinten ugyanaz a művelet-sorrend. Az egyetlen eltérés a transformer belső float32 aritmetikájának kernel-szintű különbsége (ORT vs ATen), ami fp32-ben tipikusan 1e-5 – 1e-6 nagyságrendű.

> **✅ EZ MÉRÉSSEL BIZONYÍTVA LETT — lásd 4.7.2.** cos min **0,9999997616**, max abs elemeltérés **1,788e-07**, 13/13 szöveg a küszöb felett, és a rangsor **6/6 query-n azonos** (max score-delta 2,384e-07). Az fp32 ONNX **drop-in csere**: a ~54k indexelt vektor **egyetlen bitjét sem kell újraszámolni.**

**Méret (mért, 4.7.6):** `onnxruntime==1.28.0` = 18,3 MB wheel / **58 MB telepítve**; a minimális deployolható stack (`onnxruntime`+`tokenizers`+`numpy`) **176 MB**. A `.onnx` fp32 modellfájl **440,36 MB — egyetlen fájl**, nincs külső `.onnx_data`.

**Reális image-delta a B opcióra: ≈ 176 MB + 440 MB = 616 MB** — szemben az A opció **1 339 MB**-jával (894 MB torch stack + 445 MB safetensors). **Megtakarítás: ~720 MB (−54%).**

**Memória (mért, 4.7.5):** 594,5 MB peak RSS vs a torch 802,5 MB-ja → **−208 MB workerenként** — **de csak akkor, ha a `transformers` teljesen kimarad** (6.3). Ha bent marad, az ONNX út **937,7 MB-ot** eszik, azaz rosszabb, mint a tiszta torch.

> **FIGYELEM — `optimum` NEM használható runtime-ban:** `requires_dist` szerint `transformers>=4.29` **ÉS `torch>=1.11`**. Az `optimum` csak az **export-oldalon** (CI/dev) szerepelhet, a prod image-ben soha. Runtime-ra kizárólag `onnxruntime` + `tokenizers` megy.

**int8 dinamikus kvantálás (B2) — ŐSZINTE ÉRTÉKELÉS.**
Az int8 kvantálás **eltolja** az embedding-teret. A veszély itt nem az abszolút cosine érték (0.98 önmagában "jónak" hangzik), hanem hogy a **rangsor** megváltozik: a 768-dim térben egy 54k elemű korpusz top-5 találatai gyakran 0.01–0.03 cosine-on belül vannak egymástól, tehát egy 0.01–0.02-es query-oldali drift átrendezheti a találati sorrendet. Ráadásul aszimmetrikus helyzet: az indexelt vektorok fp32 torch-csal készültek, csak a query lenne int8. **A döntéshez a rangsor-teszt (8.3) az irányadó, nem a nyers cosine.**

**Hol lakjon a `.onnx` fájl?**

| Változat | Előny | Hátrány |
|----------|-------|---------|
| **Az image-be sütve** (git LFS vagy CI-ben generált) | nincs runtime hálózati függés, determinisztikus | az image nő a modell méretével; a repo/CI-nek kezelnie kell egy bináris artifactot |
| Object storage (S3/R2) + boot-time download | kis image | runtime hálózati függés, boot-lassulás, credential-kezelés |
| HF Hub-ról letöltve boot-kor | nincs saját infra | **ugyanaz a runtime HF-függés, amit el akarunk kerülni**; nincs is hivatalos ONNX export ehhez a modellhez |

**Ajánlás: CI-ben exportálni, artifactként az image-be sütni**, a `.onnx` SHA256-ját pedig verifikálni a build során.

**Kód-változás:** közepes, de **jól izolált** — az `embedding_service.py` egy backend-absztrakciót kap (7.2), a publikus interfész (`embed_text_async(text, preprocess, use_cache)`) **változatlan marad**.

**Hibamódok:**
- Az export lépés elmarad/eltörik → a build-nek **hasítania kell** (fail fast), nem szabad némán torch-fallbackre váltani.
- `HUBERT_REVISION` bump után az `.onnx` **elavul** → az exportot a revízióhoz kell kötni, és a fájlnévbe/checksumba bele kell írni.
- ORT thread-kezelés: alapból az összes magot használja; 2 workerrel túlfoglalhat. `intra_op_num_threads` explicit beállítása kell.

---

### 4.3 C opció — külön embedding microservice

**Hogyan működik:** egy második Railway service (saját `Dockerfile`, torch + transformers), ami egy `POST /embed {"texts": [...]}` endpointot ad. A backend HTTP-n hívja (`httpx`, már benne van: `requirements.prod.txt:37`).

**Embedding-tér kompatibilitás: 100%** (ugyanaz a torch kódút).

**Előny:** a backend image marad kicsi; az embedding külön skálázható; a modell csak egy helyen töltődik (nem workerenként).

**Hátrány — és ez sok:**
- **Költség:** +1 Railway service. A `docs/BUDGET_AND_RESOURCES.md:114-117` alapján a backend Pro $50/hó; egy kisebb worker service **BECSLÉS: +$5–20/hó**.
- **Latencia:** +1 hálózati RTT. Railway private networkön belül BECSLÉS: +2–10 ms, ami a 50–200 ms inference mellett elhanyagolható. **De** cold startnál (ha a service alszik) másodpercek.
- **Hibakezelés:** új hibamód (a microservice le van, timeout, 502). Kell timeout + retry + circuit breaker. Ha rosszul csináljuk, pont ugyanazt a csendes degradációt kapjuk vissza más köntösben.
- **Deploy-komplexitás:** két service verzió-szinkronban tartása; a modell-revízió driftje két helyen.
- **Ugyanaz a RAM-probléma**, csak áthelyezve — a microservice-nek is kell a memória.

**Verdikt:** akkor helyes, ha az embedding-terhelés jelentősen megnő vagy GPU kell. Ma **over-engineering** — egy egyszemélyes projektben egy plusz deployolandó, monitorozandó, fizetendő komponens.

---

### 4.4 D opció — hosted embedding API

**KRITIKUS: ez nem egy "csere", hanem egy adatmigrációs projekt.**

A meglévő ~54k (a `CLAUDE.md` szerint 35 000+; a pontos szám operátori ellenőrzést igényel — lásd 9.1) vektor a `hubert-base-cc` **768-dimenziós terében** él. Bármely külső szolgáltató **más térben és gyakran más dimenzióban** dolgozik:

| Szolgáltató | Modell | Dim | Magyar támogatás | Reindex kell? |
|-------------|--------|-----|------------------|---------------|
| OpenAI | `text-embedding-3-small` | 1536 | többnyelvű, jó | **IGEN, teljes** |
| OpenAI | `text-embedding-3-large` | 3072 | többnyelvű, jó | **IGEN, teljes** |
| Cohere | `embed-multilingual-v3.0` | 1024 | jó | **IGEN, teljes** |
| Voyage | `voyage-3` | 1024 | többnyelvű | **IGEN, teljes** |
| Jina | `jina-embeddings-v3` | 1024 (matryoshka) | többnyelvű | **IGEN, teljes** |

**A reindex valós költsége (őszintén):**

- A Qdrant collection-t **újra kell építeni** (a dimenzió változik → nem lehet in-place, új collection + átállás kell).
- Újra kell embedelni: ~54k vektor (DTC + recall + complaint).
- Ha a teljes NHTSA complaint korpuszt is indexelni akarjuk (a roadmap 170 000+-ról beszél, `tasks/roadmap_v1_to_q2_2026.md:16`), az nagyságrendekkel több.
- **Költség BECSLÉS** (OpenAI `text-embedding-3-small`, $0.02 / 1M token, átlag ~250 token/rekord): 54 000 × 250 = 13,5M token ≈ **$0,27**. 170 000 rekordra ≈ **$0,85**. Azaz a **pénzköltség elhanyagolható**.
- **Az idő- és kockázatköltség nem az.** A pipeline-t (`scripts/index_qdrant_hubert.py`) át kell írni, a `EMBEDDING_DIMENSION`-t, a collection-kezelést, a `consistency_service.py`-t, a teszteket. A dual-write/átállás alatt a régi és új collection párhuzamosan él. **BECSLÉS: 2–4 nap fejlesztés + validáció.**
- **Új futó költség és külső függés minden egyes query-re** — pont az, amitől a HuBERT lokális futása megszabadított (`CLAUDE.md`: "Lokális futás: Nincs API limit (Groq kimerült)").

**Nincs olyan hosted provider, ami `hubert-base-cc`-t szolgálná ki.** Ez egy SZTAKI-HLT akadémiai modell, nem szerepel egyetlen kereskedelmi embedding API katalógusában sem.

#### D2 — HuggingFace Inference Endpoint a `hubert-base-cc`-vel

Ez az egyetlen hosted út, ami **megőrzi az embedding-teret**: dedikált HF Inference Endpointon futtatható a pontos modell.
- **De:** a HF feature-extraction endpoint a `last_hidden_state`-et vagy a HF saját poolingját adja vissza — a mean-pooling+L2-t **nekünk kell** a kliens oldalon elvégezni (ami jó, mert így kontrolláljuk). Ellenőrizni kell, hogy pontosan mit ad vissza.
- **Költség:** a dedikált CPU endpoint **BECSLÉS: ~$0,06–0,12/óra ≈ $45–90/hó** folyamatos futás mellett. Scale-to-zero esetén olcsóbb, de a cold start 30–60 s.
- **Verdikt:** funkcionálisan a C opció, csak drágábban és külső vendorral. A C-nél nincs jobb.

---

### 4.5 E opció — nincs query embedding (lexikai only)

**Hogyan működik:** kivesszük a szemantikus ágat, marad a PostgreSQL text search (`dtc_codes.py:462`) és a Neo4j gráf (`rag_service.py:807`), kiegészítve egy kézzel kurált magyar tünet → DTC mapping táblával.

**Ami elveszik:**
- A ~54k Qdrant vektor **teljesen halott befektetéssé válik** (és a Qdrant Cloud költsége is: `docs/BUDGET_AND_RESOURCES.md:133` — Professional 8GB RAM tervezve).
- Az NHTSA complaint hasonlóság (`similar_complaints` a ResultPage-en) megszűnik.
- A recall szemantikus illesztés megszűnik.
- **A termék központi ígérete sérül** — `CLAUDE.md`: "RAG alapja: A diagnosztikai AI innen keres releváns információt". A `tasks/bug_broken_promises.md:28` már most is ezt a törött ígéretet dokumentálja.
- A magyar morfológia miatt a lexikai keresés **gyengén** működik magyarul ("rángat" / "rángatás" / "rángatva" nem egyezik trigram nélkül).

**Mikor lenne mégis jó:** ha a döntés az, hogy a szemantikus keresés nem éri meg az infrastruktúra-terhet. Akkor viszont **ki kell mondani** a felhasználó felé is, és a Qdrant-ot le kell építeni. Félúton maradni (fizetni a Qdrantért, miközben nullvektort küldünk bele) a legrosszabb állapot — **ez a jelenlegi állapot.**

---

### 4.6 F opció — kisebb többnyelvű modell (ha a reindex amúgy is asztalon van)

Ha a `HUBERT_REVISION` pinelés miatt (2.5/3.) vagy a `vehicle_make` payload-javítás miatt (`tasks/code_review_fixes_2c28099.md:48-58`) **amúgy is reindexelünk**, felmerül: érdemes-e modellt is váltani?

| Modell | Dim | Paraméter | Magyar | Megjegyzés |
|--------|-----|-----------|--------|------------|
| `SZTAKI-HLT/hubert-base-cc` (jelenlegi) | 768 | ~110M | **natív magyar**, magyar korpuszon tanítva | **NEM** sentence-embedding modellként tanítva — mean-pooled BERT, ami elméletileg gyengébb, mint egy kontrasztívan tanított modell |
| `intfloat/multilingual-e5-small` | 384 | ~118M | jó (100 nyelv) | **kontrasztívan tanított** retrievalre; `query:` / `passage:` prefix kötelező |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | ~118M | közepes-jó | kicsi, gyors |
| `intfloat/multilingual-e5-base` | 768 | ~278M | nagyon jó | nagyobb, lassabb |

**Fontos szakmai megjegyzés:** a `hubert-base-cc` egy **nyers MLM-BERT**, nem sentence-transformer. A mean-pooled nyers BERT embedding a szakirodalom szerint jellemzően **gyengébb** szemantikus keresésre, mint egy retrieval-re finomhangolt modell — még akkor is, ha az kisebb és többnyelvű. Elképzelhető tehát, hogy egy `multilingual-e5-small` **jobb** magyar retrieval minőséget adna 384 dimenzióban, feleakkora indexszel.

**DE:** ez egy külön, mérendő hipotézis, nem egy feltevés. Kellene hozzá egy magyar nyelvű retrieval kiértékelő halmaz (kb. 50–100 query→várt DTC pár), amit a projekt **jelenleg nem birtokol**.

**Verdikt: NE most.** Ez a "gépjármű-diagnosztikai retrieval minőség" projekt, nem a "szemantikus keresés bekapcsolása" projekt. Először működjön, aztán mérjünk, aztán optimalizáljunk. **Felvéve a backlogba.**

---

### 4.7 Mért adatok

#### 4.7.1 Csomagméretek — ÉLŐBEN ELLENŐRIZVE (PyPI JSON API + download.pytorch.org, 2026-07-25)

**A torch kérdés:**

| Csomag | Wheel | Megjegyzés |
|--------|-------|-----------|
| `torch==2.2.0` (PyPI default, cp311 manylinux1_x86_64) | **720,5 MB** | CUDA-bundled |
| + 11 db `nvidia-*` wheel (cudnn 697,8 / cublas 391,6 / cusparse 186,9 / nccl 158,3 / cusolver 118,4 / cufft 116,0 / curand 53,9 / …) | **1 759,8 MB** | `platform_system=="Linux" and platform_machine=="x86_64"` esetén **feltétel nélkül** |
| + `triton==2.2.0` | 160,2 MB | szintén Linux-x86_64-re pinelve |
| **PyPI default ÖSSZESEN** | **2 640,4 MB (2,58 GiB)** | ← ez a `requirements.prod.txt:49` "~2GB" |
| **`torch==2.2.0+cpu`** (pytorch.org/whl/cpu, cp311) | **178,1 MB** | **nulla nvidia/triton dep** |
| `torch==2.5.1+cpu` | 166,6 MB | a legkisebb a mintázott CPU buildek közül |
| `torch==2.13.0+cpu` | 182,9 MB | legújabb a cpu indexen |

> **Megtakarítás CUDA → CPU wheel: 2 462 MB (2,40 GiB).**

**Kicsomagolt (lemezen, ténylegesen megmérve `pip download` + unzip):**

| Artifact | Wheel | Kicsomagolva | Arány |
|----------|-------|--------------|-------|
| `torch==2.2.0+cpu` | 178,1 MB | **648,4 MB** | 3,64× |
| `onnxruntime==1.28.0` | 18,3 MB | **51,7 MB** | 2,83× |

*(A CUDA wheelek kicsomagolt mérete NEM lett megmérve — hasonló aránnyal ~6–9 GB-ra becsülhető, de ez BECSLÉS.)*

**A transformers valós ára** (`transformers==4.37.2` + csak azok a függőségek, amik még NINCSENEK a `requirements.prod.txt`-ben):

| Csomag | Wheel |
|--------|-------|
| `transformers==4.37.2` | 8,01 MB |
| `tokenizers==0.15.2` | 3,41 MB |
| `safetensors==0.4.2` | 1,21 MB |
| `huggingface-hub==0.20.3` | 0,31 MB |
| `regex`, `filelock` | 0,76 MB |
| **ÚJ bájt összesen** | **≈ 13,7 MB** |

*(A `pyyaml`, `requests`, `packaging`, `numpy`, `tqdm` már bent van a langchain/explicit pinek miatt.)*

**`sentence-transformers==2.3.1`:** maga 0,13 MB, **de** ~56 MB scipy/sklearn/Pillow/nltk/sentencepiece farkat és egy `torch>=1.11` hard requirementet hoz. **Nem kell, ne kerüljön vissza.**

**ONNX út:**

| Csomag | Wheel | Megjegyzés |
|--------|-------|-----------|
| `onnxruntime==1.28.0` | **18,3 MB** | depek: flatbuffers (0,03), protobuf (0,16), numpy, packaging |
| `optimum==2.2.0` | 0,15 MB | **DE `requires_dist`: `transformers>=4.29` ÉS `torch>=1.11`** → csak build/CI oldalon! |

#### 4.7.2 Numerikus ekvivalencia — TÉNYLEGESEN LEMÉRVE

> Környezet: Python 3.11.15, `torch 2.13.0+cpu`, `onnxruntime 1.28.0`, `onnx 1.22.0`, `transformers 5.14.1`, numpy 2.4.6, 4 vCPU / 16 GB. Export: `torch.onnx.export` (legacy path, `dynamo=False`), opset 14, dinamikus batch+seq tengely, 9,9 s. A referencia-pooling **pontosan** a `embedding_service.py:291-312` + `:395` szerint.

**A modell:**

| Adat | Érték |
|------|-------|
| Paraméterszám (`BertModel`) | **110 618 112** |
| `model.safetensors` | **445,0 MB** |
| HF cache összesen | 425 MB |
| Config | 12 réteg, hidden 768, vocab 32 001, max_pos 512 |
| **`hubert_fp32.onnx`** | **440,36 MB — EGYETLEN fájl**, nincs külső `.onnx_data` |

> Megjegyzés: a repóban nincs `tokenizer.json`; a fast tokenizer a `vocab.txt`-ből épül (`do_lower_case: false`). Ez a 6.3-ban fontossá válik.

**torch ↔ ONNX-fp32 ekvivalencia (13 magyar tesztszöveg, 2–448 token):**

| Metrika | Mért érték | Küszöb (8.2) | |
|---------|-----------|--------------|---|
| cosine **min** | **0,9999997616** | > 0,9999 | ✅ |
| cosine átlag | 0,9999999908 | — | ✅ |
| max abs elemenkénti eltérés | **1,788e-07** | < 1e-4 | ✅ |
| `cos > 0.9999` arány | **13 / 13** | 13/13 | ✅ |

Vegyes hosszúságú 4-es batch is egyezett (cos 1,0000) → a dinamikus tengelyek általánosítanak. Mind a torch, mind az ONNX-fp32 **bit-pontosan reprodukálható** (30 futásból 1 különböző kimenet).

**Rangsor-teszt (8.3) — 40 dokumentumos magyar autós korpusz torch-csal indexelve, 6 query:**

| Query encoder | top-5 sorrend azonos | top-1 azonos | átlag top-5 átfedés | max score-delta |
|---------------|:--------------------:|:------------:|:-------------------:|-----------------|
| **onnx-fp32** | **6 / 6** ✅ | 6 / 6 | **1,00** | **2,384e-07** |
| onnx-int8 (QInt8) | 0 / 6 ❌ | 4 / 6 | 0,47 | 4,372e-02 |
| onnx-uint8 (QUInt8) | 0 / 6 ❌ | 6 / 6 | 0,70 | 2,261e-02 |

> **Az embedding-tér kompatibilitás ezzel BIZONYÍTOTT, nem feltételezett.**

#### 4.7.3 Az int8 KIZÁRVA — két, egymástól független okból

1. **Pontatlanság:** `hubert_int8.onnx` = 110,83 MB (3,97× kisebb), de cos **min 0,9437 / átlag 0,9652**, max abs diff 5,02e-02, **0/13** a küszöb felett. A `QUInt8` variáns jobb (min 0,9742), a `QInt8+reduce_range` **sokkal rosszabb** (min 0,2032). A rangsor 6/6 query-n megtört, 2 query-n a **top-1 is** — pl. *"kek fust jon a kipufogobol hidegindiaskor"*: torch/fp32 helyesen a turbó-olajfogyasztást hozza (0,9737), az int8 a lambda-szondát (0,4899).
   **Miért fatális:** a mért korpuszban a torch top1–top2 margók **0,0004–0,0127** — az int8 perturbációja (2–4e-02) ennek **3–100×-osa.** 54k vektornál a margók még szűkebbek, tehát az int8 ott **rosszabbul** viselkedne, nem jobban.
2. **Nemdeterminizmus (ez önmagában is kizáró ok):** alapértelmezett szálszámmal az int8 modell **ugyanarra a bemenetre futásonként MÁS eredményt ad.** Ugyanaz a session, 30 ismétlés: egy 21-tokenes szövegre 11 különböző kimenet (cos a torch-hoz 0,358–0,972), egy 448-tokenesre **29 különböző kimenet 30-ból**. `intra_op_num_threads=1`-gyel stabilizálódik. Az fp32 ugyanezen a teszten **1/30** (teljesen determinisztikus). Ez race/uninicializált-memória hibára utal az ORT 1.28 dinamikusan kvantált BERT kerneljeiben.

> **Verdikt: az int8 nem "kicsit pontatlanabb" — futásonként más választ adna ugyanarra a kérdésre. NEM SZÁLLÍTHATÓ.**

#### 4.7.4 Latencia (medián 20 futásból, 3 warmup)

| Bemenet | Szálak | torch | onnx-fp32 | onnx-int8 |
|---------|--------|-------|-----------|-----------|
| 21 token (tipikus query) | 4 (default) | 30,3 ms | **17,6 ms** | 6,8 ms |
| 448 token | 4 (default) | **203 ms** | 274 ms | 164 ms |
| 21 token | 1 | 75,6 ms | **57,1 ms** | 9,6 ms |
| 448 token | 1 | **634 ms** | 755 ms | 267 ms |

> **Fontos aszimmetria:** az ONNX-fp32 **~1,7× gyorsabb rövid (query méretű) szövegre**, de **10–30%-kal lassabb hosszú (448 tokenes) inputon**. A runtime query-út tehát nyer; a tömeges hosszú-dokumentum indexelés nem. Ez az AutoCognitix profiljához illik (a query rövid tünetszöveg, az indexelés offline).

**Cold load (lokális cache-ből):** `AutoModel.from_pretrained` **0,35–0,62 s** vs `ort.InferenceSession(fp32)` **0,87–0,99 s** — az ORT session ~2× lassabban áll fel, de mindkettő elhanyagolható a 100 s-os healthcheck timeouthoz képest.

#### 4.7.5 Memória (peak RSS) — és a `transformers` csapda

| Folyamat | Peak RSS |
|----------|----------|
| torch + transformers + hubert | **802,5 MB** |
| onnxruntime + fp32, **de `transformers` importálva** | 937,7 MB ⚠️ |
| **onnxruntime + fp32, torch-mentes (`tokenizers` a `vocab.txt`-ből)** | **594,5 MB** |

> **KRITIKUS IMPLEMENTÁCIÓS RÉSZLET:** a `transformers` import **behúzza a torchot** (+367 MB, még mielőtt bármi betöltődne). **Az ONNX csak akkor spórol memóriát, ha a `transformers` teljesen kimarad**, és a tokenizálás közvetlenül `tokenizers.BertWordPieceTokenizer(vocab.txt, lowercase=False)`-szal történik.
>
> **Ez le lett mérve:** ez a tokenizer a 13 tesztszöveg mindegyikére **azonos token ID-kat** ad, mint az `AutoTokenizer`. (`do_lower_case: false` a config szerint — a `lowercase=False` tehát kötelező, és ez a beállítás hibázás esetén csendben rontaná az embeddinget.)

#### 4.7.6 Telepített méret (venv site-packages, `du`-val mérve)

| Stack | Méret |
|-------|-------|
| `torch(+cpu)` egyedül | **894 MB** (torch 754 + sympy 80 + networkx 19 + mpmath 5,1) |
| `onnxruntime` egyedül | **58 MB** |
| **minimális deployolható stack** (`onnxruntime`+`tokenizers`+`numpy`) | **204 MB** (176 MB pip/setuptools nélkül) |

> A `torch` telepített mérete (894 MB) **nagyobb, mint a puszta wheel-kicsomagolás (648 MB)**, mert a `sympy`/`networkx`/`mpmath` függőségek hozzáadódnak.
>
> **Teljes image-delta összevetés (runtime + modell):**
> - **A (torch):** 894 MB + 445 MB safetensors ≈ **1 339 MB**
> - **B (ONNX):** 176 MB + 440 MB `.onnx` ≈ **616 MB**
> - **Különbség: ~720 MB (−54%)**, plusz **−208 MB RSS workerenként** (802,5 → 594,5).

#### 4.7.7 A mérés korlátai (őszintén)

- A mérés **`transformers 5.14.1`** + `sdpa` attention mellett futott, nem a projekt által pinelt `4.37.2` + `eager` mellett. A fp32 paritás-bizonyítás a *mért* torch-gráf és az *abból exportált* ONNX között áll fenn, tehát önmagában érvényes — de **a 8.2/8.3 verifikációt újra le kell futtatni a projekt pinelt környezetében** (7.0) a shipping előtt. Az eltérés várhatóan ~1e-6 nagyságrendű, azaz bőven a 0,9999 küszöb felett, de ez BECSLÉS, nem mérés.
- **Recall@k valódi relevancia-címkékkel: NEM MÉRVE** (nincs címkézett halmaz — lásd 9.6). Csak a torch-rangsorral való egyezés lett mérve, ami viszont **pontosan a migrációs kérdés helyes metrikája**.
- A teljes 54k-vektoros korpusz viselkedése **nem lett mérve** (40 dokumentumos proxy). A margók 54k-nál szűkebbek → az fp32 (2,4e-07 delta) továbbra is bőven biztonságos, az int8 viszont **rosszabbul** járna.
- Batch-throughput és a default feletti ORT graph-optimalizációs szintek: **nem mérve**.

---

## 5. A CSENDES NULLVEKTOR — a legfontosabb tanulság

Bármelyik opciót választjuk, **ez a javítás kötelező, és önmagában is a legnagyobb értékű változtatás.**

### 5.1 Miért maradhatott hónapokig rejtve?

Három egymásra rakódó csendes fallback:

1. `embedding_service.py:362-364` — nincs torch → `[0.0]*768`, `logger.warning` szinten, **kivétel nélkül**
2. `rag_service.py:500-505` — `except (RuntimeError, Exception)` → `query_embedding = None` → `return []`, `logger.warning`
3. `diagnosis_service.py:652-658` — `except Exception` → `_fallback_diagnosis()`, ami a felhasználónak **ugyanúgy néz ki**, mint egy valódi AI-jelentés (`tasks/bug_broken_promises.md:32-35`)

Ráadásul a `[0.0]*768` **típushelyes**: 768 elemű float lista. Minden downstream ellenőrzés (dimenzió, típus) átengedi. Nincs olyan invariáns a rendszerben, ami kimondaná: *"egy similarity search query vektorának egységhosszúnak kell lennie"*.

### 5.2 A javítás elve

> **Egy nullvektor SOHA nem kerülhet be egy similarity search-be. Az embedding hiánya HIBA, nem üres eredmény.**

Három szint:

| Szint | Változtatás | Hely |
|-------|-------------|------|
| **1. Fail loudly** | `TORCH_AVAILABLE == False` (ill. backend nem elérhető) esetén az `embed_text` **`EmbeddingUnavailableError`-t dob**, nem nullvektort ad | `embedding_service.py:362-364`, `:425-427` |
| **2. Explicit degraded flag** | a `retrieve_from_qdrant` elkapja, de a `RAGContext`-be beállít egy `semantic_search_available=False` + `degraded_reason` mezőt, ami **felbugyog a `/diagnosis/analyze` válaszába** (a meglévő `used_fallback` mellé) | `rag_service.py:497-505`, `diagnosis` schema |
| **3. Startup + health probe** | boot-kor egyszeri önteszt: embedel egy fix magyar mondatot, ellenőrzi hogy `abs(norm - 1.0) < 1e-4`; az eredmény megjelenik a `/api/v1/health/detailed`-ben `embedding: ok/unavailable/degraded` néven | `main.py` lifespan, `health.py:467+` |

Plusz egy **defenzív guard a vektor-keresés kapujában** (`qdrant_client.py:search()`): ha `np.linalg.norm(query_vector) < 1e-6`, dobjon `ValueError`-t. Ez az utolsó védvonal: bármilyen jövőbeli út, ami nullvektort próbál keresésre használni, azonnal hasít.

### 5.3 Amit az üres szöveggel kell csinálni

`embedding_service.py:371-373` üres szövegre is `[0.0]*768`-at ad. Ez az **indexelésnél** védhető (egy üres rekord ne kapjon random vektort), de a **query oldalon** nem: az üres query-t a hívónak kell kiszűrnie, a embedding rétegnek pedig `ValueError`-t kell dobnia. Javasolt: az `embed_text` kapjon egy `allow_empty: bool = False` paramétert; a batch-indexelő útján `True`.

---

## 6. AJÁNLÁS

### 6.1 Hogyan alakult a döntés (a bizonyítékok sorrendjében)

Ez a szakasz szándékosan mutatja meg az érvelés útját, mert két premissza is megdőlt menet közben:

1. **Kiinduló feltevés:** "a torch 2 GB, ezért ki kell hagyni" → **RÉSZBEN CÁFOLVA.** A 2,58 GiB a **CUDA** wheelre igaz; a `+cpu` wheel 178 MB, telepítve 894 MB. A tiltás oka tehát egy **wheel-index félreértés** volt — de a torch még CPU-ban sem "kicsi".
2. **Második feltevés:** "akkor az A a legegyszerűbb, vigyük vissza a torchot" → **CÁFOLVA a numerikus mérés által.** Az ONNX-fp32 ekvivalenciája nem "valószínű", hanem **bizonyított**: cos ≥ 0,9999997, max elemeltérés 1,79e-07, és a rangsor **6/6 query-n azonos**. Ezzel a B egyetlen valódi kockázata — hogy elrontja az embedding-teret — **megszűnt**, miközben minden előnye megmaradt.
3. **Harmadik feltevés:** "az int8 még kisebb és gyorsabb lenne" → **HATÁROZOTTAN CÁFOLVA**, két független okból (4.7.3).

### 6.2 Elsődleges ajánlás: **B opció — ONNX Runtime fp32, a `.onnx` az image-be sütve, `transformers` NÉLKÜL**

**Indoklás — mind a négy pont mért adaton áll:**

1. **Az embedding-tér kompatibilitás BIZONYÍTOTT, nem feltételezett.** cos **min 0,9999997616**, max abs elemeltérés **1,79e-07**, és a 40 dokumentumos rangsor-teszten **6/6 query-n azonos top-5 sorrend, 2,4e-07 max score-delta** (4.7.2). A ~54k már indexelt vektor **egyetlen bitjét sem kell újraszámolni.** Ez az a kockázat, ami miatt a B kérdéses volt — és a mérés elvette.

2. **~720 MB kisebb image, ~208 MB kisebb RSS workerenként.**
   - A (torch): 894 MB stack + 445 MB safetensors ≈ **1 339 MB**; RSS 802,5 MB
   - B (ONNX): 176 MB stack + 440 MB `.onnx` ≈ **616 MB**; RSS **594,5 MB**
   - 2 gunicorn workerrel (`Dockerfile.prod:96`) ez **1,61 GB vs 1,19 GB** RAM — ami egy dokumentálatlan RAM-limitű Railway service-nél (9.1) érdemi mozgástér.

3. **~1,7× gyorsabb a query-úton.** 21 tokenes tipikus tünetszövegre 17,6 ms vs 30,3 ms (4 szál). A hosszú (448 tokenes) inputon ugyan 10–30%-kal lassabb, de az a **offline indexelés** profilja, nem a runtime query-é — és az indexelés amúgy is torch-csal, scriptből fut.

4. **A `torch` teljesen kikerül a prod image-ből** — vele együtt a `sympy`/`networkx`/`mpmath` farok és a torch teljes autograd/JIT/distributed felülete, amiből a projekt **semmit nem használ** (`torch.no_grad()` mindenhol: `:388`, `:534`).

### 6.3 A B opció KÖTELEZŐ implementációs feltétele

> **A `transformers` NEM kerülhet be a prod image-be.** A mérés szerint a `transformers` import **behúzza a torchot** (+367 MB), és ezzel az ONNX teljes memória-előnye elvész (937,7 MB RSS — **rosszabb, mint a tiszta torch 802,5 MB-ja**).
>
> A tokenizálás ezért közvetlenül `tokenizers.BertWordPieceTokenizer(vocab.txt, lowercase=False)`-szal történik. **Ez le lett mérve: a 13 tesztszöveg mindegyikére azonos token ID-kat ad, mint az `AutoTokenizer`.**
>
> A `lowercase=False` **kritikus** (`config.json`: `do_lower_case: false`). Ha valaki elrontja, az embedding csendben romlik — ezért a 8.6 regressziós teszt egyik befagyasztott fixture-je **kötelezően tartalmazzon magyar nagybetűs/ékezetes szöveget**.

### 6.4 Fallback ajánlás: **A opció — `torch==2.x+cpu` + `transformers`**

Az A akkor lép elő, ha **bármelyik** teljesül:
- a 7.0 újra-verifikáció **a projekt pinelt `transformers==4.37.2` környezetében** nem hozza a 8.2/8.3 küszöböket (4.7.7 — a mérés `transformers 5.14.1`-gyel futott);
- vagy a `.onnx` bináris artifact kezelése (CI/git LFS) szervezetileg nem vállalható (9.4);
- vagy a 8.1 revízió-teszt megbukik — **ekkor mindkét opció másodlagos**, mert teljes reindex kell.

Az A továbbra is teljesen működőképes, csak drágább (image, RAM) és lassabb a query-úton. **Mellékhaszon nála:** a `scripts/index_qdrant*.py` és a `docs/RAILWAY_DEPLOYMENT.md:204-210` reindex-lépése működővé válna Railway-en, ami ma nem az.

> **A két opció nem zárja ki egymást.** A 7.2 backend-absztrakcióval a `torch` és az `onnx` backend egymás mellett él, a váltás egyetlen env-változó (`EMBEDDING_BACKEND`). Az indexelő scriptek (offline, dev gépen) továbbra is a torch úton futnak — **ott a torch marad, és az az embedding-tér horgonya.**

### 6.5 Amit NEM ajánlok:
- **int8 / uint8 kvantálás (B2) — HATÁROZOTT NEM.** Nem "kicsit pontatlanabb": (a) a rangsor 6/6 query-n megtört, 2 query-n a top-1 is; (b) **alapértelmezett szálszámmal nemdeterminisztikus** — 30 futásból 29 különböző kimenet ugyanarra a hosszú inputra (4.7.3). Egy diagnosztikai terméknél, ahol a felhasználó ugyanarra a kérdésre kétszer más választ kapna, ez nem tárgyalható.
- **D / D2** — reindexet és/vagy új futó költséget kényszerít, cserébe semmit nem old meg, amit B/A ne oldana meg.
- **C** — a mai terhelésre over-engineering.
- **F (modellváltás)** — külön projekt, kiértékelő halmaz (9.6) nélkül nem felelős döntés.

### 6.6 A javítás MINDIG három részből áll

Ezek **együtt** szállítandók, különben a probléma nem szűnik meg:

| # | Rész | Nélküle mi történik |
|---|------|---------------------|
| **R1** | Működő query embedding (B vagy A) | nullvektor marad |
| **R2** | RAG collection-drift javítása (`rag_service.py:794, 801` → unified `autocognitix` + `type` szűrő) | `/diagnosis/analyze` **továbbra is** 0 kontextust kap |
| **R3** | Safety guard (5. fejezet) | a következő ilyen hiba megint hónapokig rejtve marad |

---

## 7. Implementációs terv (B opció elsődlegesen, A-val mint felcserélhető backenddel)

### 7.0 Fázis 0 — ADATGYŰJTÉS és ÚJRA-VERIFIKÁCIÓ (a kód érintése előtt)

Négy dolog, amit **a kód érintése előtt** meg kell szerezni:

1. **9.2** — `GET /collections` a Qdrant Cloudon: mi a tényleges pontszám az `autocognitix`-ban, léteznek-e a `*_hu` collectionök.
2. **8.1** — a modell-revízió rekonstrukciós teszt. **Ha ez megbukik, a torch/ONNX kérdés másodlagos** — teljes reindex kell, és a modellválasztás (F) újranyílik.
3. **8.2 + 8.3 ÚJRAFUTTATÁSA a projekt pinelt környezetében** (`transformers==4.37.2`, a prodban használt `HUBERT_REVISION`-nel). A 4.7.2 mérés `transformers 5.14.1`-gyel készült; a paritás onnan **átvihető, de nem átvett** (4.7.7). Ez az egyetlen fennmaradó technikai kockázat a B opcióban.
4. **9.1** — Railway backend service RAM limit (dashboard → Metrics). Ez már nem dönt a B/A között (a B mindkét irányban jobb), de a `--workers` beállításához kell.

### 7.1 Fájlok, amiket érint

| Fájl | Változás | A opció | B opció |
|------|----------|:-------:|:-------:|
| `backend/app/services/embedding_service.py` | backend-absztrakció + a nullvektor-fallback cseréje | ✅ | ✅ |
| `backend/app/services/embedding_backends/__init__.py` *(új)* | `EmbeddingBackend` protokoll + `_select_backend()` | ✅ | ✅ |
| `backend/app/services/embedding_backends/torch_backend.py` *(új)* | a meglévő torch forward-pass kiemelve | ✅ | – |
| `backend/app/services/embedding_backends/onnx_backend.py` *(új)* | ORT session forward-pass | – | ✅ |
| `backend/requirements.prod.txt` | A: `+torch==2.x+cpu`, `+transformers==4.37.2` · B: `+onnxruntime`, `+tokenizers` | ✅ | ✅ |
| `backend/Dockerfile.prod` | A: CPU wheel index + build-time modell-prefetch · B: `.onnx` bemásolás | ✅ | ✅ |
| `backend/app/core/config.py` | `EMBEDDING_BACKEND` beállítás + `HUBERT_REVISION` SHA-ra | ✅ | ✅ |
| `scripts/export_hubert_onnx.py` *(új)* | reprodukálható ONNX export | – | ✅ |
| `backend/app/services/rag_service.py` | **R2** — collection-drift javítás (`:794`, `:801`) | ✅ | ✅ |
| `backend/app/db/qdrant_client.py` | **R3** — nullvektor guard a `search()`-ben (`:189` előtt) | ✅ | ✅ |
| `backend/app/api/v1/endpoints/health.py` | **R3** — `embedding` komponens a `/detailed`-be | ✅ | ✅ |
| `backend/app/main.py` | **R3** — boot-time embedding önteszt (nem blokkoló a healthcheckre) | ✅ | ✅ |
| `backend/tests/unit/test_embedding_service.py` | a "nincs torch → nullvektor" tesztek átírása "→ raises"-re | ✅ | ✅ |
| `backend/tests/test_sprint_review_audit.py` | új regressziós tesztek (8.6) | ✅ | ✅ |

### 7.1b A B opció (AJÁNLOTT) konkrét diffje

**`backend/requirements.prod.txt`** — a `:46-51` kommentblokk helyére:
```
# AI/ML - Local Hungarian embeddings (huBERT) via ONNX Runtime.
# torch/transformers are DELIBERATELY absent: importing `transformers` pulls in
# torch (+367 MB RSS, measured) and cancels the entire benefit of the ONNX path.
# Tokenization goes through `tokenizers` directly (BertWordPieceTokenizer on
# vocab.txt, lowercase=False -- config.json says do_lower_case: false).
# Measured: identical token IDs to AutoTokenizer; embeddings cos >= 0.9999997
# vs the torch reference that produced the indexed Qdrant vectors.
onnxruntime==1.28.0
tokenizers>=0.15,<0.16
```

**`backend/Dockerfile.prod`** — új export stage a builder ELÉ (a `.onnx` sosem a prod stage-ben készül):
```dockerfile
# ---- Stage 0: ONNX export (heavy, torch present, DISCARDED after export) ----
FROM python:3.11-slim-bookworm AS onnx-export
ARG HUBERT_REVISION=main
ARG HUBERT_ONNX_SHA256
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.0+cpu transformers==4.37.2 onnx
COPY scripts/export_hubert_onnx.py .
RUN python export_hubert_onnx.py \
        --revision "${HUBERT_REVISION}" \
        --out /models/hubert_fp32.onnx \
        --vocab-out /models/vocab.txt \
        --verify           # fails the build if cos < 0.9999 (see 8.2/8.3)
RUN echo "${HUBERT_ONNX_SHA256}  /models/hubert_fp32.onnx" | sha256sum -c -
```
majd a production stage-ben (`:59` után):
```dockerfile
COPY --from=onnx-export --chown=appuser:appgroup /models ./models
ENV HUBERT_ONNX_PATH=/app/models/hubert_fp32.onnx \
    HUBERT_VOCAB_PATH=/app/models/vocab.txt
```

> **Miért a build-stage és nem git LFS:** így a `.onnx` **determinisztikusan a pinelt `HUBERT_REVISION`-ből** származik, nincs esély rá, hogy egy elfelejtett artifact és a config szétcsússzon. A `--verify` + `sha256sum -c` miatt egy elcsúszott export **build-hibát** okoz, nem csendes minőségromlást. Cserébe a build lassabb (~10 s export + a torch telepítése az eldobott stage-ben).

**`Dockerfile.prod:96`** — RAM-függő (9.1): a mért 594,5 MB/worker mellett a `--workers 2` ≈ 1,19 GB. Ha a plan ezt nem bírja: `--workers 1 --threads 4`.

**ORT szálkezelés** — kötelező explicit beállítás, különben 2 worker × összes mag túlfoglal:
```python
so = ort.SessionOptions()
so.intra_op_num_threads = int(settings.EMBEDDING_ORT_THREADS or 2)
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
```

### 7.1c Az A opció (FALLBACK) konkrét diffje

**`backend/requirements.prod.txt`** — a `:46-51` kommentblokk helyére:
```
# AI/ML - Local Hungarian embeddings (huBERT)
# CPU-only torch: 178 MB wheel / 648 MB installed, ZERO nvidia-* deps.
# The old "~2GB" note referred to the default PyPI (CUDA) wheel — NOT this one.
# Requires: --extra-index-url https://download.pytorch.org/whl/cpu  (see Dockerfile.prod)
torch==2.2.0+cpu
transformers==4.37.2
# NOTE: sentence-transformers is deliberately NOT installed — the project
# implements its own mean-pooling (embedding_service.py:291-312); it would only
# add ~56 MB of scipy/sklearn/Pillow.
# NOTE: spacy/huspacy is deliberately NOT installed — adding it would change
# preprocess_hungarian() from a no-op to real lemmatization, which would put
# queries in a DIFFERENT text space than the indexed vectors. Requires a reindex.
```

**`backend/Dockerfile.prod`** — a builder stage (`:27-31`) kiegészítése:
```dockerfile
COPY requirements.prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.prod.txt

# Bake the huBERT weights into the image so production never downloads from
# HuggingFace at runtime (no HF_HOME exists in this repo today - verified).
ARG HUBERT_REVISION=main
ENV HF_HOME=/opt/hf-cache
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
import os; r=os.environ.get('HUBERT_REVISION_ARG','main'); \
AutoTokenizer.from_pretrained('SZTAKI-HLT/hubert-base-cc', revision=r, use_fast=True); \
AutoModel.from_pretrained('SZTAKI-HLT/hubert-base-cc', revision=r)"
```
majd a production stage-ben (`:59` után):
```dockerfile
COPY --from=builder /opt/hf-cache /opt/hf-cache
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1          # fail loudly instead of silently hitting the network
```

> A `HF_HUB_OFFLINE=1` szándékos: ha a modell valamiért nincs az image-ben, a szolgáltatás **hangosan** hasít, nem csendben letölt 440 MB-ot az első user kérése közben.

**`Dockerfile.prod:96`** — RAM-függő (9.1):
- ha bőven van RAM: marad `--workers 2`
- ha szűk: `--workers 1` + `--threads 4` (a `_thread_pool` `embedding_service.py:57` amúgy is 4 workeres)

**`backend/app/core/config.py`** — új beállítás a `:182` után:
```python
# "auto" | "torch" | "onnx" | "disabled". Never silently falls back to zero vectors.
EMBEDDING_BACKEND: str = "auto"
```

### 7.2 Az `embedding_service.py` átalakítása — a publikus interfész NEM változik

**Kötelező invariáns:** `embed_text_async(text, preprocess, use_cache)`, `embed_batch_async(...)`, `get_embedding_service()`, `embed_text()`, `embed_batch()` szignatúrája és szemantikája **változatlan**. Csak a belső inference-út cserélődik.

Vázlat:

```python
# embedding_service.py — a lazy import blokk (jelenlegi :27-42) helyére

class EmbeddingUnavailableError(RuntimeError):
    """Az embedding backend nem elérhető. SOHA nem szabad nullvektorral elfedni."""


def _select_backend() -> "EmbeddingBackend":
    """
    Sorrend: explicit env override -> ONNX (ha van .onnx + onnxruntime) -> torch -> hiba.
    Egyik ág sem ad vissza nullvektort.
    """
    mode = settings.EMBEDDING_BACKEND  # "auto" | "onnx" | "torch"
    ...
    raise EmbeddingUnavailableError(
        "No embedding backend available (onnxruntime/torch missing). "
        "Semantic search is DISABLED — refusing to emit a zero vector."
    )
```

A backendek egységes szerződése — **csak a nyers forward pass**, a pooling/normalizálás közös marad:

```python
class EmbeddingBackend(Protocol):
    def forward(self, encoded: dict) -> np.ndarray:
        """(batch, seq, 768) last_hidden_state float32."""
```

Az ONNX backend:

```python
outputs = self._session.run(
    ["last_hidden_state"],
    {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
)
last_hidden = outputs[0]                       # (B, S, 768) float32
```

A **közös** pooling — pontosan a jelenlegi `_mean_pooling` (`:291-312`) numpy-változata:

```python
def _mean_pool_l2(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    mask = attention_mask[..., None].astype(np.float32)          # :306 megfelelője
    summed = (last_hidden * mask).sum(axis=1)                    # :309
    counts = np.clip(mask.sum(axis=1), 1e-9, None)               # :310  (clamp min=1e-9)
    pooled = summed / counts                                     # :312
    norms = np.clip(np.linalg.norm(pooled, ord=2, axis=1, keepdims=True), 1e-12, None)
    return pooled / norms                                        # :395  (L2, p=2, dim=1)
```

> **A `clamp(min=1e-9)` és az `ord=2, axis=1` pontos átvétele nem stílus kérdése — ez az embedding-tér identitása.**

**A tokenizálás — a B opcióban `transformers` NÉLKÜL** (6.3; a `transformers` import behúzná a torchot):

```python
from tokenizers import BertWordPieceTokenizer

# config.json: do_lower_case == false  -> lowercase=False is MANDATORY.
_tok = BertWordPieceTokenizer(settings.HUBERT_VOCAB_PATH, lowercase=False)
_tok.enable_truncation(max_length=512)          # == truncation=True, max_length=512
_tok.enable_padding()                            # == padding=True (longest-in-batch)

enc = _tok.encode_batch(texts)
input_ids      = np.array([e.ids for e in enc], dtype=np.int64)
attention_mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
token_type_ids = np.array([e.type_ids for e in enc], dtype=np.int64)
```

> **Mérve:** ez a tokenizer a 13 magyar tesztszöveg mindegyikére **azonos token ID-kat** ad, mint az `AutoTokenizer.from_pretrained(..., use_fast=True)` (4.7.5).
>
> Az **A opcióban** ugyanez marad `AutoTokenizer`-rel, csak `return_tensors="np"`:
> ```python
> encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="np")
> ```

### 7.3 Az ONNX export (`scripts/export_hubert_onnx.py`)

```
optimum-cli export onnx \
  --model SZTAKI-HLT/hubert-base-cc \
  --task feature-extraction \
  --opset 14 \
  --framework pt \
  <out_dir>
```
Vagy `torch.onnx.export`-tal, `dynamic_axes={"input_ids": {0:"batch",1:"seq"}, ...}`.

**Kötelező elemek:**
- a `HUBERT_REVISION`-t **commit SHA-ra pinelni** (jelenleg `"main"`, `config.py:181`) és az exportnál használni — ez egyben megoldja a régóta nyitott H4 auditfindingot (`tasks/audit_sprint13_master.md:43`)
- az export után **azonnal lefuttatni a 8.2 + 8.3 verifikációt**, a script hasítson, ha nem megy át
- a `.onnx` SHA256-ját kiírni; a Dockerfile ellenőrizze

### 7.4 Rollout

1. **Redis embedding cache flush** *(kötelező!)* — a mérgezett nullvektorok TTL-je 1 óra (`redis_cache.py:527`, `CacheTTL.EMBEDDINGS=3600`). Deploy előtt vagy után azonnal: `SCAN` + `DEL` a `embed:*` prefixre. Enélkül a deploy utáni első órában a felhasználók továbbra is nullvektort kapnak, és azt hihetjük, a javítás nem működött.
2. Deploy staging/preview környezetre, ha van; különben alacsony forgalmú időablakban prodra.
3. A `/api/v1/health/detailed` mutassa: `embedding: "ok"`, backend név, betöltési idő.
4. Élő ellenőrzés (7.6).

### 7.5 Rollback

| Kiváltó | Lépés | Idő |
|---------|-------|-----|
| OOM / restart loop (`railway.toml:9-10` max 3 retry után halott) | Railway rollback az előző deploymentre | ~1–2 perc |
| Az embedding rossz eredményt ad, de a service él | `EMBEDDING_BACKEND=disabled` env → a szemantikus ág **explicit degraded** módba megy (üres, de **jelzett** kontextussal), a lexikai + Neo4j út tovább szolgál | ~30 s (env change + restart) |
| Boot lassulás | `--workers 1` a `Dockerfile.prod:96`-ban | 1 deploy |

**Kulcs:** a `EMBEDDING_BACKEND=disabled` út **NEM** a régi csendes nullvektor. A rendszer ilyenkor is *tudja és jelzi*, hogy degradált. Ez a különbség.

> **FIGYELEM — a CD nem véd meg:** a `cd.yml:265-317` smoke-tesztek `continue-on-error: true`-val futnak, és teljesen kimaradnak, ha a `vars.API_URL` nincs beállítva. A `cd.yml:362-456` "rollback" job pedig **csak egy GitHub issue-t nyit** ("Manual intervention required for rollback"). Azaz **egy törött deploy nem bukik el automatikusan, és nem áll vissza magától.** A 7.6 élő ellenőrzést emiatt **kézzel, a deploy után azonnal** kell elvégezni.

### 7.6 Hogyan bizonyítjuk élőben, hogy működik

1. **Norma-ellenőrzés a health endpointon**
   `GET /api/v1/health/detailed` → `embedding.status == "ok"`, `embedding.self_test_norm ≈ 1.0`
2. **Szemantikus DTC keresés magyar tünettel** (nem DTC-kód alakú, hogy a `is_code_query` ág ne vigye el — `dtc_codes.py:443-447`)
   `GET /api/v1/dtc/search?query=rángat a motor gyorsításkor&use_semantic=true`
   → nem üres, és a logban ott a már meglévő telemetria (`dtc_codes.py:515-519`):
   `semantic dtc search: collection=autocognitix type=dtc hits=N pg_matched=M` — **`hits` > 0** kell.
3. **A teljes diagnózis**
   `POST /api/v1/diagnosis/analyze` valós tünetszöveggel → a válasz `sources` / `similar_complaints` mezői **nem üresek**, és `used_fallback == false`.
4. **Negatív kontroll:** értelmetlen query (pl. `"qqqq zzzz"`) → **kevés vagy nulla** találat a `score_threshold` felett. Ha erre is tele van a találati lista, valami baj van a szűréssel.
5. **Log-alapú:** deploy után 24 óra alatt **nulla** `"torch not available - returning zero vector"` üzenet.

---

## 8. Verifikációs eljárás — az embedding-tér kompatibilitás bizonyítása

> **Ez a legfontosabb kapu. A shipping ELŐTT kell lefuttatni, izolált környezetben, a repo módosítása nélkül.**

### 8.1 A0 — a modell-revízió ellenőrzése (mindkét opciónál kötelező)

Mivel a `HUBERT_REVISION="main"` (`config.py:181`), előbb tisztázni kell, hogy a **ma letöltött** `hubert-base-cc` ugyanaz-e, mint amivel a 54k vektor készült.

Eljárás:
1. Töltsd le a modellt, jegyezd fel a HF commit SHA-t.
2. Vegyél ki a Qdrant `autocognitix` collection-ből 20 pontot **a vektorukkal együtt** (`with_vectors=True`), és a payloadjukból **rekonstruáld pontosan az indexelt szöveget** az `index_qdrant_hubert.py` formulái szerint:
   - DTC: `f"{code}: {description} - {category} {subcategory}"` (`:188`)
   - complaint: `f"{make} {model} {year} - {component}: {description}"` (`:285`)
   - recall: `f"{make} {model} {year} - {component}. {summary} Consequence: {consequence} Remedy: {remedy}"` (`:390`)
3. Embedeld ezeket a mai torch úttal, és számolj cosine-t a Qdrantból visszaolvasott vektorral.

**Átmenő küszöb: min cosine > 0.999 mind a 20 mintán.**

> **Ha ez megbukik, a probléma NAGYOBB, mint a torch hiánya:** a 54k vektor egy másik modell-verzióval készült, és **teljes reindex kell** — ami viszont megnyitja az F opciót (modellváltás), mert ha úgyis reindexelünk, a modellválasztás újra asztalra kerül.
>
> Megjegyzés: a payloadok csonkoltak (`description[:1000]`, `summary[:1000]`), az eredeti embedding viszont a csonkolatlan (max 8000 karakteres) szövegből készült. Ezért **elsősorban a DTC típusú pontokat használd** a rekonstrukcióhoz — azoknak a payloadja teljes.

### 8.2 A1 — torch ↔ ONNX ekvivalencia  *(EGYSZER MÁR LEFUTOTT — újrafuttatandó a pinelt env-ben)*

Legalább 12 reprezentatív magyar szövegen (rövid tünet, hosszú panasz, 1-tokenes, DTC-formátumú, ~400 tokenes):

```
cos = dot(v_torch, v_onnx)          # mindkettő L2-normalizált -> a dot a cosine
```

| Metrika | **Átmenő küszöb** | **Mért (4.7.2)** | |
|---------|-------------------|------------------|---|
| min cosine (torch vs onnx-fp32) | **> 0,9999** | **0,9999997616** | ✅ |
| max abs elemenkénti eltérés | < 1e-4 | **1,788e-07** | ✅ |
| a norma mindkét oldalon | \|norm − 1.0\| < 1e-5 | teljesült | ✅ |
| determinizmus (30 futás) | 1 különböző kimenet | **1 / 30** | ✅ |

> **Miért kell mégis újrafuttatni:** a 4.7.2 mérés `transformers 5.14.1` + `sdpa` attention mellett készült, a projekt viszont `4.37.2`-t pinel (`requirements.txt:46`). A várható eltérés ~1e-6 (bőven a küszöb felett), de **ez BECSLÉS** — a 7.0 lépésben a projekt pinelt környezetében kell reprodukálni.

### 8.3 A2 — rangsor-ekvivalencia *(ez dönt, nem a nyers cosine)* — **LEFUTOTT, ÁTMENT**

1. Építs egy ~40 mondatos magyar autós korpuszt, embedeld **torch-csal** (ez szimulálja a már indexelt 54k-t).
2. Futtass 6 query-t, embedelve (a) torch-csal, (b) ONNX-szel.
3. Hasonlítsd a **top-5 sorrendet**.

| Metrika | **Átmenő küszöb** | **Mért fp32** | **Mért int8** |
|---------|-------------------|---------------|---------------|
| top-5 sorrend azonos | **6/6 query-re** | **6/6** ✅ | 0/6 ❌ |
| top-1 azonos | 6/6 | 6/6 ✅ | 4/6 ❌ |
| score-eltérés a top-5-ben | < 1e-3 | **2,384e-07** ✅ | 4,372e-02 ❌ |

> **Kontextus, ami ezt élessé teszi:** a mért korpuszban a torch top1–top2 margók **0,0004–0,0127** között vannak. Az fp32 eltérése (2,4e-07) ennek **1/1600 – 1/50 000-e** → biztonságos. Az int8 eltérése (4,4e-02) a margó **3–100×-osa** → garantáltan átrendezi a sorrendet.

### 8.4 A3 — int8 döntés — **LEFUTOTT, ELVETVE**

Ugyanez az A1+A2, de int8-cal. A szabály az volt: *"ha a top-5 sorrend akár egyetlen query-n is eltér → int8 ELVETVE"*. **Mind a 6 query-n eltért**, ráadásul a modell **nemdeterminisztikus** default szálszámmal (4.7.3). **Lezárva: int8 nem szállítható.** Ha valaki később mégis felveti, ez a szekció a válasz.

### 8.5 A4 — end-to-end Qdrant smoke (staging)

Élő Qdrant ellen, olvasás-only:
- ugyanaz a query a régi (torch, lokálisan) és az új (prod backend) úton
- a top-10 DTC kódok halmaza **legalább 9/10-ben egyezik**

### 8.6 Regressziós tesztek (a repóba)

```python
def test_embedding_never_returns_zero_vector_when_backend_missing():
    """A nullvektor-fallback TILOS: hiányzó backend -> EmbeddingUnavailableError."""

def test_qdrant_search_rejects_zero_query_vector():
    """qdrant_client.search() ValueError-t dob nulla normájú query vektorra."""

def test_rag_uses_unified_collection():
    """rag_service NEM hivatkozhat a dtc_embeddings_hu / symptom_embeddings_hu nevekre."""

def test_pooling_matches_reference_vectors():
    """Befagyasztott (text -> 768-dim) referencia vektorok; cosine > 0.9999."""
```

Az utolsó a legértékesebb: **fagyassz be 5 referencia vektort a repóba** (JSON fixture, ~15 KB). Ez örökre megvédi az embedding-teret bármely jövőbeli refaktortól, modellfrissítéstől, backend-cserétől.

---

## 9. Nyitott kérdések — CSAK a felhasználó/operátor tudja megválaszolni

| # | Kérdés | Miért kritikus |
|---|--------|----------------|
| **9.1** | **Mekkora a Railway backend service RAM limitje / melyik plan fut?** A repóban ez **sehol nincs dokumentálva** — a `docs/DEPLOYMENT.md:227-231` 2G/1G értékei docker-compose limitek, nem Railway-limitek; a `docs/RAILWAY_DEPLOYMENT.md:247-257` csak Free ($5 kredit) / Hobby ($5/hó) árat említ. | A B/A választást már nem ez dönti el (a B mindkét irányban jobb), de a `--workers` beállításához kell: a mért peak RSS **594,5 MB (ONNX)** / **802,5 MB (torch)** workerenként → 2 workerrel **1,19 GB** ill. **1,61 GB**. Egy pillantás a Railway dashboard Metrics fülére megválaszolja. |
| **9.2** | **Hány pont van ténylegesen a `autocognitix` collectionben, és léteznek-e még a `*_hu` collectionök?** A repó ellentmond: `CLAUDE.md` 35 000+, a feladatkiírás ~54 652, `tasks/todo.md:494-497` szerint `dtc_embeddings_hu`-ban 3 579, a `qdrant_client.py:311` szerint viszont "(empty)". | Az R2 javítás (collection-drift) formája ettől függ. Egy `GET /collections` a Qdrant Cloudon eldönti. Kapcsolódó: a "751K complaint" szám **sehol nem szerepel a repóban** (a `data/nhtsa/complaints/` könyvtár üres/nem létezik ebben a checkoutban), a roadmap 170 000+ célról beszél (`tasks/roadmap_v1_to_q2_2026.md:16`) ~8 GPU-órás reindex-becsléssel (`:108`). Ha a D/F opció (teljes reindex) szóba kerül, **ez a szám a költség fő hajtóereje**, és tisztázni kell. |
| **9.3** | **Tudjuk-e, melyik HF commit SHA-val készültek a meglévő vektorok?** (a `HUBERT_REVISION="main"` miatt nincs rögzítve) | Ha nem, a 8.1 rekonstrukciós teszt az egyetlen mód a kiderítésére. Bukás esetén teljes reindex. |
| **9.4** | **Elfogadható-e egy build-stage-ben generált 440 MB-os `.onnx` artifact az image-ben?** A javaslat szerint NEM git LFS, hanem egy eldobható Docker build-stage exportálja a pinelt revízióból, checksum-ellenőrzéssel (7.1b). Cserébe a build lassabb (a torch telepítése az eldobott stage-ben). | **Ez a B opció egyetlen nem-technikai előfeltétele.** Ha nem vállalható, a fallback az A (7.1c). |
| **9.5** | **A szemantikus keresés stratégiai prioritás-e, vagy elhagyható?** | Ha elhagyható, az E opció + a Qdrant leépítése a becsületes válasz, és megspórolunk egy Qdrant Cloud előfizetést. Ha nem, akkor a jelenlegi állapot (fizetünk érte, de nullvektort küldünk bele) tarthatatlan. |
| **9.6** | **Van-e (vagy készíthető-e) egy 50–100 elemű magyar query→várt-DTC kiértékelő halmaz?** | Enélkül semmilyen retrieval-minőség állítás nem bizonyítható, és az F opció (modellváltás) nem értékelhető felelősen. |

---

## 10. Összefoglaló javaslat

1. **Most azonnal, kódtól függetlenül (30 perc):** válaszold meg a **9.1** (Railway RAM) és **9.2** (Qdrant collection-ok) kérdést, és futtasd le a **8.1** revízió-rekonstrukciós tesztet. Ez a három adat eldönti az egész irányt — és a 8.1 bukása esetén az egész terv változik.
2. **Fázis 1 (fél nap) — R3 safety guard.** Nullvektor tilalom (`EmbeddingUnavailableError`), Qdrant query-vektor guard, `/health/detailed` embedding-próba, explicit `degraded` flag a diagnózis-válaszban. **Ez önmagában is szállítható, és a legmagasabb megtérülésű változtatás:** megakadályozza, hogy a következő hasonló hiba is hónapokig rejtve maradjon. Mellékhatásként **a hiba azonnal láthatóvá válik prodban** — ami kellemetlen, de helyes.
3. **Fázis 2 (fél nap) — R2 collection-drift.** `rag_service.py:794/801` átállítása a unified `autocognitix` collectionre `type` diszkriminátorral (a `search_unified()` már létezik, `qdrant_client.py:253`). **Figyelem a leképezésre:** az `autocognitix` collection **csak három payload-típust ismer** — `"dtc"`, `"complaint"`, `"recall"` (`index_qdrant_hubert.py:210, :313, :419`). **Nincs `"symptom"` típus.** A `rag_service.py:799-805` "symptom search" ága ezért `type_="complaint"`-re képezendő (az NHTSA panaszszövegek a tünetleírás legközelebbi megfelelői) — vagy tudatosan megszüntetendő. Egyúttal a `vehicle_make` filter kivezetése vagy javítása (`:803`): a unified payload kulcsa `"make"`, nem `"vehicle_make"` (`index_qdrant_hubert.py:313-315`), és **nyers NHTSA all-caps** ("VOLKSWAGEN") — tehát a mai filter itt is garantáltan 0 találatot adna.
4. **Fázis 3 (1–2 nap) — R1 embedding backend.** **ONNX Runtime fp32** (7.1b): build-stage export a pinelt revízióból, `onnxruntime` + `tokenizers` a prod image-be, **`transformers`/`torch` nélkül** (6.3), a backend-absztrakcióval (7.2) úgy megírva, hogy a torch út fallbackként bekapcsolható maradjon (`EMBEDDING_BACKEND=torch`). Ha a 7.0/3. újra-verifikáció megbukik → **A opció** (7.1c), ugyanazzal az absztrakcióval.
5. **Rollout:** Redis `embed:*` flush (7.4) → deploy → 7.6 élő ellenőrzés **kézzel** (a CD smoke-tesztek `continue-on-error`-osak).
6. **Backlog (sorrendben):** `HUBERT_REVISION` SHA-pin (H4 auditfinding, `tasks/audit_sprint13_master.md:43`) → `vehicle_make` payload-egységesítés + reindex → magyar retrieval kiértékelő halmaz (9.6) → és **csak ezután** az F opció (modellváltás) mérése.

> **Egy mondatban:** a szemantikus keresés azért néma, mert a prod image-ből kihagyott torch helyett a kód csendben nullvektort ad — a megoldás az **ONNX Runtime fp32**, ami mérésileg **bit-közeli azonos** embeddinget ad (cos ≥ 0,9999997, rangsor 6/6), így a ~54k indexelt vektor **érintetlen marad**, miközben az image 720 MB-tal, a memória 208 MB-tal kisebb és a query 1,7× gyorsabb. A javítás három részes (embedding + collection-drift + safety guard), és a legfontosabb rész **nem** az embedding visszahozása, hanem annak garantálása, hogy **legközelebb hangosan hasítson**.
