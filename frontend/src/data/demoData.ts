/**
 * Demo Data - P0300 Random/Multiple Cylinder Misfire Simulation
 *
 * Szimulált hibakód: P0300 (Több hengeres égéskimaradás)
 * Jármű: Volkswagen Golf VII 1.4 TSI (2018)
 * Árak: Valós magyar alkatrész webshopokból (Bárdi Autó, Uni Autó, AUTODOC)
 * Utolsó ár-frissítés: 2026-03-01
 */

import type { DiagnosisResponse, PartWithPrice, TotalCostEstimate } from '../services/api';

// =============================================================================
// Store-specific pricing for parts cards
// =============================================================================

export interface StorePricing {
  storeName: string;
  storeUrl: string;
  storeLogoColor: string;
  price: number;
  priceMax?: number;
  currency: string;
  inStock: boolean;
  deliveryDays: number;
  brand: string;
}

export interface DemoPartWithStores extends PartWithPrice {
  description: string;
  partNumber: string;
  oemNumber: string;
  stores: StorePricing[];
  isOem: boolean;
  qualityRating: number;
  compatibilityNote: string;
}

// =============================================================================
// Demo Parts with Real Hungarian Store Prices
// =============================================================================

export const demoParts: DemoPartWithStores[] = [
  {
    id: 'spark_plug_set',
    name: 'Gyújtógyertya készlet (4 db)',
    name_en: 'Spark Plug Set (4 pcs)',
    category: 'Gyújtás',
    price_range_min: 11560,
    price_range_max: 21960,
    labor_hours: 0.5,
    currency: 'HUF',
    description: 'NGK vagy BOSCH irídium gyújtógyertya szett a 1.4 TSI CZCA motorhoz. Az égéskimaradás leggyakoribb oka a kopott vagy hibás gyújtógyertya.',
    partNumber: 'NGK 95770 / BKUR6ET-10',
    oemNumber: '04E 905 612 C',
    isOem: false,
    qualityRating: 4.5,
    compatibilityNote: 'VW Golf VII 1.4 TSI (CZCA/CMBA motor) 2012-2020',
    stores: [
      {
        storeName: 'Bárdi Autó',
        storeUrl: 'https://www.bardiauto.hu',
        storeLogoColor: '#E31E24',
        price: 11890,
        priceMax: 15960,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 1,
        brand: 'NGK BKUR6ET-10',
      },
      {
        storeName: 'Unix Autó',
        storeUrl: 'https://www.unixauto.com',
        storeLogoColor: '#1A56DB',
        price: 13200,
        priceMax: 19960,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 1,
        brand: 'BOSCH FR5KPP332S',
      },
      {
        storeName: 'AUTODOC',
        storeUrl: 'https://www.autodoc.hu',
        storeLogoColor: '#FF6900',
        price: 15552,
        priceMax: 21960,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 3,
        brand: 'DENSO Iridium IK20TT',
      },
    ],
  },
  {
    id: 'ignition_coil',
    name: 'Gyújtótekercs',
    name_en: 'Ignition Coil',
    category: 'Gyújtás',
    price_range_min: 12900,
    price_range_max: 36034,
    labor_hours: 0.3,
    currency: 'HUF',
    description: 'Egyedi gyújtótekercs a közvetlen gyújtási rendszerhez. A hibás tekercs egyetlen henger égéskimaradását okozza, de a P0300 kódnál ajánlott mind a 4 ellenőrzése.',
    partNumber: 'BERU ZSE032',
    oemNumber: '04E 905 110 B',
    isOem: false,
    qualityRating: 4.2,
    compatibilityNote: 'VW/Audi/Skoda/Seat 1.4 TSI EA211 motorcsaládhoz',
    stores: [
      {
        storeName: 'Bárdi Autó',
        storeUrl: 'https://www.bardiauto.hu',
        storeLogoColor: '#E31E24',
        price: 12900,
        priceMax: 22500,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 1,
        brand: 'BERU ZSE032',
      },
      {
        storeName: 'Unix Autó',
        storeUrl: 'https://www.unixauto.com',
        storeLogoColor: '#1A56DB',
        price: 14500,
        priceMax: 28900,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 2,
        brand: 'HELLA 5DA 193 175-631',
      },
      {
        storeName: 'AUTODOC',
        storeUrl: 'https://www.autodoc.hu',
        storeLogoColor: '#FF6900',
        price: 14503,
        priceMax: 36034,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 3,
        brand: 'BOSCH 0 986 221 095',
      },
    ],
  },
  {
    id: 'air_filter',
    name: 'Levegőszűrő',
    name_en: 'Air Filter',
    category: 'Szűrők',
    price_range_min: 3490,
    price_range_max: 7200,
    labor_hours: 0.2,
    currency: 'HUF',
    description: 'A szennyezett levegőszűrő miatt a motor nem kap elég levegőt, ami egyenetlen égést és égéskimaradást okozhat. Egyszerűen cserélhető, érdemes a gyertyacsere mellett elvégezni.',
    partNumber: 'MANN C 27 009',
    oemNumber: '04E 129 620 D',
    isOem: false,
    qualityRating: 4.7,
    compatibilityNote: 'VW Golf VII / Skoda Octavia III 1.4 TSI (2012-2020)',
    stores: [
      {
        storeName: 'Bárdi Autó',
        storeUrl: 'https://www.bardiauto.hu',
        storeLogoColor: '#E31E24',
        price: 3490,
        priceMax: 5290,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 1,
        brand: 'FILTRON AP 183/3',
      },
      {
        storeName: 'Unix Autó',
        storeUrl: 'https://www.unixauto.com',
        storeLogoColor: '#1A56DB',
        price: 3890,
        priceMax: 6490,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 1,
        brand: 'MANN C 27 009',
      },
      {
        storeName: 'AUTODOC',
        storeUrl: 'https://www.autodoc.hu',
        storeLogoColor: '#FF6900',
        price: 4190,
        priceMax: 7200,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 3,
        brand: 'MAHLE LX 3525',
      },
    ],
  },
  {
    id: 'fuel_filter',
    name: 'Üzemanyagszűrő',
    name_en: 'Fuel Filter',
    category: 'Szűrők',
    price_range_min: 4590,
    price_range_max: 14500,
    labor_hours: 0.5,
    currency: 'HUF',
    description: 'Az eldugult üzemanyagszűrő nem megfelelő üzemanyag-ellátást okoz, ami különösen terhelés alatt jelentkezik égéskimaradásként. A Golf VII-ben a szűrő az üzemanyagtankban található.',
    partNumber: 'MANN WK 69/2',
    oemNumber: '04E 201 511 C',
    isOem: false,
    qualityRating: 4.3,
    compatibilityNote: 'VW Golf VII 1.4 TSI benzines modellek (2012-2020)',
    stores: [
      {
        storeName: 'Bárdi Autó',
        storeUrl: 'https://www.bardiauto.hu',
        storeLogoColor: '#E31E24',
        price: 4590,
        priceMax: 8900,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 1,
        brand: 'MANN WK 69/2',
      },
      {
        storeName: 'Unix Autó',
        storeUrl: 'https://www.unixauto.com',
        storeLogoColor: '#1A56DB',
        price: 5200,
        priceMax: 11500,
        currency: 'HUF',
        inStock: false,
        deliveryDays: 3,
        brand: 'BOSCH F 026 403 006',
      },
      {
        storeName: 'AUTODOC',
        storeUrl: 'https://www.autodoc.hu',
        storeLogoColor: '#FF6900',
        price: 5890,
        priceMax: 14500,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 4,
        brand: 'MAHLE KL 756',
      },
    ],
  },
  {
    id: 'fuel_injector',
    name: 'Befecskendező szelep (injektor)',
    name_en: 'Fuel Injector',
    category: 'Üzemanyag rendszer',
    price_range_min: 32900,
    price_range_max: 72000,
    labor_hours: 1.5,
    currency: 'HUF',
    description: 'A közvetlen befecskendezésű (FSI/TSI) injektor eltömődése vagy meghibásodása egyenetlen üzemanyag-elosztást okoz. A P0300 kódnál kompressziómérés után derül ki, melyik injektort érdemes cserélni.',
    partNumber: 'BOSCH 0 261 500 160',
    oemNumber: '04E 906 036 Q',
    isOem: false,
    qualityRating: 4.0,
    compatibilityNote: 'VW Golf VII 1.4 TSI közvetlen befecskendezésű (MPI+FSI) motor',
    stores: [
      {
        storeName: 'Bárdi Autó',
        storeUrl: 'https://www.bardiauto.hu',
        storeLogoColor: '#E31E24',
        price: 32900,
        priceMax: 55000,
        currency: 'HUF',
        inStock: false,
        deliveryDays: 3,
        brand: 'BOSCH 0 261 500 160',
      },
      {
        storeName: 'Unix Autó',
        storeUrl: 'https://www.unixauto.com',
        storeLogoColor: '#1A56DB',
        price: 35000,
        priceMax: 62000,
        currency: 'HUF',
        inStock: false,
        deliveryDays: 4,
        brand: 'VDO A2C59517083',
      },
      {
        storeName: 'AUTODOC',
        storeUrl: 'https://www.autodoc.hu',
        storeLogoColor: '#FF6900',
        price: 38900,
        priceMax: 72000,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 5,
        brand: 'DELPHI 28397897',
      },
    ],
  },
  {
    id: 'oxygen_sensor',
    name: 'Lambda szonda (előkatalikus)',
    name_en: 'Oxygen Sensor (Pre-Cat)',
    category: 'Kipufogó / Emissziók',
    price_range_min: 18500,
    price_range_max: 48000,
    labor_hours: 0.8,
    currency: 'HUF',
    description: 'A lambda szonda feladata a kipufogógázok oxigéntartalmának mérése. Hibás szonda esetén a motorvezérlő nem tudja optimalizálni a keveréket, ami égéskimaradáshoz vezet.',
    partNumber: 'BOSCH 0 258 010 032',
    oemNumber: '04E 906 262 R',
    isOem: false,
    qualityRating: 4.1,
    compatibilityNote: 'VW Golf VII 1.4 TSI – kipufogó előtti (Bank 1 Sensor 1)',
    stores: [
      {
        storeName: 'Bárdi Autó',
        storeUrl: 'https://www.bardiauto.hu',
        storeLogoColor: '#E31E24',
        price: 18500,
        priceMax: 35000,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 2,
        brand: 'NGK OZA851-EE3',
      },
      {
        storeName: 'Unix Autó',
        storeUrl: 'https://www.unixauto.com',
        storeLogoColor: '#1A56DB',
        price: 19900,
        priceMax: 42000,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 2,
        brand: 'BOSCH 0 258 010 032',
      },
      {
        storeName: 'AUTODOC',
        storeUrl: 'https://www.autodoc.hu',
        storeLogoColor: '#FF6900',
        price: 22500,
        priceMax: 48000,
        currency: 'HUF',
        inStock: true,
        deliveryDays: 3,
        brand: 'DENSO DOX-0120',
      },
    ],
  },
];

// =============================================================================
// Demo Diagnosis Response - Full P0300 Simulation
// =============================================================================

export const demoCostEstimate: TotalCostEstimate = {
  parts_min: 83940,
  parts_max: 199694,
  labor_min: 28800,
  labor_max: 45600,
  total_min: 112740,
  total_max: 245294,
  currency: 'HUF',
  estimated_hours: 3.8,
  difficulty: 'medium',
  disclaimer:
    'Az árak a Bárdi Autó, Unix Autó és AUTODOC webshopok 2026. márciusi kínálata alapján készültek. A tényleges költségek a szerviz munkadíjától és az alkatrész minőségétől függően eltérhetnek. Az összesítés az összes lehetséges alkatrész cseréjét tartalmazza – a tényleges javításhoz nem feltétlenül szükséges mindegyik.',
};

export const demoDiagnosisResponse: DiagnosisResponse = {
  id: 'demo-p0300-showcase',
  vehicle_make: 'Volkswagen',
  vehicle_model: 'Golf VII',
  vehicle_year: 2018,
  dtc_codes: ['P0300', 'P0301', 'P0304'],
  symptoms:
    'Reggelente a motor nagyon rángatózik hidegindításnál, kb. 1-2 percig tart amíg „összekapja magát". A check engine lámpa folyamatosan világít, és autópályán 130-nál néha villog is. A fogyasztás észrevehetően megnőtt (8-ról 10.5 literre ugrott városban). Alapjáraton egyenetlen a motor járása, rezeg az egész autó. A szerelő szerint kompressziós problémára gyanakszik, de a szomszédom azt mondta, hogy valószínűleg gyertyát kell cserélni.',
  probable_causes: [
    {
      title: 'Kopott vagy hibás gyújtógyertyák',
      description:
        'A 1.4 TSI motor nagy kompresszióviszonyú közvetlen befecskendezésű motor, amely rendkívül érzékeny a gyújtógyertya állapotára. A kopott elektródák nem képesek megfelelő szikrát adni, különösen hidegindításkor amikor a motor gazdagabb keveréket használ.',
      confidence: 0.92,
      related_dtc_codes: ['P0300', 'P0301', 'P0304'],
      components: ['Gyújtógyertya', 'Gyújtótekercs'],
    },
    {
      title: 'Hibás gyújtótekercs (1. vagy 4. henger)',
      description:
        'Az egyedi gyújtótekercsek (COP rendszer) bármelyike meghibásodhat. A P0301 és P0304 kódok az 1. és 4. henger specifikus problémájára utalnak. Méréssel (ohmméterrel) azonosítható a hibás tekercs.',
      confidence: 0.78,
      related_dtc_codes: ['P0301', 'P0304'],
      components: ['Gyújtótekercs', 'Gyújtótekercs csatlakozó'],
    },
    {
      title: 'Eltömődött üzemanyagszűrő vagy injektorprobléma',
      description:
        'A 98.000 km-es futásteljesítménynél az üzemanyagszűrő cseréje esedékes. Az eltömődött szűrő nyomásesést okoz a befecskendező rendszerben, ami egyenetlen üzemanyag-elosztáshoz vezet.',
      confidence: 0.45,
      related_dtc_codes: ['P0300'],
      components: ['Üzemanyagszűrő', 'Befecskendező szelep'],
    },
  ],
  recommended_repairs: [
    {
      title: '1. lépés: Gyújtógyertya csere (4 db)',
      description:
        'Cserélje ki mind a 4 gyújtógyertyát irídium típusra. A VW specifikáció szerint a 1.4 TSI motorhoz NGK BKUR6ET-10 vagy BOSCH FR5KPP332S típus ajánlott. Fontos: ne keverjen különböző típusú gyertyákat! A cserénél ellenőrizze az elektróda kopásának mértékét – ez visszajelzést ad a motor állapotáról.',
      estimated_cost_min: 11560,
      estimated_cost_max: 21960,
      estimated_cost_currency: 'HUF',
      difficulty: 'beginner',
      parts_needed: ['Gyújtógyertya készlet (4 db)', 'Gyújtógyertya aljzat zsír'],
      estimated_time_minutes: 30,
      tools_needed: [
        { name: 'Gyújtógyertya kulcs (16mm)', icon_hint: 'build' },
        { name: 'Nyomatékkulcs (20-30 Nm)', icon_hint: 'precision_manufacturing' },
        { name: 'Hézagmérő', icon_hint: 'straighten' },
      ],
      expert_tips: [
        'Meleg motornál SOHA ne csavarozzon gyertyát – az alumínium hengerfej könnyen menetesedik!',
      ],
      root_cause_explanation:
        'A gyújtógyertyák a TSI motorokban kb. 60.000 km-enként cserélendők. A 98.000 km-es futásnál már rég esedékes a csere.',
    },
    {
      title: '2. lépés: Gyújtótekercs ellenőrzése és szükség szerinti csere',
      description:
        'A P0301 és P0304 kódok az 1. és 4. henger specifikus problémájára utalnak. Cserélje meg a gyanús henger gyújtótekercsét egy másik hengeréével. Ha a hiba követi a tekercset, az a hibás. A VW kód: 04E 905 110 B.',
      estimated_cost_min: 12900,
      estimated_cost_max: 36034,
      estimated_cost_currency: 'HUF',
      difficulty: 'beginner',
      parts_needed: ['Gyújtótekercs (1-2 db)'],
      estimated_time_minutes: 20,
      tools_needed: [
        { name: 'Torx T30 csavarhúzó', icon_hint: 'build' },
        { name: 'Multiméter (ellenállás mérés)', icon_hint: 'electrical_services' },
      ],
      expert_tips: [
        'A tekercsek megcserélése a legegyszerűbb diagnosztikai módszer – ha a hiba követi a tekercset, megtaláltad a hibásat.',
      ],
      root_cause_explanation:
        'A gyújtótekercsek a VW 1.4 TSI motorokban ismert gyenge pont. 80.000 km felett gyakori a meghibásodás.',
    },
    {
      title: '3. lépés: Levegőszűrő csere',
      description:
        'A szennyezett levegőszűrő miatt a motor nem kapja meg az optimális levegőmennyiséget. Ez a keverékarány eltolódásához és égéskimaradáshoz vezet. A csere pár perc alatt elvégezhető.',
      estimated_cost_min: 3490,
      estimated_cost_max: 7200,
      estimated_cost_currency: 'HUF',
      difficulty: 'beginner',
      parts_needed: ['Levegőszűrő betét'],
      estimated_time_minutes: 10,
      tools_needed: [
        { name: 'Torx T25 csavarhúzó', icon_hint: 'build' },
      ],
      expert_tips: [
        'Tartsa a szűrőt fény felé – ha nem lát át rajta, ideje cserélni. Érdemes a gyertyacserével együtt elvégezni.',
      ],
      root_cause_explanation:
        'Az 1.4 TSI motor a szívósor kialakítása miatt érzékeny a levegőmennyiségre.',
    },
    {
      title: '4. lépés: Üzemanyagszűrő csere',
      description:
        'A 98.000 km-es futásnál az üzemanyagszűrő cseréje már esedékes (VW ajánlás: 60.000 km). Az eltömődött szűrő csökkentett nyomást okoz az injektoroknál, ami terhelés alatt égéskimaradáshoz vezet.',
      estimated_cost_min: 4590,
      estimated_cost_max: 14500,
      estimated_cost_currency: 'HUF',
      difficulty: 'intermediate',
      parts_needed: ['Üzemanyagszűrő', 'O-gyűrű készlet'],
      estimated_time_minutes: 30,
      tools_needed: [
        { name: 'Csőszorító fogó', icon_hint: 'plumbing' },
        { name: 'Torx T30 készlet', icon_hint: 'build' },
        { name: 'Biztonsági szemüveg', icon_hint: 'visibility' },
      ],
      expert_tips: [
        'FIGYELEM: A csere előtt engedje le a rendszer nyomását! A benzin tűzveszélyes – tartsa kéznél a tűzoltó készüléket.',
      ],
      root_cause_explanation:
        'A szűrő 98.000 km-en már 38.000 km-rel túl van a VW által javasolt csereidőn.',
    },
    {
      title: '5. lépés: Diagnosztika – Lambda szonda és injektor vizsgálat',
      description:
        'Ha az előző lépések után a P0300 továbbra is fennáll, szükséges a lambda szonda jelének ellenőrzése OBD diagnosztikával (oszcilloszkóp ajánlott), valamint az injektorok szóróképének vizsgálata. Ez már szerviz szintű diagnosztikát igényel.',
      estimated_cost_min: 18500,
      estimated_cost_max: 120000,
      estimated_cost_currency: 'HUF',
      difficulty: 'professional',
      parts_needed: ['Lambda szonda (szükség esetén)', 'Befecskendező szelep (szükség esetén)'],
      estimated_time_minutes: 90,
      tools_needed: [
        { name: 'OBD-II diagnosztikai eszköz', icon_hint: 'bluetooth_connected' },
        { name: 'Oszcilloszkóp', icon_hint: 'timeline' },
        { name: 'Üzemanyag-nyomásmérő', icon_hint: 'speed' },
        { name: 'Injektortisztító folyadék', icon_hint: 'science' },
      ],
      expert_tips: [
        'Az injektorok tisztítása ultrahangos módszerrel sokszor megoldja a problémát csere nélkül is – kérje meg a szervizben.',
      ],
      root_cause_explanation:
        'Ha a gyújtás és szűrők cseréje nem oldja meg a problémát, mélyebb rendszerszintű vizsgálat szükséges.',
    },
  ],
  confidence_score: 0.87,
  sources: [
    {
      type: 'database',
      title: 'VW Golf VII 1.4 TSI szerviz adatbázis – P0300 hibakód elemzés',

      relevance_score: 0.95,
    },
    {
      type: 'tsb',
      title: 'VW Technikai Szerviz Közlemény TPI 2029411 – 1.4 TSI égéskimaradás',

      relevance_score: 0.88,
    },
    {
      type: 'forum',
      title: 'Golf7.hu fórum – P0300/P0301 tapasztalatok (412 hozzászólás)',

      relevance_score: 0.72,
    },
    {
      type: 'manual',
      title: 'ELSA Workshop Manual – EA211 1.4 TSI Gyújtási rendszer',

      relevance_score: 0.91,
    },
  ],
  created_at: '2026-03-01T10:30:00.000Z',
  parts_with_prices: demoParts.map((p) => ({
    id: p.id,
    name: p.name,
    name_en: p.name_en,
    category: p.category,
    price_range_min: p.price_range_min,
    price_range_max: p.price_range_max,
    labor_hours: p.labor_hours,
    currency: p.currency,
  })),
  total_cost_estimate: demoCostEstimate,
  root_cause_analysis:
    'A Volkswagen Golf VII 1.4 TSI (CZCA motorkód) motorvezérlő egysége (ECU) P0300, P0301 és P0304 hibakódokat regisztrált, amelyek több hengeres égéskimaradásra utalnak. A főtengely pozíció szenzor (CKP) által észlelt szöggyorsulás-ingadozás egyértelműen az 1. és 4. hengernél a legerőteljesebb, ami az 1. és 4. hengernél (a sor két szélső hengerénél) a legerőteljesebb.\n\nAz elemzés 87%-os konfidenciával állapítja meg, hogy a probléma gyökérokát a gyújtási rendszer kopása képezi, elsősorban:\n\n• **Gyújtógyertyák** (98.000 km-nél a VW 60.000 km-es ajánláshoz képest 63%-kal túlfutott)\n• **Gyújtótekercsek** (az 1. és 4. henger egyedi COP tekercseinél megnövekedett belső ellenállás)\n• **Levegő- és üzemanyagszűrő** (a szűrő kapacitás csökkenése rontja a keverék-arányokat)\n\nA hiba progresszív jellegű: hidegindításkor a legkifejezettebb (sűrűbb keverék → nagyobb szikraigény), és terhelés alatt (autópálya tempó) is megjelenik, ahol a villódzó check engine lámpa (MI villogás) komoly katalyzátor-károsodás kockázatára figyelmeztet.\n\n⚠️ **Azonnali beavatkozás ajánlott** a katalyzátor védelmében – az elégetlen üzemanyag túlhevítheti és visszafordíthatatlanul károsíthatja a katalizátort, amelynek cseréje 150.000-350.000 Ft.',
};

// =============================================================================
// Vehicle Image for Demo
// =============================================================================

export const demoVehicleImage =
  'https://images.unsplash.com/photo-1541899481282-d53bffe3c35d?w=800&h=600&fit=crop';

// =============================================================================
// Demo Vehicle Details
// =============================================================================

export const demoVehicleDetails = {
  licensePlate: 'ABC-123',
  vin: 'WVWZZZAUZJW123456',
  mileage: '98.420 km',
  engineInfo: '1.4 TSI 125 LE (CZCA)',
  fuelType: 'Benzin',
  transmission: 'DSG-7 automata',
  color: 'Indium szürke metál',
};
