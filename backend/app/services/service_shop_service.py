"""
Service Shop Service - Hungarian auto repair shop search and listing.

Static data MVP with 30 curated shops across Hungary.
Supports filtering by region, vehicle make, service type,
and distance-based sorting via Haversine formula.

Author: AutoCognitix Team
"""

import math
from typing import Any, Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Static Data - Hungarian Regions (19 counties + Budapest)
# =============================================================================

REGIONS: List[Dict[str, Any]] = [
    {"id": "budapest", "name": "Budapest", "county": "Budapest", "lat": 47.4979, "lng": 19.0402},
    {"id": "pest", "name": "Pest megye", "county": "Pest", "lat": 47.45, "lng": 19.25},
    {"id": "baranya", "name": "Baranya megye", "county": "Baranya", "lat": 46.07, "lng": 18.23},
    {
        "id": "bacs_kiskun",
        "name": "Bács-Kiskun megye",
        "county": "Bács-Kiskun",
        "lat": 46.60,
        "lng": 19.66,
    },
    {"id": "bekes", "name": "Békés megye", "county": "Békés", "lat": 46.67, "lng": 21.08},
    {
        "id": "borsod",
        "name": "Borsod-Abaúj-Zemplén megye",
        "county": "Borsod-Abaúj-Zemplén",
        "lat": 48.10,
        "lng": 20.78,
    },
    {
        "id": "csongrad",
        "name": "Csongrád-Csanád megye",
        "county": "Csongrád-Csanád",
        "lat": 46.42,
        "lng": 20.42,
    },
    {"id": "fejer", "name": "Fejér megye", "county": "Fejér", "lat": 47.19, "lng": 18.41},
    {
        "id": "gyor_moson_sopron",
        "name": "Győr-Moson-Sopron megye",
        "county": "Győr-Moson-Sopron",
        "lat": 47.68,
        "lng": 17.63,
    },
    {
        "id": "hajdu_bihar",
        "name": "Hajdú-Bihar megye",
        "county": "Hajdú-Bihar",
        "lat": 47.53,
        "lng": 21.63,
    },
    {"id": "heves", "name": "Heves megye", "county": "Heves", "lat": 47.90, "lng": 20.37},
    {
        "id": "jasz_nagykun_szolnok",
        "name": "Jász-Nagykun-Szolnok megye",
        "county": "Jász-Nagykun-Szolnok",
        "lat": 47.17,
        "lng": 20.18,
    },
    {
        "id": "komarom_esztergom",
        "name": "Komárom-Esztergom megye",
        "county": "Komárom-Esztergom",
        "lat": 47.73,
        "lng": 18.64,
    },
    {"id": "nograd", "name": "Nógrád megye", "county": "Nógrád", "lat": 47.90, "lng": 19.50},
    {"id": "somogy", "name": "Somogy megye", "county": "Somogy", "lat": 46.35, "lng": 17.80},
    {
        "id": "szabolcs",
        "name": "Szabolcs-Szatmár-Bereg megye",
        "county": "Szabolcs-Szatmár-Bereg",
        "lat": 48.10,
        "lng": 21.95,
    },
    {"id": "tolna", "name": "Tolna megye", "county": "Tolna", "lat": 46.43, "lng": 18.70},
    {"id": "vas", "name": "Vas megye", "county": "Vas", "lat": 47.23, "lng": 16.62},
    {"id": "veszprem", "name": "Veszprém megye", "county": "Veszprém", "lat": 47.09, "lng": 17.91},
    {"id": "zala", "name": "Zala megye", "county": "Zala", "lat": 46.84, "lng": 16.84},
]


# =============================================================================
# Static Data - 30 Hungarian Auto Repair Shops
# =============================================================================

SHOPS: List[Dict[str, Any]] = [
    # --- Budapest (10 shops) ---
    {
        "id": "shop_01",
        "name": "AutoMester Budapest",
        "address": "Váci út 45",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.5136,
        "lng": 19.0568,
        "phone": "+36 1 234 5678",
        "website": None,
        "rating": 4.5,
        "review_count": 127,
        "specializations": ["general", "german"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "suspension"],
        "opening_hours": "H-P: 8:00-17:00, Szo: 8:00-12:00",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_02",
        "name": "German Car Service Kft.",
        "address": "Fehérvári út 112",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.4567,
        "lng": 19.0345,
        "phone": "+36 1 345 6789",
        "website": "https://germancar.hu",
        "rating": 4.8,
        "review_count": 203,
        "specializations": ["german"],
        "accepted_makes": ["BMW", "Audi", "Mercedes", "Volkswagen"],
        "price_level": 3,
        "services": ["diagnosis", "engine", "electronics", "bodywork"],
        "opening_hours": "H-P: 7:30-18:00",
        "has_inspection": True,
        "has_courtesy_car": True,
    },
    {
        "id": "shop_03",
        "name": "Rapid Szerviz",
        "address": "Kerepesi út 78",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.4891,
        "lng": 19.1123,
        "phone": "+36 1 456 7890",
        "website": None,
        "rating": 4.2,
        "review_count": 89,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["brakes", "suspension", "oil_change", "tires"],
        "opening_hours": "H-P: 8:00-16:30",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_04",
        "name": "Autopálya Szervizközpont",
        "address": "Budaörsi út 30",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.4623,
        "lng": 18.9812,
        "phone": "+36 1 567 8901",
        "website": "https://autopalya-szerviz.hu",
        "rating": 4.6,
        "review_count": 156,
        "specializations": ["general", "japanese"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "air_conditioning"],
        "opening_hours": "H-P: 7:00-19:00, Szo: 8:00-14:00",
        "has_inspection": True,
        "has_courtesy_car": True,
    },
    {
        "id": "shop_05",
        "name": "ElektroMotor Kft.",
        "address": "Hungária körút 56",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.5023,
        "lng": 19.0967,
        "phone": "+36 1 678 9012",
        "website": None,
        "rating": 4.3,
        "review_count": 72,
        "specializations": ["electric", "diagnosis"],
        "accepted_makes": ["Tesla", "Nissan", "BMW"],
        "price_level": 3,
        "services": ["diagnosis", "electronics", "battery", "charging"],
        "opening_hours": "H-P: 8:00-17:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_06",
        "name": "Budai Karosszéria",
        "address": "Csörsz utca 18",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.4789,
        "lng": 19.0156,
        "phone": "+36 1 789 0123",
        "website": None,
        "rating": 4.0,
        "review_count": 54,
        "specializations": ["bodywork"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["bodywork", "painting", "dent_repair"],
        "opening_hours": "H-P: 8:00-16:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_07",
        "name": "Óbuda AutoCenter",
        "address": "Bécsi út 154",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.5456,
        "lng": 19.0234,
        "phone": "+36 1 890 1234",
        "website": "https://obuda-autocenter.hu",
        "rating": 4.4,
        "review_count": 98,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "oil_change", "tires"],
        "opening_hours": "H-P: 7:30-17:30, Szo: 8:00-12:00",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_08",
        "name": "Suzuki-Toyota Specialist",
        "address": "Soroksári út 68",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.4345,
        "lng": 19.0789,
        "phone": "+36 1 901 2345",
        "website": None,
        "rating": 4.7,
        "review_count": 145,
        "specializations": ["japanese"],
        "accepted_makes": ["Suzuki", "Toyota", "Honda", "Mazda"],
        "price_level": 1,
        "services": ["diagnosis", "engine", "brakes", "suspension", "oil_change"],
        "opening_hours": "H-P: 8:00-17:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_09",
        "name": "Premium Auto Klinika",
        "address": "Andrássy út 88",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.5167,
        "lng": 19.0678,
        "phone": "+36 1 012 3456",
        "website": "https://premiumautoklinika.hu",
        "rating": 4.9,
        "review_count": 234,
        "specializations": ["german", "diagnosis"],
        "accepted_makes": ["BMW", "Audi", "Mercedes", "Porsche"],
        "price_level": 3,
        "services": ["diagnosis", "engine", "electronics", "air_conditioning"],
        "opening_hours": "H-P: 8:00-18:00",
        "has_inspection": True,
        "has_courtesy_car": True,
    },
    {
        "id": "shop_10",
        "name": "Gumi és Fék Centrum",
        "address": "Thököly út 45",
        "city": "Budapest",
        "region": "budapest",
        "lat": 47.5012,
        "lng": 19.0845,
        "phone": "+36 1 123 4567",
        "website": None,
        "rating": 3.9,
        "review_count": 67,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["brakes", "tires", "suspension", "oil_change"],
        "opening_hours": "H-P: 7:00-16:00, Szo: 7:00-12:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    # --- Pest megye (3 shops) ---
    {
        "id": "shop_11",
        "name": "Budaörs AutoPark",
        "address": "Szabadság út 22",
        "city": "Budaörs",
        "region": "pest",
        "lat": 47.4512,
        "lng": 18.9567,
        "phone": "+36 23 456 789",
        "website": None,
        "rating": 4.3,
        "review_count": 76,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "oil_change"],
        "opening_hours": "H-P: 8:00-17:00",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_12",
        "name": "Szentendrei Szerviz Pont",
        "address": "Duna korzó 15",
        "city": "Szentendre",
        "region": "pest",
        "lat": 47.6695,
        "lng": 19.0756,
        "phone": "+36 26 345 678",
        "website": None,
        "rating": 4.1,
        "review_count": 43,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["oil_change", "brakes", "tires"],
        "opening_hours": "H-P: 8:00-16:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_13",
        "name": "Érd Autójavító",
        "address": "Budai út 55",
        "city": "Érd",
        "region": "pest",
        "lat": 47.3812,
        "lng": 18.9234,
        "phone": "+36 23 567 890",
        "website": None,
        "rating": 4.4,
        "review_count": 88,
        "specializations": ["general", "french"],
        "accepted_makes": ["Renault", "Peugeot", "Citroen"],
        "price_level": 1,
        "services": ["diagnosis", "engine", "brakes", "suspension"],
        "opening_hours": "H-P: 7:30-16:30",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    # --- Győr (2 shops) ---
    {
        "id": "shop_14",
        "name": "Győri Autószerviz Kft.",
        "address": "Fehérvári út 25",
        "city": "Győr",
        "region": "gyor_moson_sopron",
        "lat": 47.6834,
        "lng": 17.6345,
        "phone": "+36 96 234 567",
        "website": "https://gyori-autoszerviz.hu",
        "rating": 4.5,
        "review_count": 112,
        "specializations": ["general", "german"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "suspension", "electronics"],
        "opening_hours": "H-P: 7:00-17:00",
        "has_inspection": True,
        "has_courtesy_car": True,
    },
    {
        "id": "shop_15",
        "name": "Audi-VW Specialist Győr",
        "address": "Ipar utca 8",
        "city": "Győr",
        "region": "gyor_moson_sopron",
        "lat": 47.6912,
        "lng": 17.6456,
        "phone": "+36 96 345 678",
        "website": None,
        "rating": 4.7,
        "review_count": 167,
        "specializations": ["german"],
        "accepted_makes": ["Audi", "Volkswagen", "Skoda", "Seat"],
        "price_level": 2,
        "services": ["diagnosis", "engine", "electronics", "oil_change"],
        "opening_hours": "H-P: 8:00-17:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    # --- Debrecen (2 shops) ---
    {
        "id": "shop_16",
        "name": "Debreceni Autóház Szerviz",
        "address": "Kishegyesi út 12",
        "city": "Debrecen",
        "region": "hajdu_bihar",
        "lat": 47.5234,
        "lng": 21.6345,
        "phone": "+36 52 234 567",
        "website": "https://debreautohaz.hu",
        "rating": 4.4,
        "review_count": 95,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "air_conditioning"],
        "opening_hours": "H-P: 7:30-17:00, Szo: 8:00-12:00",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_17",
        "name": "Cívis Gyorsszerviz",
        "address": "Péterfia utca 30",
        "city": "Debrecen",
        "region": "hajdu_bihar",
        "lat": 47.5301,
        "lng": 21.6278,
        "phone": "+36 52 345 678",
        "website": None,
        "rating": 4.1,
        "review_count": 62,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["oil_change", "brakes", "tires", "suspension"],
        "opening_hours": "H-P: 8:00-16:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    # --- Szeged (2 shops) ---
    {
        "id": "shop_18",
        "name": "Szegedi Motor Klinika",
        "address": "Londoni körút 9",
        "city": "Szeged",
        "region": "csongrad",
        "lat": 46.2530,
        "lng": 20.1414,
        "phone": "+36 62 234 567",
        "website": None,
        "rating": 4.6,
        "review_count": 134,
        "specializations": ["general", "diagnosis"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "electronics", "brakes"],
        "opening_hours": "H-P: 7:30-17:30",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_19",
        "name": "Tisza Autó Szerviz",
        "address": "Felső Tisza-part 34",
        "city": "Szeged",
        "region": "csongrad",
        "lat": 46.2612,
        "lng": 20.1567,
        "phone": "+36 62 345 678",
        "website": None,
        "rating": 4.0,
        "review_count": 48,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["oil_change", "brakes", "tires"],
        "opening_hours": "H-P: 8:00-16:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    # --- Pécs (2 shops) ---
    {
        "id": "shop_20",
        "name": "Mecsek AutoCenter",
        "address": "Rákóczi út 42",
        "city": "Pécs",
        "region": "baranya",
        "lat": 46.0727,
        "lng": 18.2323,
        "phone": "+36 72 234 567",
        "website": "https://mecsek-autocenter.hu",
        "rating": 4.5,
        "review_count": 101,
        "specializations": ["general", "german"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "suspension", "bodywork"],
        "opening_hours": "H-P: 7:30-17:00",
        "has_inspection": True,
        "has_courtesy_car": True,
    },
    {
        "id": "shop_21",
        "name": "Pécsi Olajcsere Express",
        "address": "Megyeri út 15",
        "city": "Pécs",
        "region": "baranya",
        "lat": 46.0812,
        "lng": 18.2189,
        "phone": "+36 72 345 678",
        "website": None,
        "rating": 4.2,
        "review_count": 55,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["oil_change", "tires", "brakes"],
        "opening_hours": "H-P: 7:00-15:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    # --- Miskolc (2 shops) ---
    {
        "id": "shop_22",
        "name": "Borsodi Autójavító",
        "address": "Zsolcai kapu 28",
        "city": "Miskolc",
        "region": "borsod",
        "lat": 48.0956,
        "lng": 20.7834,
        "phone": "+36 46 234 567",
        "website": None,
        "rating": 4.3,
        "review_count": 78,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["diagnosis", "engine", "brakes", "suspension"],
        "opening_hours": "H-P: 7:30-16:30",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_23",
        "name": "Miskolc Dízel Szerviz",
        "address": "Győri kapu 52",
        "city": "Miskolc",
        "region": "borsod",
        "lat": 48.1023,
        "lng": 20.7756,
        "phone": "+36 46 345 678",
        "website": None,
        "rating": 4.4,
        "review_count": 63,
        "specializations": ["general", "diesel"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "injection", "turbo"],
        "opening_hours": "H-P: 8:00-17:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    # --- Other regions (1 each) ---
    {
        "id": "shop_24",
        "name": "Kecskeméti Autó Centrum",
        "address": "Halasi út 18",
        "city": "Kecskemét",
        "region": "bacs_kiskun",
        "lat": 46.8964,
        "lng": 19.6897,
        "phone": "+36 76 234 567",
        "website": None,
        "rating": 4.2,
        "review_count": 56,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["diagnosis", "engine", "brakes", "oil_change"],
        "opening_hours": "H-P: 7:30-16:30",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_25",
        "name": "Békéscsabai AutoDoktor",
        "address": "Szarvasi út 7",
        "city": "Békéscsaba",
        "region": "bekes",
        "lat": 46.6834,
        "lng": 21.0912,
        "phone": "+36 66 234 567",
        "website": None,
        "rating": 4.0,
        "review_count": 41,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["diagnosis", "engine", "brakes"],
        "opening_hours": "H-P: 8:00-16:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_26",
        "name": "Székesfehérvári Profi Szerviz",
        "address": "Seregélyesi út 20",
        "city": "Székesfehérvár",
        "region": "fejer",
        "lat": 47.1867,
        "lng": 18.4108,
        "phone": "+36 22 234 567",
        "website": "https://profi-szerviz-szfvar.hu",
        "rating": 4.6,
        "review_count": 92,
        "specializations": ["general", "german"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "electronics", "air_conditioning"],
        "opening_hours": "H-P: 7:00-17:00",
        "has_inspection": True,
        "has_courtesy_car": True,
    },
    {
        "id": "shop_27",
        "name": "Szombathelyi Auto Javító",
        "address": "Szelestei út 6",
        "city": "Szombathely",
        "region": "vas",
        "lat": 47.2306,
        "lng": 16.6218,
        "phone": "+36 94 234 567",
        "website": None,
        "rating": 4.1,
        "review_count": 38,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["engine", "brakes", "suspension", "oil_change"],
        "opening_hours": "H-P: 8:00-16:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_28",
        "name": "Nyíregyházi CarFix",
        "address": "Tokaji út 14",
        "city": "Nyíregyháza",
        "region": "szabolcs",
        "lat": 47.9556,
        "lng": 21.7178,
        "phone": "+36 42 234 567",
        "website": None,
        "rating": 4.3,
        "review_count": 52,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["diagnosis", "engine", "brakes", "oil_change"],
        "opening_hours": "H-P: 7:30-16:30",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_29",
        "name": "Veszprémi Motorház",
        "address": "Csap utca 11",
        "city": "Veszprém",
        "region": "veszprem",
        "lat": 47.0934,
        "lng": 17.9125,
        "phone": "+36 88 234 567",
        "website": None,
        "rating": 4.5,
        "review_count": 74,
        "specializations": ["general", "diagnosis"],
        "accepted_makes": [],
        "price_level": 2,
        "services": ["diagnosis", "engine", "brakes", "electronics"],
        "opening_hours": "H-P: 8:00-17:00",
        "has_inspection": True,
        "has_courtesy_car": False,
    },
    {
        "id": "shop_30",
        "name": "Zalaegerszegi Profi Auto",
        "address": "Balatoni út 36",
        "city": "Zalaegerszeg",
        "region": "zala",
        "lat": 46.8417,
        "lng": 16.8416,
        "phone": "+36 92 234 567",
        "website": None,
        "rating": 4.2,
        "review_count": 47,
        "specializations": ["general"],
        "accepted_makes": [],
        "price_level": 1,
        "services": ["engine", "brakes", "oil_change", "tires"],
        "opening_hours": "H-P: 7:30-16:00",
        "has_inspection": False,
        "has_courtesy_car": False,
    },
]


# =============================================================================
# Service Shop Service
# =============================================================================


class ServiceShopService:
    """Service for searching and filtering Hungarian auto repair shops."""

    def __init__(self) -> None:
        """Initialize with static shop and region data."""
        self._shops = SHOPS
        self._regions = REGIONS
        logger.info(
            "ServiceShopService inicializálva: %d szerviz, %d régió",
            len(self._shops),
            len(self._regions),
        )

    def _haversine_distance(
        self,
        lat1: float,
        lng1: float,
        lat2: float,
        lng2: float,
    ) -> float:
        """Calculate distance between two points using Haversine formula.

        Args:
            lat1, lng1: First point coordinates.
            lat2, lng2: Second point coordinates.

        Returns:
            Distance in kilometers.
        """
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
        )
        return r * 2 * math.asin(math.sqrt(a))

    def search_shops(
        self,
        region: Optional[str] = None,
        vehicle_make: Optional[str] = None,
        service_type: Optional[str] = None,
        sort_by: Optional[str] = "rating",
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Search and filter shops.

        Args:
            region: Filter by region ID.
            vehicle_make: Filter by accepted vehicle make.
            service_type: Filter by service type.
            sort_by: Sort field (rating, distance, name).
            lat: User latitude for distance calculation.
            lng: User longitude for distance calculation.
            limit: Max results to return.
            offset: Number of results to skip.

        Returns:
            Dict with 'shops' list and 'total' count.
        """
        filtered = list(self._shops)

        # Filter by region
        if region:
            region_lower = region.lower()
            filtered = [s for s in filtered if s["region"].lower() == region_lower]

        # Filter by vehicle make
        if vehicle_make:
            make_upper = vehicle_make.upper()
            filtered = [
                s
                for s in filtered
                if not s["accepted_makes"]
                or any(m.upper() == make_upper for m in s["accepted_makes"])
            ]

        # Filter by service type
        if service_type:
            svc_lower = service_type.lower()
            filtered = [s for s in filtered if svc_lower in [sv.lower() for sv in s["services"]]]

        # Calculate distances if coordinates provided
        if lat is not None and lng is not None:
            for shop in filtered:
                shop = dict(shop)
                shop["distance_km"] = round(
                    self._haversine_distance(lat, lng, shop["lat"], shop["lng"]),
                    1,
                )
            # Recalculate with distance
            enriched: List[Dict[str, Any]] = []
            for shop in filtered:
                shop_copy = dict(shop)
                shop_copy["distance_km"] = round(
                    self._haversine_distance(lat, lng, shop_copy["lat"], shop_copy["lng"]),
                    1,
                )
                enriched.append(shop_copy)
            filtered = enriched

        total = len(filtered)

        # Sort
        if sort_by == "distance" and lat is not None and lng is not None:
            filtered.sort(key=lambda s: s.get("distance_km", 9999))
        elif sort_by == "name":
            filtered.sort(key=lambda s: s["name"])
        else:
            # Default: sort by rating descending
            filtered.sort(key=lambda s: s["rating"], reverse=True)

        # Paginate
        filtered = filtered[offset : offset + limit]

        return {"shops": filtered, "total": total}

    def get_regions(self) -> List[Dict[str, Any]]:
        """Get all regions with shop counts.

        Returns:
            List of region dicts with shop_count.
        """
        region_counts: Dict[str, int] = {}
        for shop in self._shops:
            rid = shop["region"]
            region_counts[rid] = region_counts.get(rid, 0) + 1

        result: List[Dict[str, Any]] = []
        for region in self._regions:
            r = dict(region)
            r["shop_count"] = region_counts.get(r["id"], 0)
            result.append(r)

        return result

    def get_shop_by_id(self, shop_id: str) -> Optional[Dict[str, Any]]:
        """Get a single shop by ID.

        Args:
            shop_id: Unique shop identifier.

        Returns:
            Shop dict or None if not found.
        """
        for shop in self._shops:
            if shop["id"] == shop_id:
                return dict(shop)
        return None


# =============================================================================
# Singleton
# =============================================================================

_service_instance: Optional[ServiceShopService] = None


def get_service_shop_service() -> ServiceShopService:
    """Get or create the singleton ServiceShopService instance.

    Returns:
        ServiceShopService singleton instance.
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = ServiceShopService()
    return _service_instance
