"""
Parts Price Service for AutoCognitix.

Alkatresz arak es javitasi koltseg becslese.

Features:
- DTC kodhoz tartozo alkatreszek listazasa
- Alkatresz ar becsles (min/max)
- Munkadij becsles nehezseg alapjan
- Redis cache 24h TTL
- Fallback statikus adatokra

Author: AutoCognitix Team
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# Static Data - Fallback prices and DTC-Part mappings
# =============================================================================

# Munkaora dij kategoriank (HUF/ora)
LABOR_RATES: Dict[str, Dict[str, int]] = {
    "easy": {"min": 8000, "max": 15000},
    "medium": {"min": 12000, "max": 25000},
    "hard": {"min": 20000, "max": 40000},
    "expert": {"min": 35000, "max": 60000},
}

# Statikus alkatresz arak (fallback)
STATIC_PARTS_PRICES: Dict[str, Dict[str, Any]] = {
    "oxygen_sensor": {
        "name": "Lambda szonda (oxigen erzekelo)",
        "name_en": "Oxygen Sensor",
        "category": "sensors",
        "price_min": 12000,
        "price_max": 85000,
        "price_avg": 35000,
        "labor_hours": 0.5,
    },
    "catalytic_converter": {
        "name": "Katalizator",
        "name_en": "Catalytic Converter",
        "category": "exhaust",
        "price_min": 80000,
        "price_max": 450000,
        "price_avg": 180000,
        "labor_hours": 2.0,
    },
    "maf_sensor": {
        "name": "Legtomegmero szenzor (MAF)",
        "name_en": "Mass Air Flow Sensor",
        "category": "sensors",
        "price_min": 15000,
        "price_max": 85000,
        "price_avg": 35000,
        "labor_hours": 0.5,
    },
    "air_filter": {
        "name": "Legszuro",
        "name_en": "Air Filter",
        "category": "filters",
        "price_min": 3000,
        "price_max": 15000,
        "price_avg": 6000,
        "labor_hours": 0.25,
    },
    "spark_plug": {
        "name": "Gyujtagyertya",
        "name_en": "Spark Plug",
        "category": "ignition",
        "price_min": 1500,
        "price_max": 8000,
        "price_avg": 3500,
        "labor_hours": 0.5,
    },
    "ignition_coil": {
        "name": "Gyujtotrafo",
        "name_en": "Ignition Coil",
        "category": "ignition",
        "price_min": 8000,
        "price_max": 45000,
        "price_avg": 18000,
        "labor_hours": 0.5,
    },
    "egr_valve": {
        "name": "EGR szelep",
        "name_en": "EGR Valve",
        "category": "emissions",
        "price_min": 25000,
        "price_max": 120000,
        "price_avg": 55000,
        "labor_hours": 1.5,
    },
    "throttle_body": {
        "name": "Fojtoszelep haz",
        "name_en": "Throttle Body",
        "category": "fuel_system",
        "price_min": 35000,
        "price_max": 150000,
        "price_avg": 70000,
        "labor_hours": 1.0,
    },
    "fuel_pump": {
        "name": "Uzemanyag szivatyu",
        "name_en": "Fuel Pump",
        "category": "fuel_system",
        "price_min": 25000,
        "price_max": 120000,
        "price_avg": 55000,
        "labor_hours": 2.0,
    },
    "fuel_filter": {
        "name": "Uzemanyag szuro",
        "name_en": "Fuel Filter",
        "category": "filters",
        "price_min": 3000,
        "price_max": 18000,
        "price_avg": 8000,
        "labor_hours": 0.5,
    },
    "abs_sensor": {
        "name": "ABS kerekfordulat erzekelo",
        "name_en": "ABS Wheel Speed Sensor",
        "category": "sensors",
        "price_min": 8000,
        "price_max": 45000,
        "price_avg": 18000,
        "labor_hours": 0.75,
    },
    "crankshaft_sensor": {
        "name": "Fotengelyhelyzet erzekelo",
        "name_en": "Crankshaft Position Sensor",
        "category": "sensors",
        "price_min": 10000,
        "price_max": 55000,
        "price_avg": 25000,
        "labor_hours": 1.0,
    },
    "camshaft_sensor": {
        "name": "Vezermutengelyhelyzet erzekelo",
        "name_en": "Camshaft Position Sensor",
        "category": "sensors",
        "price_min": 10000,
        "price_max": 55000,
        "price_avg": 25000,
        "labor_hours": 0.75,
    },
    "coolant_temp_sensor": {
        "name": "Hutoviz homerseklet erzekelo",
        "name_en": "Coolant Temperature Sensor",
        "category": "sensors",
        "price_min": 3000,
        "price_max": 18000,
        "price_avg": 8000,
        "labor_hours": 0.5,
    },
    "thermostat": {
        "name": "Termosztat",
        "name_en": "Thermostat",
        "category": "cooling",
        "price_min": 5000,
        "price_max": 35000,
        "price_avg": 15000,
        "labor_hours": 1.5,
    },
    "water_pump": {
        "name": "Vizszivatyu",
        "name_en": "Water Pump",
        "category": "cooling",
        "price_min": 15000,
        "price_max": 85000,
        "price_avg": 35000,
        "labor_hours": 3.0,
    },
}

# DTC kod -> alkatreszek mapping
DTC_PARTS_MAPPING: Dict[str, List[str]] = {
    # Uzemanyag rendszer
    "P0171": ["maf_sensor", "air_filter", "oxygen_sensor", "fuel_filter"],
    "P0172": ["maf_sensor", "air_filter", "oxygen_sensor", "fuel_filter"],
    "P0174": ["maf_sensor", "air_filter", "oxygen_sensor"],
    "P0175": ["maf_sensor", "air_filter", "oxygen_sensor"],
    # Lambda szonda
    "P0130": ["oxygen_sensor"],
    "P0131": ["oxygen_sensor"],
    "P0132": ["oxygen_sensor"],
    "P0133": ["oxygen_sensor"],
    "P0134": ["oxygen_sensor"],
    "P0135": ["oxygen_sensor"],
    "P0136": ["oxygen_sensor"],
    "P0137": ["oxygen_sensor"],
    "P0138": ["oxygen_sensor"],
    "P0139": ["oxygen_sensor"],
    "P0140": ["oxygen_sensor"],
    "P0141": ["oxygen_sensor"],
    # Gyujtasi rendszer
    "P0300": ["spark_plug", "ignition_coil"],
    "P0301": ["spark_plug", "ignition_coil"],
    "P0302": ["spark_plug", "ignition_coil"],
    "P0303": ["spark_plug", "ignition_coil"],
    "P0304": ["spark_plug", "ignition_coil"],
    "P0305": ["spark_plug", "ignition_coil"],
    "P0306": ["spark_plug", "ignition_coil"],
    "P0307": ["spark_plug", "ignition_coil"],
    "P0308": ["spark_plug", "ignition_coil"],
    # Katalizator
    "P0420": ["catalytic_converter", "oxygen_sensor"],
    "P0421": ["catalytic_converter", "oxygen_sensor"],
    "P0430": ["catalytic_converter", "oxygen_sensor"],
    "P0431": ["catalytic_converter", "oxygen_sensor"],
    # EGR
    "P0400": ["egr_valve"],
    "P0401": ["egr_valve"],
    "P0402": ["egr_valve"],
    "P0403": ["egr_valve"],
    "P0404": ["egr_valve"],
    "P0405": ["egr_valve"],
    # MAF
    "P0100": ["maf_sensor", "air_filter"],
    "P0101": ["maf_sensor", "air_filter"],
    "P0102": ["maf_sensor", "air_filter"],
    "P0103": ["maf_sensor", "air_filter"],
    "P0104": ["maf_sensor", "air_filter"],
    # Fojtoszelep
    "P0120": ["throttle_body"],
    "P0121": ["throttle_body"],
    "P0122": ["throttle_body"],
    "P0123": ["throttle_body"],
    "P0124": ["throttle_body"],
    # Hutesi rendszer
    "P0115": ["coolant_temp_sensor"],
    "P0116": ["coolant_temp_sensor", "thermostat"],
    "P0117": ["coolant_temp_sensor"],
    "P0118": ["coolant_temp_sensor"],
    "P0119": ["coolant_temp_sensor"],
    "P0125": ["thermostat", "coolant_temp_sensor"],
    "P0126": ["thermostat"],
    "P0128": ["thermostat", "coolant_temp_sensor"],
    # Fotengely/vezeromutengely
    "P0335": ["crankshaft_sensor"],
    "P0336": ["crankshaft_sensor"],
    "P0337": ["crankshaft_sensor"],
    "P0338": ["crankshaft_sensor"],
    "P0339": ["crankshaft_sensor"],
    "P0340": ["camshaft_sensor"],
    "P0341": ["camshaft_sensor"],
    "P0342": ["camshaft_sensor"],
    "P0343": ["camshaft_sensor"],
    "P0344": ["camshaft_sensor"],
    # Uzemanyag szivatyu
    "P0230": ["fuel_pump"],
    "P0231": ["fuel_pump"],
    "P0232": ["fuel_pump"],
    # ABS
    "C0035": ["abs_sensor"],
    "C0040": ["abs_sensor"],
    "C0045": ["abs_sensor"],
    "C0050": ["abs_sensor"],
}


# =============================================================================
# Cache Helper
# =============================================================================

class PartsPriceCache:
    """Redis cache helper for parts prices with in-memory fallback."""

    CACHE_PREFIX = "parts:"
    DEFAULT_TTL = 86400  # 24 hours

    def __init__(self):
        self._redis = None
        self._in_memory_cache: Dict[str, tuple] = {}

    async def _get_redis(self):
        """Get Redis connection (lazy init)."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                redis_url = getattr(settings, "REDIS_URL", None)
                if redis_url:
                    self._redis = await aioredis.from_url(redis_url)
                else:
                    return None
            except Exception as e:
                logger.warning(f"Redis kapcsolodasi hiba: {e}, in-memory cache hasznalata")
                return None
        return self._redis

    def _make_key(self, *args) -> str:
        """Generate cache key."""
        key_data = ":".join(str(a).lower() for a in args)
        return f"{self.CACHE_PREFIX}{hashlib.md5(key_data.encode()).hexdigest()}"

    async def get(self, *args) -> Optional[str]:
        """Get value from cache."""
        key = self._make_key(*args)

        # Try Redis first
        redis = await self._get_redis()
        if redis:
            try:
                return await redis.get(key)
            except Exception as e:
                logger.warning(f"Redis get hiba: {e}")

        # Fallback to in-memory cache
        if key in self._in_memory_cache:
            value, expiry = self._in_memory_cache[key]
            if expiry > datetime.now().timestamp():
                return value
            del self._in_memory_cache[key]

        return None

    async def set(self, value: str, *args, ttl: int = DEFAULT_TTL) -> None:
        """Set value in cache."""
        key = self._make_key(*args)

        # Try Redis first
        redis = await self._get_redis()
        if redis:
            try:
                await redis.setex(key, ttl, value)
                return
            except Exception as e:
                logger.warning(f"Redis set hiba: {e}")

        # Fallback to in-memory cache
        expiry = datetime.now().timestamp() + ttl
        self._in_memory_cache[key] = (value, expiry)

    async def close(self) -> None:
        """Close connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# =============================================================================
# Parts Price Service
# =============================================================================

class PartsPriceService:
    """
    Alkatresz ar es javitasi koltseg szolgaltatas.

    Features:
    - Alkatresz ar lekerdezés
    - DTC alapu alkatresz kereses
    - Javitasi koltseg becsles
    - Redis cache 24h TTL
    - Fallback statikus adatokra
    """

    _instance: Optional["PartsPriceService"] = None
    _initialized: bool = False

    def __new__(cls) -> "PartsPriceService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize service."""
        if self._initialized:
            return

        self._initialized = True
        self._cache = PartsPriceCache()
        logger.info("PartsPriceService inicializalva")

    async def get_part_price(
        self,
        part_key: str,
        vehicle_make: Optional[str] = None,
        vehicle_model: Optional[str] = None,
        vehicle_year: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get part price information.

        Args:
            part_key: Part key (e.g., "oxygen_sensor")
            vehicle_make: Vehicle make (optional, for price adjustment)
            vehicle_model: Vehicle model (optional)
            vehicle_year: Vehicle year (optional)

        Returns:
            Part info dict or None if not found
        """
        # Check cache
        cache_key = f"part:{part_key}:{vehicle_make}:{vehicle_model}:{vehicle_year}"
        cached = await self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit: {part_key}")
            return json.loads(cached)

        # Get from static data
        part_data = STATIC_PARTS_PRICES.get(part_key.lower())
        if not part_data:
            logger.warning(f"Alkatresz nem talalhato: {part_key}")
            return None

        # Price adjustment based on vehicle
        price_multiplier = 1.0
        if vehicle_make:
            make_upper = vehicle_make.upper()
            if make_upper in ["BMW", "MERCEDES", "AUDI", "PORSCHE", "LEXUS"]:
                price_multiplier = 1.5
            elif make_upper in ["VOLKSWAGEN", "FORD", "OPEL", "TOYOTA"]:
                price_multiplier = 1.0
            elif make_upper in ["DACIA", "SUZUKI", "KIA", "HYUNDAI"]:
                price_multiplier = 0.85

        result = {
            "id": part_key,
            "name": part_data["name"],
            "name_en": part_data.get("name_en", ""),
            "category": part_data.get("category", "other"),
            "price_range_min": int(part_data["price_min"] * price_multiplier),
            "price_range_max": int(part_data["price_max"] * price_multiplier),
            "price_avg": int(part_data.get("price_avg", 0) * price_multiplier),
            "labor_hours": part_data.get("labor_hours", 1.0),
            "currency": "HUF",
            "sources": [],
            "from_static": True,
        }

        # Cache result
        await self._cache.set(json.dumps(result), cache_key)

        return result

    async def get_parts_for_dtc(
        self,
        dtc_code: str,
        vehicle_make: Optional[str] = None,
        vehicle_model: Optional[str] = None,
        vehicle_year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get parts needed for a DTC code.

        Args:
            dtc_code: DTC code (e.g., "P0420")
            vehicle_make: Vehicle make
            vehicle_model: Vehicle model
            vehicle_year: Vehicle year

        Returns:
            List of parts
        """
        dtc_upper = dtc_code.upper().strip()

        # Check cache
        cache_key = f"dtc_parts:{dtc_upper}:{vehicle_make}:{vehicle_model}:{vehicle_year}"
        cached = await self._cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit DTC alkatreszek: {dtc_upper}")
            return json.loads(cached)

        # Get parts from mapping
        part_keys = DTC_PARTS_MAPPING.get(dtc_upper, [])
        if not part_keys:
            logger.info(f"Nincs ismert alkatresz a {dtc_upper} kodhoz")
            return []

        # Get part details
        parts = []
        for part_key in part_keys:
            part_info = await self.get_part_price(
                part_key=part_key,
                vehicle_make=vehicle_make,
                vehicle_model=vehicle_model,
                vehicle_year=vehicle_year,
            )
            if part_info:
                parts.append(part_info)

        # Cache result
        if parts:
            await self._cache.set(json.dumps(parts), cache_key)

        logger.info(f"Talalt alkatreszek ({dtc_upper}): {len(parts)} db")
        return parts

    async def estimate_repair_cost(
        self,
        dtc_code: Optional[str] = None,
        parts: Optional[List[Dict[str, Any]]] = None,
        vehicle_make: Optional[str] = None,
        vehicle_model: Optional[str] = None,
        vehicle_year: Optional[int] = None,
        include_labor: bool = True,
    ) -> Dict[str, Any]:
        """
        Estimate repair cost for a DTC code.

        Args:
            dtc_code: DTC code (optional if parts provided)
            parts: Parts list (optional if dtc_code provided)
            vehicle_make: Vehicle make
            vehicle_model: Vehicle model
            vehicle_year: Vehicle year
            include_labor: Include labor cost

        Returns:
            Cost estimate dict
        """
        # Get parts if not provided
        if parts is None:
            if dtc_code:
                parts = await self.get_parts_for_dtc(
                    dtc_code=dtc_code,
                    vehicle_make=vehicle_make,
                    vehicle_model=vehicle_model,
                    vehicle_year=vehicle_year,
                )
            else:
                parts = []

        # Calculate parts cost
        parts_cost_min = sum(p.get("price_range_min", 0) for p in parts)
        parts_cost_max = sum(p.get("price_range_max", 0) for p in parts)

        # Calculate labor cost
        labor_cost_min = 0
        labor_cost_max = 0
        total_hours = 0.0

        if include_labor and parts:
            total_hours = sum(p.get("labor_hours", 1.0) for p in parts)

            # Determine difficulty based on hours
            if total_hours <= 1.0:
                difficulty = "easy"
            elif total_hours <= 2.5:
                difficulty = "medium"
            elif total_hours <= 4.0:
                difficulty = "hard"
            else:
                difficulty = "expert"

            rates = LABOR_RATES.get(difficulty, LABOR_RATES["medium"])
            labor_cost_min = int(total_hours * rates["min"])
            labor_cost_max = int(total_hours * rates["max"])
        else:
            difficulty = "medium"

        # Build result
        result = {
            "dtc_code": dtc_code,
            "repair_name": self._get_repair_name(dtc_code),
            "repair_description": self._get_repair_description(dtc_code),
            "parts": parts,
            "parts_cost_min": parts_cost_min,
            "parts_cost_max": parts_cost_max,
            "labor_cost_min": labor_cost_min,
            "labor_cost_max": labor_cost_max,
            "total_cost_min": parts_cost_min + labor_cost_min,
            "total_cost_max": parts_cost_max + labor_cost_max,
            "currency": "HUF",
            "estimated_hours": total_hours,
            "difficulty": difficulty,
            "confidence": self._calculate_confidence(parts, dtc_code),
            "notes": self._generate_notes(parts, vehicle_make),
            "vehicle_info": f"{vehicle_make or ''} {vehicle_model or ''} {vehicle_year or ''}".strip(),
            "disclaimer": "A becselesek tajekoztato jelleguek. A tenyleges arak a szerviz, "
            "az alkatresz minoseg es az allapot fuggvenyeben elterhetnek.",
        }

        logger.info(
            f"Koltsegbecsles ({dtc_code}): "
            f"{result['total_cost_min']:,} - {result['total_cost_max']:,} HUF"
        )

        return result

    def _get_repair_name(self, dtc_code: Optional[str]) -> str:
        """Get repair name based on DTC code."""
        if not dtc_code:
            return "Altalanos javitas"

        dtc_upper = dtc_code.upper()
        repair_names = {
            "P0420": "Katalizator rendszer javitas",
            "P0300": "Gyujtasi rendszer javitas",
            "P0171": "Uzemanyag rendszer javitas",
            "P0101": "Legtomegmero csere",
            "P0130": "Lambda szonda csere",
            "P0400": "EGR rendszer javitas",
        }

        # Prefix-based search
        for prefix, name in repair_names.items():
            if dtc_upper.startswith(prefix[:4]):
                return name

        return f"Javitas ({dtc_code})"

    def _get_repair_description(self, dtc_code: Optional[str]) -> str:
        """Get repair description."""
        if not dtc_code:
            return ""

        descriptions = {
            "P04": "A katalizator vagy a kipufogo rendszer ellenorzese es szukseg eseten csereje.",
            "P03": "A gyujtasi rendszer (gyertyak, gyujtotrafok) ellenorzese es csereje.",
            "P01": "A motor vezerlo szenzorok es uzemanyag rendszer ellenorzese.",
            "C00": "Az ABS rendszer szenzorainak es kabelezesenek ellenorzese.",
        }

        for prefix, desc in descriptions.items():
            if dtc_code.upper().startswith(prefix):
                return desc

        return "A diagnosztika alapjan szukseges javitasi munkalatok."

    def _calculate_confidence(
        self, parts: List[Dict[str, Any]], dtc_code: Optional[str]
    ) -> float:
        """Calculate estimate confidence."""
        if not parts:
            return 0.3

        base_confidence = 0.5

        # Boost if DTC has known parts
        if dtc_code and dtc_code.upper() in DTC_PARTS_MAPPING:
            base_confidence += 0.2

        # Boost if all parts have prices
        if all(p.get("price_range_min", 0) > 0 for p in parts):
            base_confidence += 0.1

        # Boost if from static data
        if all(p.get("from_static", False) for p in parts):
            base_confidence += 0.1

        return min(base_confidence, 0.95)

    def _generate_notes(
        self, parts: List[Dict[str, Any]], vehicle_make: Optional[str]
    ) -> str:
        """Generate notes for the estimate."""
        notes = []

        if not parts:
            notes.append("Nincs ismert alkatresz adat ehhez a hibakohoz.")
            return " ".join(notes)

        # Premium vehicle warning
        if vehicle_make and vehicle_make.upper() in [
            "BMW",
            "MERCEDES",
            "AUDI",
            "PORSCHE",
        ]:
            notes.append("Premium markanal az arak magasabbak lehetnek.")

        # Catalytic converter warning
        if any("catalytic" in p.get("name_en", "").lower() for p in parts):
            notes.append("A katalizator csere jelentos koltseg, fontolja meg a felujitott opciokat.")

        # Multiple parts
        if len(parts) > 2:
            notes.append("Tobb alkatresz erintett, erdemes csomagban kerni arajanlatot.")

        return " ".join(notes) if notes else ""

    async def close(self) -> None:
        """Close service."""
        await self._cache.close()
        logger.info("PartsPriceService leallitva")


# =============================================================================
# Module-level Functions
# =============================================================================

_service_instance: Optional[PartsPriceService] = None


def get_parts_price_service() -> PartsPriceService:
    """
    Get singleton service instance.

    Returns:
        PartsPriceService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = PartsPriceService()
    return _service_instance


async def get_repair_cost_estimate(
    dtc_code: str,
    vehicle_make: Optional[str] = None,
    vehicle_model: Optional[str] = None,
    vehicle_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience function for repair cost estimation.

    Args:
        dtc_code: DTC code
        vehicle_make: Vehicle make
        vehicle_model: Vehicle model
        vehicle_year: Vehicle year

    Returns:
        Cost estimate dict
    """
    service = get_parts_price_service()
    return await service.estimate_repair_cost(
        dtc_code=dtc_code,
        vehicle_make=vehicle_make,
        vehicle_model=vehicle_model,
        vehicle_year=vehicle_year,
    )
