#!/usr/bin/env python3
"""
Database Seed Script for AutoCognitix

This script populates PostgreSQL and Neo4j databases with initial data:
- DTC codes from generic_codes.json
- European vehicle manufacturers and models
- Test user for development
- Neo4j graph nodes and relationships

Usage:
    python scripts/seed_database.py --postgres    # Seed only PostgreSQL
    python scripts/seed_database.py --neo4j      # Seed only Neo4j
    python scripts/seed_database.py --all        # Seed all databases
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.postgres.models import (
    Base,
    DTCCode,
    User,
    VehicleMake,
    VehicleModel,
)
from backend.app.db.neo4j_models import (
    ComponentNode,
    DTCNode,
    RepairNode,
    SymptomNode,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Data paths - use merged file with all codes (3579 codes from multiple sources)
DTC_CODES_PATH = PROJECT_ROOT / "data" / "dtc_codes" / "all_codes_merged.json"


# European vehicle manufacturers (Top 20)
EUROPEAN_MAKES: List[Dict[str, Any]] = [
    {"id": "volkswagen", "name": "Volkswagen", "country": "Germany"},
    {"id": "bmw", "name": "BMW", "country": "Germany"},
    {"id": "mercedes", "name": "Mercedes-Benz", "country": "Germany"},
    {"id": "audi", "name": "Audi", "country": "Germany"},
    {"id": "porsche", "name": "Porsche", "country": "Germany"},
    {"id": "opel", "name": "Opel", "country": "Germany"},
    {"id": "skoda", "name": "Skoda", "country": "Czech Republic"},
    {"id": "seat", "name": "SEAT", "country": "Spain"},
    {"id": "renault", "name": "Renault", "country": "France"},
    {"id": "peugeot", "name": "Peugeot", "country": "France"},
    {"id": "citroen", "name": "Citroen", "country": "France"},
    {"id": "fiat", "name": "Fiat", "country": "Italy"},
    {"id": "alfa_romeo", "name": "Alfa Romeo", "country": "Italy"},
    {"id": "ferrari", "name": "Ferrari", "country": "Italy"},
    {"id": "lamborghini", "name": "Lamborghini", "country": "Italy"},
    {"id": "volvo", "name": "Volvo", "country": "Sweden"},
    {"id": "saab", "name": "Saab", "country": "Sweden"},
    {"id": "jaguar", "name": "Jaguar", "country": "United Kingdom"},
    {"id": "land_rover", "name": "Land Rover", "country": "United Kingdom"},
    {"id": "mini", "name": "MINI", "country": "United Kingdom"},
]


# Popular models for each make
VEHICLE_MODELS: List[Dict[str, Any]] = [
    # Volkswagen
    {"id": "vw_golf", "name": "Golf", "make_id": "volkswagen", "year_start": 1974, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "MQB"},
    {"id": "vw_passat", "name": "Passat", "make_id": "volkswagen", "year_start": 1973, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MQB"},
    {"id": "vw_tiguan", "name": "Tiguan", "make_id": "volkswagen", "year_start": 2007, "year_end": None, "body_types": ["suv"], "platform": "MQB"},
    {"id": "vw_polo", "name": "Polo", "make_id": "volkswagen", "year_start": 1975, "year_end": None, "body_types": ["hatchback"], "platform": "MQB A0"},
    # BMW
    {"id": "bmw_3series", "name": "3 Series", "make_id": "bmw", "year_start": 1975, "year_end": None, "body_types": ["sedan", "wagon", "coupe"], "platform": "CLAR"},
    {"id": "bmw_5series", "name": "5 Series", "make_id": "bmw", "year_start": 1972, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "CLAR"},
    {"id": "bmw_x3", "name": "X3", "make_id": "bmw", "year_start": 2003, "year_end": None, "body_types": ["suv"], "platform": "CLAR"},
    {"id": "bmw_x5", "name": "X5", "make_id": "bmw", "year_start": 1999, "year_end": None, "body_types": ["suv"], "platform": "CLAR"},
    # Mercedes-Benz
    {"id": "mb_cclass", "name": "C-Class", "make_id": "mercedes", "year_start": 1993, "year_end": None, "body_types": ["sedan", "wagon", "coupe"], "platform": "MRA"},
    {"id": "mb_eclass", "name": "E-Class", "make_id": "mercedes", "year_start": 1993, "year_end": None, "body_types": ["sedan", "wagon", "coupe"], "platform": "MRA"},
    {"id": "mb_glc", "name": "GLC", "make_id": "mercedes", "year_start": 2015, "year_end": None, "body_types": ["suv", "coupe"], "platform": "MRA"},
    # Audi
    {"id": "audi_a3", "name": "A3", "make_id": "audi", "year_start": 1996, "year_end": None, "body_types": ["hatchback", "sedan", "convertible"], "platform": "MQB"},
    {"id": "audi_a4", "name": "A4", "make_id": "audi", "year_start": 1994, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MLB"},
    {"id": "audi_a6", "name": "A6", "make_id": "audi", "year_start": 1994, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MLB"},
    {"id": "audi_q5", "name": "Q5", "make_id": "audi", "year_start": 2008, "year_end": None, "body_types": ["suv"], "platform": "MLB"},
    # Skoda
    {"id": "skoda_octavia", "name": "Octavia", "make_id": "skoda", "year_start": 1996, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MQB"},
    {"id": "skoda_fabia", "name": "Fabia", "make_id": "skoda", "year_start": 1999, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "MQB A0"},
    {"id": "skoda_superb", "name": "Superb", "make_id": "skoda", "year_start": 2001, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MQB"},
    # Renault
    {"id": "renault_clio", "name": "Clio", "make_id": "renault", "year_start": 1990, "year_end": None, "body_types": ["hatchback"], "platform": "CMF-B"},
    {"id": "renault_megane", "name": "Megane", "make_id": "renault", "year_start": 1995, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "CMF-C/D"},
    # Peugeot
    {"id": "peugeot_208", "name": "208", "make_id": "peugeot", "year_start": 2012, "year_end": None, "body_types": ["hatchback"], "platform": "CMP"},
    {"id": "peugeot_308", "name": "308", "make_id": "peugeot", "year_start": 2007, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "EMP2"},
    {"id": "peugeot_3008", "name": "3008", "make_id": "peugeot", "year_start": 2009, "year_end": None, "body_types": ["suv"], "platform": "EMP2"},
    # Citroen
    {"id": "citroen_c3", "name": "C3", "make_id": "citroen", "year_start": 2002, "year_end": None, "body_types": ["hatchback"], "platform": "CMP"},
    {"id": "citroen_c4", "name": "C4", "make_id": "citroen", "year_start": 2004, "year_end": None, "body_types": ["hatchback"], "platform": "EMP2"},
    # Fiat
    {"id": "fiat_500", "name": "500", "make_id": "fiat", "year_start": 2007, "year_end": None, "body_types": ["hatchback", "convertible"], "platform": "CMP"},
    {"id": "fiat_panda", "name": "Panda", "make_id": "fiat", "year_start": 1980, "year_end": None, "body_types": ["hatchback"], "platform": "STLA Small"},
    # Volvo
    {"id": "volvo_xc60", "name": "XC60", "make_id": "volvo", "year_start": 2008, "year_end": None, "body_types": ["suv"], "platform": "SPA"},
    {"id": "volvo_xc90", "name": "XC90", "make_id": "volvo", "year_start": 2002, "year_end": None, "body_types": ["suv"], "platform": "SPA"},
    {"id": "volvo_v60", "name": "V60", "make_id": "volvo", "year_start": 2010, "year_end": None, "body_types": ["wagon"], "platform": "SPA"},
    # Opel
    {"id": "opel_astra", "name": "Astra", "make_id": "opel", "year_start": 1991, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "EMP2"},
    {"id": "opel_corsa", "name": "Corsa", "make_id": "opel", "year_start": 1982, "year_end": None, "body_types": ["hatchback"], "platform": "CMP"},
]


# Vehicle components for Neo4j
COMPONENTS: List[Dict[str, Any]] = [
    {"name": "Mass Air Flow Sensor", "name_hu": "Levegotomeg-mero szenzor (MAF)", "system": "engine"},
    {"name": "Oxygen Sensor", "name_hu": "Oxigen szenzor (Lambda szonda)", "system": "exhaust"},
    {"name": "Catalytic Converter", "name_hu": "Katalizator", "system": "exhaust"},
    {"name": "Throttle Position Sensor", "name_hu": "Fojtoszelep-pozicio szenzor (TPS)", "system": "engine"},
    {"name": "Intake Air Temperature Sensor", "name_hu": "Szivolevego-homerseklet szenzor (IAT)", "system": "engine"},
    {"name": "Coolant Temperature Sensor", "name_hu": "Hutofolyadek-homerseklet szenzor (ECT)", "system": "cooling"},
    {"name": "Crankshaft Position Sensor", "name_hu": "Fotengely pozicio szenzor (CKP)", "system": "engine"},
    {"name": "Camshaft Position Sensor", "name_hu": "Vezermutengeiy pozicio szenzor (CMP)", "system": "engine"},
    {"name": "Spark Plug", "name_hu": "Gyujtogyertya", "system": "ignition"},
    {"name": "Ignition Coil", "name_hu": "Gyujtotekercs", "system": "ignition"},
    {"name": "Fuel Injector", "name_hu": "Befecskendez o", "system": "fuel"},
    {"name": "Fuel Pump", "name_hu": "Uzemanyag-szivattyu", "system": "fuel"},
    {"name": "Fuel Pressure Sensor", "name_hu": "Uzemanyag-nyomas szenzor", "system": "fuel"},
    {"name": "EGR Valve", "name_hu": "EGR szelep", "system": "exhaust"},
    {"name": "EVAP Purge Valve", "name_hu": "EVAP oblito szelep", "system": "evap"},
    {"name": "Knock Sensor", "name_hu": "Kopogas-erzekelo", "system": "engine"},
    {"name": "Vehicle Speed Sensor", "name_hu": "Jarmu sebessegszenzor (VSS)", "system": "transmission"},
    {"name": "Idle Air Control Valve", "name_hu": "Alapjarati levego szelep (IAC)", "system": "engine"},
    {"name": "Alternator", "name_hu": "Generator", "system": "electrical"},
    {"name": "Battery", "name_hu": "Akkumulator", "system": "electrical"},
    {"name": "Thermostat", "name_hu": "Termosztat", "system": "cooling"},
    {"name": "Water Pump", "name_hu": "Vizpumpa", "system": "cooling"},
    {"name": "Wheel Speed Sensor", "name_hu": "Kerek sebessegszenzor (ABS)", "system": "brakes"},
    {"name": "ABS Module", "name_hu": "ABS modul", "system": "brakes"},
    {"name": "ECU/PCM", "name_hu": "Motorvezerlo egyseq (ECU/PCM)", "system": "electrical"},
    {"name": "TCM", "name_hu": "Sebesseqvalto vezerlo (TCM)", "system": "transmission"},
    {"name": "Torque Converter", "name_hu": "Nyomatekvalt6", "system": "transmission"},
    {"name": "Transmission Solenoid", "name_hu": "Valto szolenoid", "system": "transmission"},
    {"name": "Airbag Module", "name_hu": "Legzsak modul", "system": "safety"},
    {"name": "Clock Spring", "name_hu": "Oramutato ruqo", "system": "safety"},
]


# Repair actions for Neo4j
REPAIRS: List[Dict[str, Any]] = [
    {"name": "Clean MAF Sensor", "name_hu": "MAF szenzor tisztitasa", "difficulty": "beginner", "estimated_time_minutes": 30, "estimated_cost_min": 2000, "estimated_cost_max": 5000},
    {"name": "Replace MAF Sensor", "name_hu": "MAF szenzor csereje", "difficulty": "intermediate", "estimated_time_minutes": 45, "estimated_cost_min": 15000, "estimated_cost_max": 45000},
    {"name": "Replace Oxygen Sensor", "name_hu": "Oxigen szenzor csereje", "difficulty": "intermediate", "estimated_time_minutes": 60, "estimated_cost_min": 12000, "estimated_cost_max": 35000},
    {"name": "Replace Catalytic Converter", "name_hu": "Katalizator csereje", "difficulty": "professional", "estimated_time_minutes": 120, "estimated_cost_min": 80000, "estimated_cost_max": 300000},
    {"name": "Replace Spark Plugs", "name_hu": "Gyujtogyertyak csereje", "difficulty": "beginner", "estimated_time_minutes": 45, "estimated_cost_min": 8000, "estimated_cost_max": 25000},
    {"name": "Replace Ignition Coil", "name_hu": "Gyujtotekercs csereje", "difficulty": "intermediate", "estimated_time_minutes": 30, "estimated_cost_min": 10000, "estimated_cost_max": 30000},
    {"name": "Clean Throttle Body", "name_hu": "Fojtoszelephaz tisztitasa", "difficulty": "beginner", "estimated_time_minutes": 30, "estimated_cost_min": 3000, "estimated_cost_max": 8000},
    {"name": "Replace Throttle Position Sensor", "name_hu": "TPS szenzor csereje", "difficulty": "intermediate", "estimated_time_minutes": 45, "estimated_cost_min": 8000, "estimated_cost_max": 25000},
    {"name": "Replace Coolant Temperature Sensor", "name_hu": "Hutofolyadek homerseklet szenzor csereje", "difficulty": "beginner", "estimated_time_minutes": 30, "estimated_cost_min": 5000, "estimated_cost_max": 15000},
    {"name": "Replace Thermostat", "name_hu": "Termosztat csereje", "difficulty": "intermediate", "estimated_time_minutes": 60, "estimated_cost_min": 8000, "estimated_cost_max": 20000},
    {"name": "Replace Crankshaft Position Sensor", "name_hu": "Fotengely pozicio szenzor csereje", "difficulty": "intermediate", "estimated_time_minutes": 60, "estimated_cost_min": 10000, "estimated_cost_max": 30000},
    {"name": "Replace Camshaft Position Sensor", "name_hu": "Vezermutengeiy szenzor csereje", "difficulty": "intermediate", "estimated_time_minutes": 45, "estimated_cost_min": 10000, "estimated_cost_max": 28000},
    {"name": "Clean EGR Valve", "name_hu": "EGR szelep tisztitasa", "difficulty": "intermediate", "estimated_time_minutes": 60, "estimated_cost_min": 5000, "estimated_cost_max": 15000},
    {"name": "Replace EGR Valve", "name_hu": "EGR szelep csereje", "difficulty": "intermediate", "estimated_time_minutes": 90, "estimated_cost_min": 25000, "estimated_cost_max": 60000},
    {"name": "Replace Fuel Pump", "name_hu": "Uzemanyag-szivattyu csereje", "difficulty": "professional", "estimated_time_minutes": 120, "estimated_cost_min": 40000, "estimated_cost_max": 100000},
    {"name": "Replace Fuel Injector", "name_hu": "Befecskendez6 csereje", "difficulty": "advanced", "estimated_time_minutes": 90, "estimated_cost_min": 15000, "estimated_cost_max": 50000},
    {"name": "Replace Gas Cap", "name_hu": "Tanksapka csereje", "difficulty": "beginner", "estimated_time_minutes": 5, "estimated_cost_min": 2000, "estimated_cost_max": 8000},
    {"name": "Replace EVAP Purge Valve", "name_hu": "EVAP oblito szelep csereje", "difficulty": "intermediate", "estimated_time_minutes": 45, "estimated_cost_min": 12000, "estimated_cost_max": 30000},
    {"name": "Replace Wheel Speed Sensor", "name_hu": "Kerek sebessegszenzor csereje", "difficulty": "intermediate", "estimated_time_minutes": 45, "estimated_cost_min": 10000, "estimated_cost_max": 25000},
    {"name": "Replace Alternator", "name_hu": "Generator csereje", "difficulty": "intermediate", "estimated_time_minutes": 90, "estimated_cost_min": 35000, "estimated_cost_max": 80000},
    {"name": "Check and Repair Wiring", "name_hu": "Vezetek ellenorzese es javitasa", "difficulty": "intermediate", "estimated_time_minutes": 60, "estimated_cost_min": 10000, "estimated_cost_max": 30000},
    {"name": "Replace Battery", "name_hu": "Akkumulator csereje", "difficulty": "beginner", "estimated_time_minutes": 20, "estimated_cost_min": 25000, "estimated_cost_max": 60000},
    {"name": "Transmission Fluid Change", "name_hu": "Valto olajcsere", "difficulty": "intermediate", "estimated_time_minutes": 60, "estimated_cost_min": 15000, "estimated_cost_max": 35000},
    {"name": "Replace Torque Converter", "name_hu": "Nyomatekvalt6 csereje", "difficulty": "professional", "estimated_time_minutes": 480, "estimated_cost_min": 150000, "estimated_cost_max": 400000},
]


# DTC to Component mapping
DTC_COMPONENT_MAP: Dict[str, List[str]] = {
    "P0100": ["Mass Air Flow Sensor"],
    "P0101": ["Mass Air Flow Sensor"],
    "P0102": ["Mass Air Flow Sensor"],
    "P0103": ["Mass Air Flow Sensor"],
    "P0110": ["Intake Air Temperature Sensor"],
    "P0115": ["Coolant Temperature Sensor"],
    "P0120": ["Throttle Position Sensor"],
    "P0125": ["Thermostat", "Coolant Temperature Sensor"],
    "P0128": ["Thermostat", "Coolant Temperature Sensor"],
    "P0130": ["Oxygen Sensor"],
    "P0131": ["Oxygen Sensor"],
    "P0133": ["Oxygen Sensor"],
    "P0135": ["Oxygen Sensor"],
    "P0141": ["Oxygen Sensor"],
    "P0171": ["Mass Air Flow Sensor", "Fuel Injector"],
    "P0172": ["Mass Air Flow Sensor", "Fuel Injector"],
    "P0174": ["Oxygen Sensor", "Fuel Injector"],
    "P0175": ["Oxygen Sensor", "Fuel Injector"],
    "P0190": ["Fuel Pressure Sensor"],
    "P0217": ["Thermostat", "Water Pump"],
    "P0230": ["Fuel Pump"],
    "P0300": ["Spark Plug", "Ignition Coil", "Fuel Injector"],
    "P0301": ["Spark Plug", "Ignition Coil"],
    "P0302": ["Spark Plug", "Ignition Coil"],
    "P0303": ["Spark Plug", "Ignition Coil"],
    "P0304": ["Spark Plug", "Ignition Coil"],
    "P0325": ["Knock Sensor"],
    "P0335": ["Crankshaft Position Sensor"],
    "P0340": ["Camshaft Position Sensor"],
    "P0401": ["EGR Valve"],
    "P0402": ["EGR Valve"],
    "P0420": ["Catalytic Converter", "Oxygen Sensor"],
    "P0430": ["Catalytic Converter", "Oxygen Sensor"],
    "P0440": ["EVAP Purge Valve"],
    "P0441": ["EVAP Purge Valve"],
    "P0442": ["EVAP Purge Valve"],
    "P0446": ["EVAP Purge Valve"],
    "P0455": ["EVAP Purge Valve"],
    "P0500": ["Vehicle Speed Sensor"],
    "P0505": ["Idle Air Control Valve"],
    "P0506": ["Idle Air Control Valve"],
    "P0507": ["Idle Air Control Valve"],
    "P0562": ["Alternator", "Battery"],
    "P0600": ["ECU/PCM"],
    "P0700": ["TCM"],
    "P0715": ["Vehicle Speed Sensor"],
    "P0720": ["Vehicle Speed Sensor"],
    "P0730": ["Transmission Solenoid"],
    "P0740": ["Torque Converter"],
    "P0741": ["Torque Converter"],
    "C0035": ["Wheel Speed Sensor"],
    "C0040": ["Wheel Speed Sensor"],
    "C0045": ["Wheel Speed Sensor"],
    "C0050": ["Wheel Speed Sensor"],
    "C0242": ["ECU/PCM"],
    "B0001": ["Airbag Module", "Clock Spring"],
    "B0100": ["Airbag Module"],
    "U0001": ["ECU/PCM"],
    "U0100": ["ECU/PCM"],
    "U0101": ["TCM"],
    "U0121": ["ABS Module"],
    "U0140": ["ECU/PCM"],
    "U0155": ["ECU/PCM"],
}


def load_dtc_codes() -> List[Dict[str, Any]]:
    """Load DTC codes from JSON file."""
    if not DTC_CODES_PATH.exists():
        logger.error(f"DTC codes file not found: {DTC_CODES_PATH}")
        return []

    with open(DTC_CODES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("codes", [])


def get_sync_db_url() -> str:
    """Convert async database URL to sync for seeding."""
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def seed_postgres_dtc_codes(session: Session, dtc_codes: List[Dict[str, Any]]) -> int:
    """Seed DTC codes into PostgreSQL."""
    count = 0

    for code_data in tqdm(dtc_codes, desc="Seeding DTC codes"):
        # Check if code already exists
        existing = session.query(DTCCode).filter_by(code=code_data["code"]).first()
        if existing:
            logger.debug(f"DTC code {code_data['code']} already exists, skipping")
            continue

        dtc = DTCCode(
            code=code_data["code"],
            description_en=code_data["description_en"],
            description_hu=code_data.get("description_hu"),
            category=code_data["category"],
            severity=code_data.get("severity", "medium"),
            is_generic=code_data.get("is_generic", True),
            system=code_data.get("system"),
            symptoms=code_data.get("symptoms", []),
            possible_causes=code_data.get("possible_causes", []),
            diagnostic_steps=code_data.get("diagnostic_steps", []),
            related_codes=code_data.get("related_codes", []),
            sources=code_data.get("sources", []),
        )
        session.add(dtc)
        count += 1

    session.commit()
    return count


def seed_postgres_vehicle_makes(session: Session) -> int:
    """Seed vehicle makes into PostgreSQL."""
    count = 0

    for make_data in tqdm(EUROPEAN_MAKES, desc="Seeding vehicle makes"):
        existing = session.query(VehicleMake).filter_by(id=make_data["id"]).first()
        if existing:
            logger.debug(f"Make {make_data['id']} already exists, skipping")
            continue

        make = VehicleMake(
            id=make_data["id"],
            name=make_data["name"],
            country=make_data.get("country"),
        )
        session.add(make)
        count += 1

    session.commit()
    return count


def seed_postgres_vehicle_models(session: Session) -> int:
    """Seed vehicle models into PostgreSQL."""
    count = 0

    for model_data in tqdm(VEHICLE_MODELS, desc="Seeding vehicle models"):
        existing = session.query(VehicleModel).filter_by(id=model_data["id"]).first()
        if existing:
            logger.debug(f"Model {model_data['id']} already exists, skipping")
            continue

        model = VehicleModel(
            id=model_data["id"],
            name=model_data["name"],
            make_id=model_data["make_id"],
            year_start=model_data["year_start"],
            year_end=model_data.get("year_end"),
            body_types=model_data.get("body_types", []),
            platform=model_data.get("platform"),
        )
        session.add(model)
        count += 1

    session.commit()
    return count


def seed_postgres_test_user(session: Session) -> tuple[bool, str]:
    """Create test user for development with a random secure password.

    Returns:
        Tuple of (created: bool, password: str). Password is only returned if user was created.
    """
    import secrets
    from passlib.hash import bcrypt

    test_email = "test@autocognitix.com"

    existing = session.query(User).filter_by(email=test_email).first()
    if existing:
        logger.info("Test user already exists")
        return False, ""

    # Security: Generate a random 16-character password for the test user
    # This ensures no hardcoded credentials exist in the codebase
    test_password = secrets.token_urlsafe(16)
    hashed_password = bcrypt.hash(test_password)

    user = User(
        email=test_email,
        hashed_password=hashed_password,
        full_name="Test User",
        is_active=True,
        is_superuser=False,
        role="user",
    )
    session.add(user)
    session.commit()
    return True, test_password


def seed_postgres(dtc_codes: List[Dict[str, Any]]) -> None:
    """Seed PostgreSQL database."""
    logger.info("Starting PostgreSQL seeding...")

    db_url = get_sync_db_url()
    engine = create_engine(db_url)

    # Create tables if they don't exist
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Seed DTC codes
        dtc_count = seed_postgres_dtc_codes(session, dtc_codes)
        logger.info(f"Seeded {dtc_count} DTC codes")

        # Seed vehicle makes
        makes_count = seed_postgres_vehicle_makes(session)
        logger.info(f"Seeded {makes_count} vehicle makes")

        # Seed vehicle models
        models_count = seed_postgres_vehicle_models(session)
        logger.info(f"Seeded {models_count} vehicle models")

        # Create test user with random password
        created, test_password = seed_postgres_test_user(session)
        if created:
            logger.info(f"Created test user: test@autocognitix.com")
            logger.info(f"Test user password (SAVE THIS - only shown once): {test_password}")

    logger.info("PostgreSQL seeding completed!")


def seed_neo4j_dtc_nodes(dtc_codes: List[Dict[str, Any]]) -> int:
    """Seed DTC nodes into Neo4j."""
    count = 0

    for code_data in tqdm(dtc_codes, desc="Seeding Neo4j DTC nodes"):
        # Check if node already exists
        existing = DTCNode.nodes.get_or_none(code=code_data["code"])
        if existing:
            logger.debug(f"DTC node {code_data['code']} already exists, skipping")
            continue

        dtc = DTCNode(
            code=code_data["code"],
            description_en=code_data["description_en"],
            description_hu=code_data.get("description_hu"),
            category=code_data["category"],
            severity=code_data.get("severity", "medium"),
            is_generic=str(code_data.get("is_generic", True)).lower(),
            system=code_data.get("system"),
        ).save()

        count += 1

    return count


def seed_neo4j_symptom_nodes(dtc_codes: List[Dict[str, Any]]) -> int:
    """Extract and seed unique symptoms from DTC codes."""
    symptoms: Set[str] = set()

    # Collect all unique symptoms
    for code_data in dtc_codes:
        for symptom in code_data.get("symptoms", []):
            symptoms.add(symptom)

    count = 0
    for symptom_name in tqdm(symptoms, desc="Seeding Neo4j Symptom nodes"):
        existing = SymptomNode.nodes.get_or_none(name=symptom_name)
        if existing:
            continue

        SymptomNode(
            name=symptom_name,
            description_hu=symptom_name,
        ).save()
        count += 1

    return count


def seed_neo4j_component_nodes() -> int:
    """Seed component nodes into Neo4j."""
    count = 0

    for comp_data in tqdm(COMPONENTS, desc="Seeding Neo4j Component nodes"):
        existing = ComponentNode.nodes.get_or_none(name=comp_data["name"])
        if existing:
            continue

        ComponentNode(
            name=comp_data["name"],
            name_hu=comp_data.get("name_hu"),
            system=comp_data.get("system"),
        ).save()
        count += 1

    return count


def seed_neo4j_repair_nodes() -> int:
    """Seed repair nodes into Neo4j."""
    count = 0

    for repair_data in tqdm(REPAIRS, desc="Seeding Neo4j Repair nodes"):
        existing = RepairNode.nodes.get_or_none(name=repair_data["name"])
        if existing:
            continue

        RepairNode(
            name=repair_data["name"],
            description_hu=repair_data.get("name_hu"),
            difficulty=repair_data.get("difficulty", "intermediate"),
            estimated_time_minutes=repair_data.get("estimated_time_minutes"),
            estimated_cost_min=repair_data.get("estimated_cost_min"),
            estimated_cost_max=repair_data.get("estimated_cost_max"),
        ).save()
        count += 1

    return count


def seed_neo4j_relationships(dtc_codes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Create relationships between nodes."""
    counts = {"causes": 0, "indicates_failure": 0}

    for code_data in tqdm(dtc_codes, desc="Creating Neo4j relationships"):
        dtc = DTCNode.nodes.get_or_none(code=code_data["code"])
        if not dtc:
            continue

        # Create DTC -> Symptom relationships
        for symptom_name in code_data.get("symptoms", []):
            symptom = SymptomNode.nodes.get_or_none(name=symptom_name)
            if symptom and not dtc.causes.is_connected(symptom):
                dtc.causes.connect(symptom, {"confidence": 0.7})
                counts["causes"] += 1

        # Create DTC -> Component relationships
        component_names = DTC_COMPONENT_MAP.get(code_data["code"], [])
        for comp_name in component_names:
            component = ComponentNode.nodes.get_or_none(name=comp_name)
            if component and not dtc.indicates_failure_of.is_connected(component):
                dtc.indicates_failure_of.connect(component, {
                    "confidence": 0.8,
                    "failure_mode": "malfunction"
                })
                counts["indicates_failure"] += 1

    return counts


def seed_neo4j(dtc_codes: List[Dict[str, Any]]) -> None:
    """Seed Neo4j database."""
    logger.info("Starting Neo4j seeding...")

    # Seed nodes
    dtc_count = seed_neo4j_dtc_nodes(dtc_codes)
    logger.info(f"Seeded {dtc_count} DTC nodes")

    symptom_count = seed_neo4j_symptom_nodes(dtc_codes)
    logger.info(f"Seeded {symptom_count} Symptom nodes")

    component_count = seed_neo4j_component_nodes()
    logger.info(f"Seeded {component_count} Component nodes")

    repair_count = seed_neo4j_repair_nodes()
    logger.info(f"Seeded {repair_count} Repair nodes")

    # Create relationships
    rel_counts = seed_neo4j_relationships(dtc_codes)
    logger.info(f"Created {rel_counts['causes']} CAUSES relationships")
    logger.info(f"Created {rel_counts['indicates_failure']} INDICATES_FAILURE_OF relationships")

    logger.info("Neo4j seeding completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed AutoCognitix databases with initial data"
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Seed PostgreSQL database only",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Seed Neo4j database only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Seed all databases",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to --all if no specific database is selected
    if not args.postgres and not args.neo4j and not args.all:
        args.all = True

    # Load DTC codes
    logger.info("Loading DTC codes from JSON...")
    dtc_codes = load_dtc_codes()
    if not dtc_codes:
        logger.error("Failed to load DTC codes. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded {len(dtc_codes)} DTC codes")

    try:
        if args.postgres or args.all:
            seed_postgres(dtc_codes)

        if args.neo4j or args.all:
            seed_neo4j(dtc_codes)

        logger.info("Database seeding completed successfully!")

    except Exception as e:
        logger.error(f"Error during seeding: {e}")
        raise


if __name__ == "__main__":
    main()
