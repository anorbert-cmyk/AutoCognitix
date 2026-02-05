#!/usr/bin/env python3
"""
Comprehensive Vehicle Database Seed Script for AutoCognitix

This script populates the database with:
- All major European makes (VW, BMW, Mercedes, Audi, Skoda, Renault, Peugeot, Opel, Fiat, Volvo, etc.)
- All major Asian makes (Toyota, Honda, Nissan, Hyundai, Kia, Mazda, Subaru, Mitsubishi, Suzuki)
- All major American makes (Ford, Chevrolet, Dodge, Jeep, GMC, Ram, Chrysler)
- Popular models with years 2000-2025
- Common engine codes
- Vehicle platforms
- DTC frequency data

Usage:
    python scripts/seed_vehicles.py --all        # Seed all vehicle data
    python scripts/seed_vehicles.py --makes      # Seed makes only
    python scripts/seed_vehicles.py --models     # Seed models only
    python scripts/seed_vehicles.py --engines    # Seed engines only
    python scripts/seed_vehicles.py --platforms  # Seed platforms only
    python scripts/seed_vehicles.py --neo4j      # Seed Neo4j vehicle nodes
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.postgres.models import (
    Base,
    VehicleDTCFrequency,
    VehicleEngine,
    VehicleMake,
    VehicleModel,
    VehicleModelEngine,
    VehiclePlatform,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# VEHICLE MAKES - Comprehensive list by region
# =============================================================================

EUROPEAN_MAKES: List[Dict[str, Any]] = [
    # German
    {"id": "volkswagen", "name": "Volkswagen", "country": "Germany"},
    {"id": "bmw", "name": "BMW", "country": "Germany"},
    {"id": "mercedes", "name": "Mercedes-Benz", "country": "Germany"},
    {"id": "audi", "name": "Audi", "country": "Germany"},
    {"id": "porsche", "name": "Porsche", "country": "Germany"},
    {"id": "opel", "name": "Opel", "country": "Germany"},
    {"id": "smart", "name": "Smart", "country": "Germany"},
    # Czech
    {"id": "skoda", "name": "Skoda", "country": "Czech Republic"},
    # Spanish
    {"id": "seat", "name": "SEAT", "country": "Spain"},
    {"id": "cupra", "name": "Cupra", "country": "Spain"},
    # French
    {"id": "renault", "name": "Renault", "country": "France"},
    {"id": "peugeot", "name": "Peugeot", "country": "France"},
    {"id": "citroen", "name": "Citroen", "country": "France"},
    {"id": "ds", "name": "DS Automobiles", "country": "France"},
    {"id": "alpine", "name": "Alpine", "country": "France"},
    # Italian
    {"id": "fiat", "name": "Fiat", "country": "Italy"},
    {"id": "alfa_romeo", "name": "Alfa Romeo", "country": "Italy"},
    {"id": "lancia", "name": "Lancia", "country": "Italy"},
    {"id": "ferrari", "name": "Ferrari", "country": "Italy"},
    {"id": "lamborghini", "name": "Lamborghini", "country": "Italy"},
    {"id": "maserati", "name": "Maserati", "country": "Italy"},
    # Swedish
    {"id": "volvo", "name": "Volvo", "country": "Sweden"},
    {"id": "polestar", "name": "Polestar", "country": "Sweden"},
    # British
    {"id": "jaguar", "name": "Jaguar", "country": "United Kingdom"},
    {"id": "land_rover", "name": "Land Rover", "country": "United Kingdom"},
    {"id": "mini", "name": "MINI", "country": "United Kingdom"},
    {"id": "bentley", "name": "Bentley", "country": "United Kingdom"},
    {"id": "rolls_royce", "name": "Rolls-Royce", "country": "United Kingdom"},
    {"id": "aston_martin", "name": "Aston Martin", "country": "United Kingdom"},
    {"id": "mclaren", "name": "McLaren", "country": "United Kingdom"},
    {"id": "lotus", "name": "Lotus", "country": "United Kingdom"},
    # Romanian
    {"id": "dacia", "name": "Dacia", "country": "Romania"},
]

ASIAN_MAKES: List[Dict[str, Any]] = [
    # Japanese
    {"id": "toyota", "name": "Toyota", "country": "Japan"},
    {"id": "lexus", "name": "Lexus", "country": "Japan"},
    {"id": "honda", "name": "Honda", "country": "Japan"},
    {"id": "acura", "name": "Acura", "country": "Japan"},
    {"id": "nissan", "name": "Nissan", "country": "Japan"},
    {"id": "infiniti", "name": "Infiniti", "country": "Japan"},
    {"id": "mazda", "name": "Mazda", "country": "Japan"},
    {"id": "subaru", "name": "Subaru", "country": "Japan"},
    {"id": "mitsubishi", "name": "Mitsubishi", "country": "Japan"},
    {"id": "suzuki", "name": "Suzuki", "country": "Japan"},
    {"id": "isuzu", "name": "Isuzu", "country": "Japan"},
    # Korean
    {"id": "hyundai", "name": "Hyundai", "country": "South Korea"},
    {"id": "kia", "name": "Kia", "country": "South Korea"},
    {"id": "genesis", "name": "Genesis", "country": "South Korea"},
    # Chinese
    {"id": "byd", "name": "BYD", "country": "China"},
    {"id": "geely", "name": "Geely", "country": "China"},
    {"id": "nio", "name": "NIO", "country": "China"},
    {"id": "xpeng", "name": "XPeng", "country": "China"},
    {"id": "mg", "name": "MG", "country": "China"},
]

AMERICAN_MAKES: List[Dict[str, Any]] = [
    # Ford
    {"id": "ford", "name": "Ford", "country": "USA"},
    {"id": "lincoln", "name": "Lincoln", "country": "USA"},
    # General Motors
    {"id": "chevrolet", "name": "Chevrolet", "country": "USA"},
    {"id": "gmc", "name": "GMC", "country": "USA"},
    {"id": "buick", "name": "Buick", "country": "USA"},
    {"id": "cadillac", "name": "Cadillac", "country": "USA"},
    # Stellantis (formerly FCA)
    {"id": "dodge", "name": "Dodge", "country": "USA"},
    {"id": "chrysler", "name": "Chrysler", "country": "USA"},
    {"id": "jeep", "name": "Jeep", "country": "USA"},
    {"id": "ram", "name": "Ram", "country": "USA"},
    # Electric
    {"id": "tesla", "name": "Tesla", "country": "USA"},
    {"id": "rivian", "name": "Rivian", "country": "USA"},
    {"id": "lucid", "name": "Lucid", "country": "USA"},
]

ALL_MAKES = EUROPEAN_MAKES + ASIAN_MAKES + AMERICAN_MAKES


# =============================================================================
# VEHICLE PLATFORMS
# =============================================================================

PLATFORMS: List[Dict[str, Any]] = [
    # VW Group
    {"code": "MQB", "name": "Modularer Querbaukasten", "manufacturer": "Volkswagen Group", "makes": ["volkswagen", "audi", "skoda", "seat", "cupra"], "year_start": 2012, "segment": "C", "body_types": ["hatchback", "sedan", "wagon", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "MQB_EVO", "name": "MQB Evo", "manufacturer": "Volkswagen Group", "makes": ["volkswagen", "audi", "skoda", "seat", "cupra"], "year_start": 2020, "segment": "C", "body_types": ["hatchback", "sedan", "wagon", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "MQB_A0", "name": "MQB A0", "manufacturer": "Volkswagen Group", "makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2017, "segment": "B", "body_types": ["hatchback", "suv"], "drivetrain_options": ["FWD"]},
    {"code": "MLB", "name": "Modularer Langsbaukasten", "manufacturer": "Volkswagen Group", "makes": ["audi", "porsche", "bentley", "lamborghini"], "year_start": 2008, "segment": "D", "body_types": ["sedan", "wagon", "suv"], "drivetrain_options": ["AWD", "RWD"]},
    {"code": "MLB_EVO", "name": "MLB Evo", "manufacturer": "Volkswagen Group", "makes": ["audi", "porsche", "bentley", "lamborghini"], "year_start": 2015, "segment": "E", "body_types": ["sedan", "wagon", "suv"], "drivetrain_options": ["AWD", "RWD"]},
    {"code": "MEB", "name": "Modularer E-Antriebs-Baukasten", "manufacturer": "Volkswagen Group", "makes": ["volkswagen", "audi", "skoda", "seat", "cupra"], "year_start": 2020, "segment": "C", "body_types": ["hatchback", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "MSB", "name": "Modular Standard Drivetrain Matrix", "manufacturer": "Volkswagen Group", "makes": ["porsche", "bentley"], "year_start": 2016, "segment": "F", "body_types": ["sedan", "coupe"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "J1", "name": "J1 Performance Platform", "manufacturer": "Volkswagen Group", "makes": ["porsche", "audi"], "year_start": 2019, "segment": "E", "body_types": ["sedan", "wagon"], "drivetrain_options": ["AWD"]},
    # BMW
    {"code": "CLAR", "name": "Cluster Architecture", "manufacturer": "BMW", "makes": ["bmw"], "year_start": 2015, "segment": "D", "body_types": ["sedan", "wagon", "coupe", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "UKL", "name": "Untere Klasse", "manufacturer": "BMW", "makes": ["bmw", "mini"], "year_start": 2014, "segment": "B", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "FAAR", "name": "Front Architecture", "manufacturer": "BMW", "makes": ["bmw", "mini"], "year_start": 2019, "segment": "C", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    # Mercedes
    {"code": "MRA", "name": "Modular Rear Architecture", "manufacturer": "Mercedes-Benz", "makes": ["mercedes"], "year_start": 2014, "segment": "D", "body_types": ["sedan", "wagon", "coupe", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "MRA2", "name": "Modular Rear Architecture 2", "manufacturer": "Mercedes-Benz", "makes": ["mercedes"], "year_start": 2020, "segment": "E", "body_types": ["sedan", "wagon", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "MFA", "name": "Modular Front Architecture", "manufacturer": "Mercedes-Benz", "makes": ["mercedes"], "year_start": 2012, "segment": "B", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "MFA2", "name": "Modular Front Architecture 2", "manufacturer": "Mercedes-Benz", "makes": ["mercedes"], "year_start": 2019, "segment": "C", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "EVA2", "name": "Electric Vehicle Architecture 2", "manufacturer": "Mercedes-Benz", "makes": ["mercedes"], "year_start": 2021, "segment": "E", "body_types": ["sedan", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    # Stellantis/PSA
    {"code": "CMP", "name": "Common Modular Platform", "manufacturer": "Stellantis", "makes": ["peugeot", "citroen", "opel", "ds", "fiat"], "year_start": 2019, "segment": "B", "body_types": ["hatchback", "suv"], "drivetrain_options": ["FWD"]},
    {"code": "EMP2", "name": "Efficient Modular Platform 2", "manufacturer": "Stellantis", "makes": ["peugeot", "citroen", "opel", "ds"], "year_start": 2013, "segment": "C", "body_types": ["hatchback", "sedan", "wagon", "suv", "van"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "STLA_MEDIUM", "name": "STLA Medium", "manufacturer": "Stellantis", "makes": ["peugeot", "citroen", "opel", "alfa_romeo", "ds", "dodge", "chrysler", "jeep"], "year_start": 2023, "segment": "C", "body_types": ["hatchback", "sedan", "wagon", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "STLA_LARGE", "name": "STLA Large", "manufacturer": "Stellantis", "makes": ["dodge", "chrysler", "jeep", "alfa_romeo", "maserati"], "year_start": 2024, "segment": "E", "body_types": ["sedan", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    # Renault
    {"code": "CMF_B", "name": "CMF-B", "manufacturer": "Renault-Nissan-Mitsubishi", "makes": ["renault", "dacia", "nissan"], "year_start": 2019, "segment": "B", "body_types": ["hatchback", "suv"], "drivetrain_options": ["FWD"]},
    {"code": "CMF_CD", "name": "CMF-C/D", "manufacturer": "Renault-Nissan-Mitsubishi", "makes": ["renault", "nissan", "mitsubishi"], "year_start": 2013, "segment": "C", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "CMF_EV", "name": "CMF-EV", "manufacturer": "Renault-Nissan-Mitsubishi", "makes": ["renault", "nissan"], "year_start": 2022, "segment": "C", "body_types": ["hatchback", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    # Toyota
    {"code": "TNGA_C", "name": "TNGA-C", "manufacturer": "Toyota", "makes": ["toyota", "lexus"], "year_start": 2015, "segment": "C", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "TNGA_K", "name": "TNGA-K", "manufacturer": "Toyota", "makes": ["toyota", "lexus"], "year_start": 2017, "segment": "D", "body_types": ["sedan", "wagon", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "TNGA_L", "name": "TNGA-L", "manufacturer": "Toyota", "makes": ["toyota", "lexus"], "year_start": 2018, "segment": "E", "body_types": ["sedan", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "TNGA_F", "name": "TNGA-F", "manufacturer": "Toyota", "makes": ["toyota", "lexus"], "year_start": 2021, "segment": "F", "body_types": ["suv", "pickup"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "E_TNGA", "name": "e-TNGA", "manufacturer": "Toyota", "makes": ["toyota", "lexus", "subaru"], "year_start": 2022, "segment": "C", "body_types": ["suv"], "drivetrain_options": ["FWD", "AWD"]},
    # Honda
    {"code": "HONDA_GLOBAL_SMALL", "name": "Honda Global Small Platform", "manufacturer": "Honda", "makes": ["honda"], "year_start": 2020, "segment": "B", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "HONDA_GLOBAL_LIGHT", "name": "Honda Global Light Platform", "manufacturer": "Honda", "makes": ["honda", "acura"], "year_start": 2018, "segment": "C", "body_types": ["sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    # Hyundai/Kia
    {"code": "K_PLATFORM", "name": "K Platform", "manufacturer": "Hyundai-Kia", "makes": ["hyundai", "kia"], "year_start": 2019, "segment": "B", "body_types": ["hatchback", "sedan"], "drivetrain_options": ["FWD"]},
    {"code": "N3_PLATFORM", "name": "N3 Platform", "manufacturer": "Hyundai-Kia", "makes": ["hyundai", "kia", "genesis"], "year_start": 2019, "segment": "C", "body_types": ["sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "M3_PLATFORM", "name": "M3 Platform", "manufacturer": "Hyundai-Kia", "makes": ["hyundai", "kia", "genesis"], "year_start": 2020, "segment": "D", "body_types": ["sedan", "wagon", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "E_GMP", "name": "Electric-Global Modular Platform", "manufacturer": "Hyundai-Kia", "makes": ["hyundai", "kia", "genesis"], "year_start": 2021, "segment": "D", "body_types": ["sedan", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    # Volvo
    {"code": "SPA", "name": "Scalable Product Architecture", "manufacturer": "Volvo", "makes": ["volvo", "polestar"], "year_start": 2015, "segment": "D", "body_types": ["sedan", "wagon", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "SPA2", "name": "Scalable Product Architecture 2", "manufacturer": "Volvo", "makes": ["volvo", "polestar"], "year_start": 2022, "segment": "E", "body_types": ["suv"], "drivetrain_options": ["AWD"]},
    {"code": "CMA", "name": "Compact Modular Architecture", "manufacturer": "Volvo/Geely", "makes": ["volvo", "polestar", "geely"], "year_start": 2017, "segment": "C", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    # Ford
    {"code": "C2", "name": "C2 Platform", "manufacturer": "Ford", "makes": ["ford"], "year_start": 2012, "segment": "C", "body_types": ["hatchback", "sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "CD6", "name": "CD6 Platform", "manufacturer": "Ford", "makes": ["ford", "lincoln"], "year_start": 2019, "segment": "D", "body_types": ["suv"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "P702", "name": "P702 Platform", "manufacturer": "Ford", "makes": ["ford"], "year_start": 2020, "segment": "F", "body_types": ["pickup"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "GE2", "name": "GE2 Platform", "manufacturer": "Ford", "makes": ["ford"], "year_start": 2021, "segment": "D", "body_types": ["suv"], "drivetrain_options": ["RWD", "AWD"]},
    # GM
    {"code": "ALPHA", "name": "Alpha Platform", "manufacturer": "General Motors", "makes": ["cadillac", "chevrolet"], "year_start": 2012, "segment": "D", "body_types": ["sedan", "coupe"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "OMEGA", "name": "Omega Platform", "manufacturer": "General Motors", "makes": ["cadillac", "buick"], "year_start": 2016, "segment": "E", "body_types": ["sedan"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "VSS_S", "name": "VSS-S Platform", "manufacturer": "General Motors", "makes": ["chevrolet", "buick"], "year_start": 2019, "segment": "C", "body_types": ["sedan", "suv"], "drivetrain_options": ["FWD", "AWD"]},
    {"code": "VSS_F", "name": "VSS-F Platform", "manufacturer": "General Motors", "makes": ["chevrolet", "gmc", "cadillac"], "year_start": 2019, "segment": "F", "body_types": ["suv", "pickup"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "ULTIUM", "name": "Ultium Platform", "manufacturer": "General Motors", "makes": ["chevrolet", "gmc", "cadillac", "buick"], "year_start": 2022, "segment": "D", "body_types": ["sedan", "suv", "pickup"], "drivetrain_options": ["RWD", "AWD"]},
    # Tesla
    {"code": "MODEL_S_X", "name": "Model S/X Platform", "manufacturer": "Tesla", "makes": ["tesla"], "year_start": 2012, "segment": "E", "body_types": ["sedan", "suv"], "drivetrain_options": ["RWD", "AWD"]},
    {"code": "MODEL_3_Y", "name": "Model 3/Y Platform", "manufacturer": "Tesla", "makes": ["tesla"], "year_start": 2017, "segment": "D", "body_types": ["sedan", "suv"], "drivetrain_options": ["RWD", "AWD"]},
]


# =============================================================================
# VEHICLE ENGINES - Common engine codes by manufacturer
# =============================================================================

ENGINES: List[Dict[str, Any]] = [
    # VW Group - EA888 Family (TSI/TFSI)
    {"code": "EA888_GEN3_1.8", "name": "EA888 Gen3 1.8 TSI", "displacement_cc": 1798, "displacement_l": 1.8, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 180, "torque_nm": 280, "family": "EA888", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2014, "year_end": 2020},
    {"code": "EA888_GEN3_2.0_190", "name": "EA888 Gen3 2.0 TSI 190HP", "displacement_cc": 1984, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 190, "torque_nm": 320, "family": "EA888", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2014},
    {"code": "EA888_GEN3_2.0_220", "name": "EA888 Gen3 2.0 TSI 220HP", "displacement_cc": 1984, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 220, "torque_nm": 350, "family": "EA888", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2014},
    {"code": "EA888_GEN3_2.0_245", "name": "EA888 Gen3 2.0 TSI 245HP", "displacement_cc": 1984, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 245, "torque_nm": 370, "family": "EA888", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "seat", "cupra"], "year_start": 2017},
    {"code": "EA888_GEN4_2.0_300", "name": "EA888 Gen4 2.0 TSI 300HP", "displacement_cc": 1984, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 300, "torque_nm": 400, "family": "EA888", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "cupra"], "year_start": 2020},

    # VW Group - EA211 Family (TSI small)
    {"code": "EA211_1.0_95", "name": "EA211 1.0 TSI 95HP", "displacement_cc": 999, "displacement_l": 1.0, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 95, "torque_nm": 175, "family": "EA211", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "skoda", "seat"], "year_start": 2015},
    {"code": "EA211_1.0_110", "name": "EA211 1.0 TSI 110HP", "displacement_cc": 999, "displacement_l": 1.0, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 110, "torque_nm": 200, "family": "EA211", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "skoda", "seat"], "year_start": 2015},
    {"code": "EA211_1.5_150", "name": "EA211 1.5 TSI 150HP", "displacement_cc": 1498, "displacement_l": 1.5, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 150, "torque_nm": 250, "family": "EA211", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2017},

    # VW Group - EA288 Family (TDI)
    {"code": "EA288_1.6_115", "name": "EA288 1.6 TDI 115HP", "displacement_cc": 1598, "displacement_l": 1.6, "cylinders": 4, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 115, "torque_nm": 250, "family": "EA288", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2014},
    {"code": "EA288_2.0_150", "name": "EA288 2.0 TDI 150HP", "displacement_cc": 1968, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 150, "torque_nm": 340, "family": "EA288", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2014},
    {"code": "EA288_2.0_190", "name": "EA288 2.0 TDI 190HP", "displacement_cc": 1968, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 190, "torque_nm": 400, "family": "EA288", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat"], "year_start": 2015},

    # BMW - B Series
    {"code": "B38_1.5_136", "name": "B38 1.5 Turbo 136HP", "displacement_cc": 1499, "displacement_l": 1.5, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 136, "torque_nm": 220, "family": "B38", "manufacturer": "BMW", "applicable_makes": ["bmw", "mini"], "year_start": 2014},
    {"code": "B46_2.0_184", "name": "B46 2.0 Turbo 184HP", "displacement_cc": 1998, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 184, "torque_nm": 300, "family": "B46", "manufacturer": "BMW", "applicable_makes": ["bmw"], "year_start": 2015},
    {"code": "B48_2.0_258", "name": "B48 2.0 Turbo 258HP", "displacement_cc": 1998, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 258, "torque_nm": 400, "family": "B48", "manufacturer": "BMW", "applicable_makes": ["bmw"], "year_start": 2015},
    {"code": "B58_3.0_340", "name": "B58 3.0 Turbo 340HP", "displacement_cc": 2998, "displacement_l": 3.0, "cylinders": 6, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 340, "torque_nm": 500, "family": "B58", "manufacturer": "BMW", "applicable_makes": ["bmw", "toyota"], "year_start": 2015},
    {"code": "B47_2.0D_150", "name": "B47 2.0d 150HP", "displacement_cc": 1995, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 150, "torque_nm": 350, "family": "B47", "manufacturer": "BMW", "applicable_makes": ["bmw"], "year_start": 2014},
    {"code": "B47_2.0D_190", "name": "B47 2.0d 190HP", "displacement_cc": 1995, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 190, "torque_nm": 400, "family": "B47", "manufacturer": "BMW", "applicable_makes": ["bmw"], "year_start": 2014},
    {"code": "B57_3.0D_265", "name": "B57 3.0d 265HP", "displacement_cc": 2993, "displacement_l": 3.0, "cylinders": 6, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 265, "torque_nm": 620, "family": "B57", "manufacturer": "BMW", "applicable_makes": ["bmw"], "year_start": 2016},

    # Mercedes - M260/M264/M274/M276
    {"code": "M260_1.6_122", "name": "M260 1.6 Turbo 122HP", "displacement_cc": 1595, "displacement_l": 1.6, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 122, "torque_nm": 200, "family": "M260", "manufacturer": "Mercedes-Benz", "applicable_makes": ["mercedes"], "year_start": 2017},
    {"code": "M264_2.0_184", "name": "M264 2.0 Turbo 184HP", "displacement_cc": 1991, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 184, "torque_nm": 300, "family": "M264", "manufacturer": "Mercedes-Benz", "applicable_makes": ["mercedes"], "year_start": 2017},
    {"code": "M264_2.0_258", "name": "M264 2.0 Turbo 258HP", "displacement_cc": 1991, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 258, "torque_nm": 370, "family": "M264", "manufacturer": "Mercedes-Benz", "applicable_makes": ["mercedes"], "year_start": 2017},
    {"code": "M256_3.0_367", "name": "M256 3.0 Turbo 367HP", "displacement_cc": 2999, "displacement_l": 3.0, "cylinders": 6, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 367, "torque_nm": 500, "family": "M256", "manufacturer": "Mercedes-Benz", "applicable_makes": ["mercedes"], "year_start": 2017},
    {"code": "OM654_2.0D_194", "name": "OM654 2.0d 194HP", "displacement_cc": 1950, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 194, "torque_nm": 400, "family": "OM654", "manufacturer": "Mercedes-Benz", "applicable_makes": ["mercedes"], "year_start": 2016},
    {"code": "OM656_3.0D_286", "name": "OM656 3.0d 286HP", "displacement_cc": 2925, "displacement_l": 3.0, "cylinders": 6, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 286, "torque_nm": 600, "family": "OM656", "manufacturer": "Mercedes-Benz", "applicable_makes": ["mercedes"], "year_start": 2017},

    # Toyota - Dynamic Force
    {"code": "A25A_FKS_2.5_203", "name": "A25A-FKS 2.5 NA 203HP", "displacement_cc": 2487, "displacement_l": 2.5, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "naturally_aspirated", "power_hp": 203, "torque_nm": 250, "family": "Dynamic Force", "manufacturer": "Toyota", "applicable_makes": ["toyota", "lexus"], "year_start": 2018},
    {"code": "A25A_FXS_2.5_HYB", "name": "A25A-FXS 2.5 Hybrid", "displacement_cc": 2487, "displacement_l": 2.5, "cylinders": 4, "configuration": "inline", "fuel_type": "hybrid", "aspiration": "naturally_aspirated", "power_hp": 218, "torque_nm": 221, "family": "Dynamic Force", "manufacturer": "Toyota", "applicable_makes": ["toyota", "lexus"], "year_start": 2018},
    {"code": "M20A_FKS_2.0_170", "name": "M20A-FKS 2.0 NA 170HP", "displacement_cc": 1987, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "naturally_aspirated", "power_hp": 170, "torque_nm": 205, "family": "Dynamic Force", "manufacturer": "Toyota", "applicable_makes": ["toyota", "lexus"], "year_start": 2018},
    {"code": "G16E_GTS_1.6T_272", "name": "G16E-GTS 1.6 Turbo 272HP", "displacement_cc": 1618, "displacement_l": 1.6, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 272, "torque_nm": 370, "family": "Dynamic Force", "manufacturer": "Toyota", "applicable_makes": ["toyota"], "year_start": 2020},

    # Hyundai/Kia - Smartstream
    {"code": "G4LD_1.4T_140", "name": "Smartstream G1.4 T-GDi 140HP", "displacement_cc": 1353, "displacement_l": 1.4, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 140, "torque_nm": 242, "family": "Smartstream", "manufacturer": "Hyundai-Kia", "applicable_makes": ["hyundai", "kia"], "year_start": 2019},
    {"code": "G4FJ_1.6T_177", "name": "Smartstream G1.6 T-GDi 177HP", "displacement_cc": 1591, "displacement_l": 1.6, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 177, "torque_nm": 265, "family": "Smartstream", "manufacturer": "Hyundai-Kia", "applicable_makes": ["hyundai", "kia"], "year_start": 2015},
    {"code": "G4KH_2.0T_245", "name": "Smartstream G2.0 T-GDi 245HP", "displacement_cc": 1998, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 245, "torque_nm": 353, "family": "Smartstream", "manufacturer": "Hyundai-Kia", "applicable_makes": ["hyundai", "kia", "genesis"], "year_start": 2019},
    {"code": "G4KN_2.5T_281", "name": "Smartstream G2.5 T-GDi 281HP", "displacement_cc": 2497, "displacement_l": 2.5, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 281, "torque_nm": 421, "family": "Smartstream", "manufacturer": "Hyundai-Kia", "applicable_makes": ["hyundai", "kia", "genesis"], "year_start": 2021},
    {"code": "D4FE_1.6D_136", "name": "Smartstream D1.6 CRDi 136HP", "displacement_cc": 1598, "displacement_l": 1.6, "cylinders": 4, "configuration": "inline", "fuel_type": "diesel", "aspiration": "turbo", "power_hp": 136, "torque_nm": 320, "family": "Smartstream", "manufacturer": "Hyundai-Kia", "applicable_makes": ["hyundai", "kia"], "year_start": 2018},

    # Ford - EcoBoost
    {"code": "ECOBOOST_1.0_125", "name": "EcoBoost 1.0 125HP", "displacement_cc": 999, "displacement_l": 1.0, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 125, "torque_nm": 200, "family": "EcoBoost", "manufacturer": "Ford", "applicable_makes": ["ford"], "year_start": 2012},
    {"code": "ECOBOOST_1.5_150", "name": "EcoBoost 1.5 150HP", "displacement_cc": 1497, "displacement_l": 1.5, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 150, "torque_nm": 240, "family": "EcoBoost", "manufacturer": "Ford", "applicable_makes": ["ford"], "year_start": 2014},
    {"code": "ECOBOOST_2.0_250", "name": "EcoBoost 2.0 250HP", "displacement_cc": 1999, "displacement_l": 2.0, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 250, "torque_nm": 373, "family": "EcoBoost", "manufacturer": "Ford", "applicable_makes": ["ford"], "year_start": 2015},
    {"code": "ECOBOOST_2.3_280", "name": "EcoBoost 2.3 280HP", "displacement_cc": 2261, "displacement_l": 2.3, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 280, "torque_nm": 420, "family": "EcoBoost", "manufacturer": "Ford", "applicable_makes": ["ford"], "year_start": 2015},

    # Renault/PSA - PureTech/TCe
    {"code": "PURETECH_1.2_130", "name": "PureTech 1.2 130HP", "displacement_cc": 1199, "displacement_l": 1.2, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 130, "torque_nm": 230, "family": "PureTech", "manufacturer": "Stellantis", "applicable_makes": ["peugeot", "citroen", "opel", "ds"], "year_start": 2014},
    {"code": "PURETECH_1.6_225", "name": "PureTech 1.6 225HP", "displacement_cc": 1598, "displacement_l": 1.6, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 225, "torque_nm": 300, "family": "PureTech", "manufacturer": "Stellantis", "applicable_makes": ["peugeot", "citroen", "ds"], "year_start": 2018},
    {"code": "TCE_1.0_100", "name": "TCe 1.0 100HP", "displacement_cc": 999, "displacement_l": 1.0, "cylinders": 3, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 100, "torque_nm": 160, "family": "TCe", "manufacturer": "Renault", "applicable_makes": ["renault", "dacia"], "year_start": 2018},
    {"code": "TCE_1.3_140", "name": "TCe 1.3 140HP", "displacement_cc": 1332, "displacement_l": 1.3, "cylinders": 4, "configuration": "inline", "fuel_type": "gasoline", "aspiration": "turbo", "power_hp": 140, "torque_nm": 240, "family": "TCe", "manufacturer": "Renault", "applicable_makes": ["renault", "dacia"], "year_start": 2018},

    # Electric Motors
    {"code": "APP310_150KW", "name": "APP310 150kW Electric", "displacement_cc": 0, "displacement_l": 0, "cylinders": 0, "configuration": "electric", "fuel_type": "electric", "aspiration": "electric", "power_hp": 204, "power_kw": 150, "torque_nm": 310, "family": "MEB", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat", "cupra"], "year_start": 2020},
    {"code": "APP310_195KW", "name": "APP310 195kW Electric", "displacement_cc": 0, "displacement_l": 0, "cylinders": 0, "configuration": "electric", "fuel_type": "electric", "aspiration": "electric", "power_hp": 265, "power_kw": 195, "torque_nm": 310, "family": "MEB", "manufacturer": "Volkswagen", "applicable_makes": ["volkswagen", "audi", "skoda", "seat", "cupra"], "year_start": 2020},
    {"code": "BMW_EDRIVE_GEN5_210", "name": "BMW eDrive Gen5 210kW", "displacement_cc": 0, "displacement_l": 0, "cylinders": 0, "configuration": "electric", "fuel_type": "electric", "aspiration": "electric", "power_hp": 286, "power_kw": 210, "torque_nm": 400, "family": "eDrive", "manufacturer": "BMW", "applicable_makes": ["bmw"], "year_start": 2020},
    {"code": "TESLA_3DM_DUAL", "name": "Tesla 3DM Dual Motor", "displacement_cc": 0, "displacement_l": 0, "cylinders": 0, "configuration": "electric", "fuel_type": "electric", "aspiration": "electric", "power_hp": 346, "power_kw": 258, "torque_nm": 527, "family": "3DM", "manufacturer": "Tesla", "applicable_makes": ["tesla"], "year_start": 2019},
    {"code": "E_GMP_160KW_RWD", "name": "E-GMP 160kW RWD", "displacement_cc": 0, "displacement_l": 0, "cylinders": 0, "configuration": "electric", "fuel_type": "electric", "aspiration": "electric", "power_hp": 218, "power_kw": 160, "torque_nm": 350, "family": "E-GMP", "manufacturer": "Hyundai-Kia", "applicable_makes": ["hyundai", "kia", "genesis"], "year_start": 2021},
    {"code": "E_GMP_239KW_AWD", "name": "E-GMP 239kW AWD", "displacement_cc": 0, "displacement_l": 0, "cylinders": 0, "configuration": "electric", "fuel_type": "electric", "aspiration": "electric", "power_hp": 325, "power_kw": 239, "torque_nm": 605, "family": "E-GMP", "manufacturer": "Hyundai-Kia", "applicable_makes": ["hyundai", "kia", "genesis"], "year_start": 2021},
]


# =============================================================================
# VEHICLE MODELS - Popular models 2000-2025
# =============================================================================

MODELS: List[Dict[str, Any]] = [
    # Volkswagen
    {"id": "vw_golf_mk7", "name": "Golf VII", "make_id": "volkswagen", "year_start": 2012, "year_end": 2020, "body_types": ["hatchback", "wagon"], "platform": "MQB", "engine_codes": ["EA211_1.0_95", "EA211_1.0_110", "EA211_1.5_150", "EA888_GEN3_2.0_190", "EA888_GEN3_2.0_220", "EA288_1.6_115", "EA288_2.0_150"]},
    {"id": "vw_golf_mk8", "name": "Golf VIII", "make_id": "volkswagen", "year_start": 2019, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "MQB_EVO", "engine_codes": ["EA211_1.0_110", "EA211_1.5_150", "EA888_GEN3_2.0_245", "EA888_GEN4_2.0_300", "EA288_2.0_150", "EA288_2.0_190"]},
    {"id": "vw_passat_b8", "name": "Passat B8", "make_id": "volkswagen", "year_start": 2014, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MQB", "engine_codes": ["EA211_1.5_150", "EA888_GEN3_1.8", "EA888_GEN3_2.0_190", "EA888_GEN3_2.0_220", "EA288_1.6_115", "EA288_2.0_150", "EA288_2.0_190"]},
    {"id": "vw_tiguan_mk2", "name": "Tiguan II", "make_id": "volkswagen", "year_start": 2016, "year_end": None, "body_types": ["suv"], "platform": "MQB", "engine_codes": ["EA211_1.5_150", "EA888_GEN3_2.0_190", "EA888_GEN3_2.0_220", "EA288_2.0_150", "EA288_2.0_190"]},
    {"id": "vw_polo_mk6", "name": "Polo VI", "make_id": "volkswagen", "year_start": 2017, "year_end": None, "body_types": ["hatchback"], "platform": "MQB_A0", "engine_codes": ["EA211_1.0_95", "EA211_1.0_110", "EA211_1.5_150", "EA888_GEN3_2.0_200"]},
    {"id": "vw_id3", "name": "ID.3", "make_id": "volkswagen", "year_start": 2020, "year_end": None, "body_types": ["hatchback"], "platform": "MEB", "engine_codes": ["APP310_150KW", "APP310_195KW"]},
    {"id": "vw_id4", "name": "ID.4", "make_id": "volkswagen", "year_start": 2020, "year_end": None, "body_types": ["suv"], "platform": "MEB", "engine_codes": ["APP310_150KW", "APP310_195KW"]},
    {"id": "vw_touareg_mk3", "name": "Touareg III", "make_id": "volkswagen", "year_start": 2018, "year_end": None, "body_types": ["suv"], "platform": "MLB_EVO"},

    # BMW
    {"id": "bmw_3series_g20", "name": "3 Series (G20)", "make_id": "bmw", "year_start": 2018, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "CLAR", "engine_codes": ["B46_2.0_184", "B48_2.0_258", "B58_3.0_340", "B47_2.0D_150", "B47_2.0D_190"]},
    {"id": "bmw_3series_f30", "name": "3 Series (F30)", "make_id": "bmw", "year_start": 2011, "year_end": 2019, "body_types": ["sedan", "wagon"], "platform": "CLAR"},
    {"id": "bmw_5series_g30", "name": "5 Series (G30)", "make_id": "bmw", "year_start": 2016, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "CLAR", "engine_codes": ["B48_2.0_258", "B58_3.0_340", "B47_2.0D_190", "B57_3.0D_265"]},
    {"id": "bmw_x3_g01", "name": "X3 (G01)", "make_id": "bmw", "year_start": 2017, "year_end": None, "body_types": ["suv"], "platform": "CLAR", "engine_codes": ["B46_2.0_184", "B48_2.0_258", "B58_3.0_340", "B47_2.0D_190"]},
    {"id": "bmw_x5_g05", "name": "X5 (G05)", "make_id": "bmw", "year_start": 2018, "year_end": None, "body_types": ["suv"], "platform": "CLAR", "engine_codes": ["B58_3.0_340", "B57_3.0D_265"]},
    {"id": "bmw_1series_f40", "name": "1 Series (F40)", "make_id": "bmw", "year_start": 2019, "year_end": None, "body_types": ["hatchback"], "platform": "FAAR", "engine_codes": ["B38_1.5_136", "B46_2.0_184", "B48_2.0_258"]},
    {"id": "bmw_i4", "name": "i4", "make_id": "bmw", "year_start": 2021, "year_end": None, "body_types": ["sedan"], "platform": "CLAR", "engine_codes": ["BMW_EDRIVE_GEN5_210"]},

    # Mercedes-Benz
    {"id": "mb_cclass_w206", "name": "C-Class (W206)", "make_id": "mercedes", "year_start": 2021, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MRA2", "engine_codes": ["M264_2.0_184", "M264_2.0_258", "M256_3.0_367", "OM654_2.0D_194"]},
    {"id": "mb_cclass_w205", "name": "C-Class (W205)", "make_id": "mercedes", "year_start": 2014, "year_end": 2021, "body_types": ["sedan", "wagon", "coupe"], "platform": "MRA"},
    {"id": "mb_eclass_w213", "name": "E-Class (W213)", "make_id": "mercedes", "year_start": 2016, "year_end": None, "body_types": ["sedan", "wagon", "coupe"], "platform": "MRA", "engine_codes": ["M264_2.0_184", "M264_2.0_258", "M256_3.0_367", "OM654_2.0D_194", "OM656_3.0D_286"]},
    {"id": "mb_glc_x254", "name": "GLC (X254)", "make_id": "mercedes", "year_start": 2022, "year_end": None, "body_types": ["suv", "coupe"], "platform": "MRA2", "engine_codes": ["M264_2.0_258", "M256_3.0_367", "OM654_2.0D_194"]},
    {"id": "mb_aclass_w177", "name": "A-Class (W177)", "make_id": "mercedes", "year_start": 2018, "year_end": None, "body_types": ["hatchback", "sedan"], "platform": "MFA2", "engine_codes": ["M260_1.6_122", "M264_2.0_184", "M264_2.0_258"]},
    {"id": "mb_eqs", "name": "EQS", "make_id": "mercedes", "year_start": 2021, "year_end": None, "body_types": ["sedan"], "platform": "EVA2"},

    # Audi
    {"id": "audi_a3_8y", "name": "A3 (8Y)", "make_id": "audi", "year_start": 2020, "year_end": None, "body_types": ["hatchback", "sedan"], "platform": "MQB_EVO", "engine_codes": ["EA211_1.0_110", "EA211_1.5_150", "EA888_GEN3_2.0_190", "EA888_GEN3_2.0_245", "EA288_2.0_150"]},
    {"id": "audi_a4_b9", "name": "A4 (B9)", "make_id": "audi", "year_start": 2015, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MLB_EVO", "engine_codes": ["EA888_GEN3_2.0_190", "EA888_GEN3_2.0_245", "EA288_2.0_150", "EA288_2.0_190"]},
    {"id": "audi_a6_c8", "name": "A6 (C8)", "make_id": "audi", "year_start": 2018, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MLB_EVO"},
    {"id": "audi_q5_fy", "name": "Q5 (FY)", "make_id": "audi", "year_start": 2016, "year_end": None, "body_types": ["suv"], "platform": "MLB_EVO", "engine_codes": ["EA888_GEN3_2.0_190", "EA888_GEN3_2.0_245", "EA288_2.0_150", "EA288_2.0_190"]},
    {"id": "audi_etron_gt", "name": "e-tron GT", "make_id": "audi", "year_start": 2021, "year_end": None, "body_types": ["sedan"], "platform": "J1"},

    # Skoda
    {"id": "skoda_octavia_mk4", "name": "Octavia IV", "make_id": "skoda", "year_start": 2019, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MQB_EVO", "engine_codes": ["EA211_1.0_110", "EA211_1.5_150", "EA888_GEN3_2.0_190", "EA888_GEN3_2.0_245", "EA288_2.0_150", "EA288_2.0_190"]},
    {"id": "skoda_octavia_mk3", "name": "Octavia III", "make_id": "skoda", "year_start": 2012, "year_end": 2020, "body_types": ["sedan", "wagon"], "platform": "MQB"},
    {"id": "skoda_superb_mk3", "name": "Superb III", "make_id": "skoda", "year_start": 2015, "year_end": None, "body_types": ["sedan", "wagon"], "platform": "MQB"},
    {"id": "skoda_karoq", "name": "Karoq", "make_id": "skoda", "year_start": 2017, "year_end": None, "body_types": ["suv"], "platform": "MQB"},
    {"id": "skoda_enyaq", "name": "Enyaq iV", "make_id": "skoda", "year_start": 2020, "year_end": None, "body_types": ["suv"], "platform": "MEB", "engine_codes": ["APP310_150KW", "APP310_195KW"]},

    # Toyota
    {"id": "toyota_corolla_e210", "name": "Corolla (E210)", "make_id": "toyota", "year_start": 2018, "year_end": None, "body_types": ["hatchback", "sedan", "wagon"], "platform": "TNGA_C", "engine_codes": ["M20A_FKS_2.0_170", "A25A_FXS_2.5_HYB"]},
    {"id": "toyota_camry_xv70", "name": "Camry (XV70)", "make_id": "toyota", "year_start": 2017, "year_end": None, "body_types": ["sedan"], "platform": "TNGA_K", "engine_codes": ["A25A_FKS_2.5_203", "A25A_FXS_2.5_HYB"]},
    {"id": "toyota_rav4_xa50", "name": "RAV4 (XA50)", "make_id": "toyota", "year_start": 2018, "year_end": None, "body_types": ["suv"], "platform": "TNGA_K", "engine_codes": ["A25A_FKS_2.5_203", "A25A_FXS_2.5_HYB"]},
    {"id": "toyota_yaris_mk4", "name": "Yaris IV", "make_id": "toyota", "year_start": 2020, "year_end": None, "body_types": ["hatchback"], "platform": "TNGA_B"},
    {"id": "toyota_gr_yaris", "name": "GR Yaris", "make_id": "toyota", "year_start": 2020, "year_end": None, "body_types": ["hatchback"], "platform": "TNGA_B", "engine_codes": ["G16E_GTS_1.6T_272"]},
    {"id": "toyota_bz4x", "name": "bZ4X", "make_id": "toyota", "year_start": 2022, "year_end": None, "body_types": ["suv"], "platform": "E_TNGA"},

    # Honda
    {"id": "honda_civic_fl", "name": "Civic (FL)", "make_id": "honda", "year_start": 2021, "year_end": None, "body_types": ["hatchback", "sedan"], "platform": "HONDA_GLOBAL_LIGHT"},
    {"id": "honda_civic_fc", "name": "Civic (FC/FK)", "make_id": "honda", "year_start": 2015, "year_end": 2021, "body_types": ["hatchback", "sedan", "coupe"], "platform": "HONDA_GLOBAL_LIGHT"},
    {"id": "honda_accord_cv", "name": "Accord (CV)", "make_id": "honda", "year_start": 2017, "year_end": None, "body_types": ["sedan"], "platform": "HONDA_GLOBAL_LIGHT"},
    {"id": "honda_crv_rw", "name": "CR-V (RW/RT)", "make_id": "honda", "year_start": 2016, "year_end": None, "body_types": ["suv"], "platform": "HONDA_GLOBAL_LIGHT"},

    # Hyundai
    {"id": "hyundai_i30_pd", "name": "i30 (PD)", "make_id": "hyundai", "year_start": 2016, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "K_PLATFORM", "engine_codes": ["G4LD_1.4T_140", "G4FJ_1.6T_177", "D4FE_1.6D_136"]},
    {"id": "hyundai_tucson_nx4", "name": "Tucson (NX4)", "make_id": "hyundai", "year_start": 2020, "year_end": None, "body_types": ["suv"], "platform": "N3_PLATFORM", "engine_codes": ["G4FJ_1.6T_177", "G4KH_2.0T_245", "D4FE_1.6D_136"]},
    {"id": "hyundai_kona_os", "name": "Kona", "make_id": "hyundai", "year_start": 2017, "year_end": None, "body_types": ["suv"], "platform": "K_PLATFORM"},
    {"id": "hyundai_ioniq5", "name": "IONIQ 5", "make_id": "hyundai", "year_start": 2021, "year_end": None, "body_types": ["suv"], "platform": "E_GMP", "engine_codes": ["E_GMP_160KW_RWD", "E_GMP_239KW_AWD"]},
    {"id": "hyundai_ioniq6", "name": "IONIQ 6", "make_id": "hyundai", "year_start": 2022, "year_end": None, "body_types": ["sedan"], "platform": "E_GMP", "engine_codes": ["E_GMP_160KW_RWD", "E_GMP_239KW_AWD"]},

    # Kia
    {"id": "kia_ceed_cd", "name": "Ceed (CD)", "make_id": "kia", "year_start": 2018, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "K_PLATFORM", "engine_codes": ["G4LD_1.4T_140", "G4FJ_1.6T_177", "D4FE_1.6D_136"]},
    {"id": "kia_sportage_nq5", "name": "Sportage (NQ5)", "make_id": "kia", "year_start": 2021, "year_end": None, "body_types": ["suv"], "platform": "N3_PLATFORM", "engine_codes": ["G4FJ_1.6T_177", "G4KH_2.0T_245", "D4FE_1.6D_136"]},
    {"id": "kia_ev6", "name": "EV6", "make_id": "kia", "year_start": 2021, "year_end": None, "body_types": ["suv"], "platform": "E_GMP", "engine_codes": ["E_GMP_160KW_RWD", "E_GMP_239KW_AWD"]},
    {"id": "kia_niro_de", "name": "Niro (DE)", "make_id": "kia", "year_start": 2022, "year_end": None, "body_types": ["suv"], "platform": "K_PLATFORM"},

    # Ford
    {"id": "ford_focus_mk4", "name": "Focus IV", "make_id": "ford", "year_start": 2018, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "C2", "engine_codes": ["ECOBOOST_1.0_125", "ECOBOOST_1.5_150", "ECOBOOST_2.0_250"]},
    {"id": "ford_fiesta_mk8", "name": "Fiesta VIII", "make_id": "ford", "year_start": 2017, "year_end": 2023, "body_types": ["hatchback"], "engine_codes": ["ECOBOOST_1.0_125", "ECOBOOST_1.5_150"]},
    {"id": "ford_mustang_s650", "name": "Mustang (S650)", "make_id": "ford", "year_start": 2023, "year_end": None, "body_types": ["coupe", "convertible"], "engine_codes": ["ECOBOOST_2.3_280"]},
    {"id": "ford_f150_14th", "name": "F-150 (14th gen)", "make_id": "ford", "year_start": 2020, "year_end": None, "body_types": ["pickup"], "platform": "P702"},
    {"id": "ford_mustang_mach_e", "name": "Mustang Mach-E", "make_id": "ford", "year_start": 2020, "year_end": None, "body_types": ["suv"], "platform": "GE2"},

    # Renault
    {"id": "renault_clio_5", "name": "Clio V", "make_id": "renault", "year_start": 2019, "year_end": None, "body_types": ["hatchback"], "platform": "CMF_B", "engine_codes": ["TCE_1.0_100", "TCE_1.3_140"]},
    {"id": "renault_megane_4", "name": "Megane IV", "make_id": "renault", "year_start": 2015, "year_end": 2023, "body_types": ["hatchback", "wagon"], "platform": "CMF_CD", "engine_codes": ["TCE_1.3_140"]},
    {"id": "renault_megane_etech", "name": "Megane E-Tech Electric", "make_id": "renault", "year_start": 2022, "year_end": None, "body_types": ["hatchback"], "platform": "CMF_EV"},
    {"id": "renault_captur_2", "name": "Captur II", "make_id": "renault", "year_start": 2019, "year_end": None, "body_types": ["suv"], "platform": "CMF_B", "engine_codes": ["TCE_1.0_100", "TCE_1.3_140"]},

    # Peugeot
    {"id": "peugeot_208_2", "name": "208 II", "make_id": "peugeot", "year_start": 2019, "year_end": None, "body_types": ["hatchback"], "platform": "CMP", "engine_codes": ["PURETECH_1.2_130"]},
    {"id": "peugeot_308_3", "name": "308 III", "make_id": "peugeot", "year_start": 2021, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "EMP2", "engine_codes": ["PURETECH_1.2_130", "PURETECH_1.6_225"]},
    {"id": "peugeot_3008_2", "name": "3008 II", "make_id": "peugeot", "year_start": 2016, "year_end": None, "body_types": ["suv"], "platform": "EMP2", "engine_codes": ["PURETECH_1.2_130", "PURETECH_1.6_225"]},
    {"id": "peugeot_5008_2", "name": "5008 II", "make_id": "peugeot", "year_start": 2016, "year_end": None, "body_types": ["suv"], "platform": "EMP2"},

    # Tesla
    {"id": "tesla_model_3", "name": "Model 3", "make_id": "tesla", "year_start": 2017, "year_end": None, "body_types": ["sedan"], "platform": "MODEL_3_Y", "engine_codes": ["TESLA_3DM_DUAL"]},
    {"id": "tesla_model_y", "name": "Model Y", "make_id": "tesla", "year_start": 2020, "year_end": None, "body_types": ["suv"], "platform": "MODEL_3_Y", "engine_codes": ["TESLA_3DM_DUAL"]},
    {"id": "tesla_model_s", "name": "Model S", "make_id": "tesla", "year_start": 2012, "year_end": None, "body_types": ["sedan"], "platform": "MODEL_S_X"},
    {"id": "tesla_model_x", "name": "Model X", "make_id": "tesla", "year_start": 2015, "year_end": None, "body_types": ["suv"], "platform": "MODEL_S_X"},

    # Volvo
    {"id": "volvo_xc60_2", "name": "XC60 II", "make_id": "volvo", "year_start": 2017, "year_end": None, "body_types": ["suv"], "platform": "SPA"},
    {"id": "volvo_xc90_2", "name": "XC90 II", "make_id": "volvo", "year_start": 2015, "year_end": None, "body_types": ["suv"], "platform": "SPA"},
    {"id": "volvo_s60_3", "name": "S60 III", "make_id": "volvo", "year_start": 2018, "year_end": None, "body_types": ["sedan"], "platform": "SPA"},
    {"id": "volvo_xc40", "name": "XC40", "make_id": "volvo", "year_start": 2017, "year_end": None, "body_types": ["suv"], "platform": "CMA"},
    {"id": "volvo_ex90", "name": "EX90", "make_id": "volvo", "year_start": 2023, "year_end": None, "body_types": ["suv"], "platform": "SPA2"},

    # Opel
    {"id": "opel_astra_l", "name": "Astra L", "make_id": "opel", "year_start": 2021, "year_end": None, "body_types": ["hatchback", "wagon"], "platform": "EMP2", "engine_codes": ["PURETECH_1.2_130", "PURETECH_1.6_225"]},
    {"id": "opel_astra_k", "name": "Astra K", "make_id": "opel", "year_start": 2015, "year_end": 2021, "body_types": ["hatchback", "wagon"]},
    {"id": "opel_corsa_f", "name": "Corsa F", "make_id": "opel", "year_start": 2019, "year_end": None, "body_types": ["hatchback"], "platform": "CMP", "engine_codes": ["PURETECH_1.2_130"]},
    {"id": "opel_mokka_2", "name": "Mokka II", "make_id": "opel", "year_start": 2020, "year_end": None, "body_types": ["suv"], "platform": "CMP"},

    # Fiat
    {"id": "fiat_500_3", "name": "500 (3rd gen)", "make_id": "fiat", "year_start": 2020, "year_end": None, "body_types": ["hatchback", "convertible"]},
    {"id": "fiat_panda_mk3", "name": "Panda III", "make_id": "fiat", "year_start": 2011, "year_end": None, "body_types": ["hatchback"]},
    {"id": "fiat_tipo", "name": "Tipo", "make_id": "fiat", "year_start": 2015, "year_end": None, "body_types": ["hatchback", "sedan", "wagon"]},

    # Chevrolet
    {"id": "chevy_silverado_4", "name": "Silverado (4th gen)", "make_id": "chevrolet", "year_start": 2018, "year_end": None, "body_types": ["pickup"], "platform": "VSS_F"},
    {"id": "chevy_equinox_3", "name": "Equinox (3rd gen)", "make_id": "chevrolet", "year_start": 2017, "year_end": None, "body_types": ["suv"], "platform": "VSS_S"},
    {"id": "chevy_bolt_ev", "name": "Bolt EV", "make_id": "chevrolet", "year_start": 2016, "year_end": None, "body_types": ["hatchback"]},
    {"id": "chevy_bolt_euv", "name": "Bolt EUV", "make_id": "chevrolet", "year_start": 2021, "year_end": None, "body_types": ["suv"]},

    # Jeep
    {"id": "jeep_wrangler_jl", "name": "Wrangler (JL)", "make_id": "jeep", "year_start": 2017, "year_end": None, "body_types": ["suv"]},
    {"id": "jeep_grand_cherokee_wl", "name": "Grand Cherokee (WL)", "make_id": "jeep", "year_start": 2021, "year_end": None, "body_types": ["suv"]},
    {"id": "jeep_compass_mp", "name": "Compass (MP)", "make_id": "jeep", "year_start": 2016, "year_end": None, "body_types": ["suv"]},

    # Dacia
    {"id": "dacia_sandero_3", "name": "Sandero III", "make_id": "dacia", "year_start": 2020, "year_end": None, "body_types": ["hatchback"], "platform": "CMF_B", "engine_codes": ["TCE_1.0_100"]},
    {"id": "dacia_duster_2", "name": "Duster II", "make_id": "dacia", "year_start": 2017, "year_end": None, "body_types": ["suv"]},
    {"id": "dacia_spring", "name": "Spring", "make_id": "dacia", "year_start": 2021, "year_end": None, "body_types": ["suv"]},
]


# =============================================================================
# COMMON DTC FREQUENCY DATA
# =============================================================================

DTC_FREQUENCIES: List[Dict[str, Any]] = [
    # VW/Audi EA888 Common Issues
    {"dtc_code": "P0171", "make_id": "volkswagen", "engine_code": "EA888_GEN3_2.0_190", "frequency": "common", "occurrence_count": 850, "common_causes": ["PCV valve failure", "Intake manifold leak", "MAF sensor dirty"], "known_fixes": ["Replace PCV valve", "Check intake manifold gaskets", "Clean or replace MAF sensor"]},
    {"dtc_code": "P0420", "make_id": "volkswagen", "engine_code": "EA888_GEN3_2.0_190", "frequency": "common", "occurrence_count": 720, "common_causes": ["Catalytic converter efficiency", "O2 sensor aging", "Engine misfire causing catalyst damage"], "known_fixes": ["Replace catalytic converter", "Replace downstream O2 sensor"]},
    {"dtc_code": "P0302", "make_id": "volkswagen", "model_id": "vw_golf_mk7", "frequency": "uncommon", "occurrence_count": 340, "common_causes": ["Carbon buildup on valves", "Faulty ignition coil", "Worn spark plug"], "known_fixes": ["Walnut blast intake cleaning", "Replace ignition coil", "Replace spark plugs"]},

    # BMW B Series Common Issues
    {"dtc_code": "P0171", "make_id": "bmw", "engine_code": "B48_2.0_258", "frequency": "common", "occurrence_count": 680, "common_causes": ["Valve cover gasket leak", "VANOS solenoid failure", "Intake boot crack"], "known_fixes": ["Replace valve cover gasket", "Replace VANOS solenoid", "Check and replace intake boot"]},
    {"dtc_code": "P0300", "make_id": "bmw", "model_id": "bmw_3series_g20", "frequency": "uncommon", "occurrence_count": 290, "common_causes": ["High-pressure fuel pump failure", "Faulty ignition coil", "Carbon buildup"], "known_fixes": ["Replace HPFP", "Replace ignition coils", "Walnut blast cleaning"]},
    {"dtc_code": "P0016", "make_id": "bmw", "engine_code": "B58_3.0_340", "frequency": "common", "occurrence_count": 520, "common_causes": ["VANOS actuator failure", "Timing chain stretch", "Oil supply issue to VANOS"], "known_fixes": ["Replace VANOS actuator", "Check timing chain", "Verify oil pressure"]},

    # Mercedes Common Issues
    {"dtc_code": "P0171", "make_id": "mercedes", "engine_code": "M264_2.0_184", "frequency": "common", "occurrence_count": 450, "common_causes": ["Air mass sensor failure", "Vacuum leak", "Fuel system lean"], "known_fixes": ["Replace MAF sensor", "Smoke test for vacuum leaks", "Check fuel pressure"]},
    {"dtc_code": "P2004", "make_id": "mercedes", "model_id": "mb_cclass_w205", "frequency": "common", "occurrence_count": 380, "common_causes": ["Intake manifold runner control stuck", "Linkage failure", "Vacuum actuator failure"], "known_fixes": ["Replace intake manifold", "Check and repair linkage"]},

    # Toyota Common Issues
    {"dtc_code": "P0420", "make_id": "toyota", "model_id": "toyota_rav4_xa50", "frequency": "uncommon", "occurrence_count": 280, "common_causes": ["Catalytic converter efficiency", "Short trips causing incomplete warmup", "Engine oil consumption"], "known_fixes": ["Replace catalytic converter", "Address oil consumption issue"]},
    {"dtc_code": "P0171", "make_id": "toyota", "engine_code": "A25A_FKS_2.5_203", "frequency": "rare", "occurrence_count": 120, "common_causes": ["Vacuum leak", "MAF sensor contamination", "Intake gasket leak"], "known_fixes": ["Check vacuum lines", "Clean or replace MAF", "Replace intake gaskets"]},

    # Hyundai/Kia Common Issues
    {"dtc_code": "P0011", "make_id": "hyundai", "engine_code": "G4KH_2.0T_245", "frequency": "common", "occurrence_count": 890, "common_causes": ["Variable valve timing control solenoid", "Oil sludge buildup", "Timing chain issues"], "known_fixes": ["Replace VVT solenoid", "Engine flush", "Check timing chain"]},
    {"dtc_code": "P0300", "make_id": "kia", "model_id": "kia_sportage_nq5", "frequency": "uncommon", "occurrence_count": 210, "common_causes": ["Ignition coil failure", "Fuel injector clog", "Spark plug wear"], "known_fixes": ["Replace ignition coils", "Clean or replace injectors", "Replace spark plugs"]},

    # Ford EcoBoost Common Issues
    {"dtc_code": "P0299", "make_id": "ford", "engine_code": "ECOBOOST_1.5_150", "frequency": "very_common", "occurrence_count": 1200, "common_causes": ["Turbocharger underboost", "Wastegate stuck", "Boost leak"], "known_fixes": ["Replace turbocharger", "Clean/replace wastegate", "Pressure test boost system"]},
    {"dtc_code": "P0171", "make_id": "ford", "engine_code": "ECOBOOST_2.0_250", "frequency": "common", "occurrence_count": 680, "common_causes": ["PCV valve failure", "Intake manifold crack", "Purge valve stuck open"], "known_fixes": ["Replace PCV valve", "Check intake manifold", "Replace purge valve"]},
]


def get_sync_db_url() -> str:
    """Convert async database URL to sync for seeding."""
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def seed_makes(session: Session) -> int:
    """Seed all vehicle makes."""
    count = 0
    for make_data in tqdm(ALL_MAKES, desc="Seeding vehicle makes"):
        existing = session.query(VehicleMake).filter_by(id=make_data["id"]).first()
        if existing:
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


def seed_platforms(session: Session) -> int:
    """Seed vehicle platforms."""
    count = 0
    for platform_data in tqdm(PLATFORMS, desc="Seeding vehicle platforms"):
        existing = session.query(VehiclePlatform).filter_by(code=platform_data["code"]).first()
        if existing:
            continue
        platform = VehiclePlatform(
            code=platform_data["code"],
            name=platform_data["name"],
            manufacturer=platform_data.get("manufacturer"),
            makes=platform_data.get("makes", []),
            year_start=platform_data.get("year_start"),
            year_end=platform_data.get("year_end"),
            segment=platform_data.get("segment"),
            body_types=platform_data.get("body_types", []),
            drivetrain_options=platform_data.get("drivetrain_options", []),
            compatible_engines=platform_data.get("compatible_engines", []),
        )
        session.add(platform)
        count += 1
    session.commit()
    return count


def seed_engines(session: Session) -> int:
    """Seed vehicle engines."""
    count = 0
    for engine_data in tqdm(ENGINES, desc="Seeding vehicle engines"):
        existing = session.query(VehicleEngine).filter_by(code=engine_data["code"]).first()
        if existing:
            continue
        engine = VehicleEngine(
            code=engine_data["code"],
            name=engine_data.get("name"),
            displacement_cc=engine_data.get("displacement_cc"),
            displacement_l=engine_data.get("displacement_l"),
            cylinders=engine_data.get("cylinders"),
            configuration=engine_data.get("configuration"),
            fuel_type=engine_data["fuel_type"],
            aspiration=engine_data.get("aspiration"),
            power_hp=engine_data.get("power_hp"),
            power_kw=engine_data.get("power_kw"),
            torque_nm=engine_data.get("torque_nm"),
            family=engine_data.get("family"),
            manufacturer=engine_data.get("manufacturer"),
            applicable_makes=engine_data.get("applicable_makes", []),
            year_start=engine_data.get("year_start"),
            year_end=engine_data.get("year_end"),
        )
        session.add(engine)
        count += 1
    session.commit()
    return count


def seed_models(session: Session) -> int:
    """Seed vehicle models."""
    count = 0

    # Build platform lookup
    platform_lookup = {}
    for platform in session.query(VehiclePlatform).all():
        platform_lookup[platform.code] = platform.id

    for model_data in tqdm(MODELS, desc="Seeding vehicle models"):
        existing = session.query(VehicleModel).filter_by(id=model_data["id"]).first()
        if existing:
            continue

        # Get platform ID if platform code specified
        platform_id = None
        if model_data.get("platform"):
            platform_id = platform_lookup.get(model_data["platform"])

        model = VehicleModel(
            id=model_data["id"],
            name=model_data["name"],
            make_id=model_data["make_id"],
            year_start=model_data["year_start"],
            year_end=model_data.get("year_end"),
            body_types=model_data.get("body_types", []),
            engine_codes=model_data.get("engine_codes", []),
            platform=model_data.get("platform"),
            platform_id=platform_id,
        )
        session.add(model)
        count += 1
    session.commit()
    return count


def seed_model_engines(session: Session) -> int:
    """Create model-engine relationships."""
    count = 0

    # Build engine lookup
    engine_lookup = {}
    for engine in session.query(VehicleEngine).all():
        engine_lookup[engine.code] = engine.id

    for model_data in tqdm(MODELS, desc="Creating model-engine relationships"):
        model = session.query(VehicleModel).filter_by(id=model_data["id"]).first()
        if not model:
            continue

        engine_codes = model_data.get("engine_codes", [])
        for i, engine_code in enumerate(engine_codes):
            engine_id = engine_lookup.get(engine_code)
            if not engine_id:
                continue

            # Check if relationship already exists
            existing = session.query(VehicleModelEngine).filter_by(
                model_id=model.id,
                engine_id=engine_id
            ).first()
            if existing:
                continue

            model_engine = VehicleModelEngine(
                model_id=model.id,
                engine_id=engine_id,
                year_start=model.year_start,
                year_end=model.year_end,
                is_base=(i == 0),  # First engine is base
            )
            session.add(model_engine)
            count += 1

    session.commit()
    return count


def seed_dtc_frequencies(session: Session) -> int:
    """Seed DTC frequency data."""
    count = 0

    for freq_data in tqdm(DTC_FREQUENCIES, desc="Seeding DTC frequencies"):
        # Check if similar record exists
        query = session.query(VehicleDTCFrequency).filter_by(dtc_code=freq_data["dtc_code"])

        if freq_data.get("make_id"):
            query = query.filter_by(make_id=freq_data["make_id"])
        if freq_data.get("model_id"):
            query = query.filter_by(model_id=freq_data["model_id"])
        if freq_data.get("engine_code"):
            query = query.filter_by(engine_code=freq_data["engine_code"])

        existing = query.first()
        if existing:
            continue

        dtc_freq = VehicleDTCFrequency(
            dtc_code=freq_data["dtc_code"],
            make_id=freq_data.get("make_id"),
            model_id=freq_data.get("model_id"),
            engine_code=freq_data.get("engine_code"),
            frequency=freq_data.get("frequency", "common"),
            occurrence_count=freq_data.get("occurrence_count", 0),
            confidence=freq_data.get("confidence", 0.7),
            source=freq_data.get("source", "seed_data"),
            common_symptoms=freq_data.get("common_symptoms", []),
            common_causes=freq_data.get("common_causes", []),
            known_fixes=freq_data.get("known_fixes", []),
        )
        session.add(dtc_freq)
        count += 1

    session.commit()
    return count


def seed_neo4j_vehicles():
    """Seed vehicle nodes in Neo4j."""
    from backend.app.db.neo4j_models import VehicleNode, DTCNode

    count = 0

    for model_data in tqdm(MODELS, desc="Seeding Neo4j vehicle nodes"):
        # Check if exists
        existing = VehicleNode.nodes.filter(
            make=model_data["make_id"],
            model=model_data["name"]
        ).first_or_none()

        if existing:
            continue

        vehicle = VehicleNode(
            make=model_data["make_id"],
            model=model_data["name"],
            year_start=model_data["year_start"],
            year_end=model_data.get("year_end"),
            platform=model_data.get("platform"),
            engine_codes=model_data.get("engine_codes", []),
        ).save()
        count += 1

        # Create relationships to common DTCs
        for freq_data in DTC_FREQUENCIES:
            if freq_data.get("model_id") == model_data["id"] or freq_data.get("make_id") == model_data["make_id"]:
                dtc_node = DTCNode.nodes.get_or_none(code=freq_data["dtc_code"])
                if dtc_node and not vehicle.has_common_issue.is_connected(dtc_node):
                    vehicle.has_common_issue.connect(dtc_node, {
                        "frequency": freq_data.get("frequency", "common"),
                        "year_start": freq_data.get("year_start") or model_data["year_start"],
                        "year_end": freq_data.get("year_end") or model_data.get("year_end"),
                    })

    return count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Seed AutoCognitix vehicle database with comprehensive data"
    )
    parser.add_argument("--makes", action="store_true", help="Seed makes only")
    parser.add_argument("--models", action="store_true", help="Seed models only")
    parser.add_argument("--engines", action="store_true", help="Seed engines only")
    parser.add_argument("--platforms", action="store_true", help="Seed platforms only")
    parser.add_argument("--dtc-freq", action="store_true", help="Seed DTC frequency data")
    parser.add_argument("--neo4j", action="store_true", help="Seed Neo4j vehicle nodes")
    parser.add_argument("--all", action="store_true", help="Seed all vehicle data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to --all if no specific option
    if not any([args.makes, args.models, args.engines, args.platforms, args.dtc_freq, args.neo4j, args.all]):
        args.all = True

    logger.info("Starting vehicle database seeding...")

    db_url = get_sync_db_url()
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        if args.makes or args.all:
            count = seed_makes(session)
            logger.info(f"Seeded {count} vehicle makes (total: {len(ALL_MAKES)})")

        if args.platforms or args.all:
            count = seed_platforms(session)
            logger.info(f"Seeded {count} vehicle platforms (total: {len(PLATFORMS)})")

        if args.engines or args.all:
            count = seed_engines(session)
            logger.info(f"Seeded {count} vehicle engines (total: {len(ENGINES)})")

        if args.models or args.all:
            count = seed_models(session)
            logger.info(f"Seeded {count} vehicle models (total: {len(MODELS)})")

            # Also create model-engine relationships
            count = seed_model_engines(session)
            logger.info(f"Created {count} model-engine relationships")

        if args.dtc_freq or args.all:
            count = seed_dtc_frequencies(session)
            logger.info(f"Seeded {count} DTC frequency records")

    if args.neo4j or args.all:
        try:
            count = seed_neo4j_vehicles()
            logger.info(f"Seeded {count} Neo4j vehicle nodes")
        except Exception as e:
            logger.warning(f"Neo4j seeding skipped (not connected): {e}")

    logger.info("Vehicle database seeding completed!")

    # Print summary
    print("\n" + "=" * 60)
    print("VEHICLE DATABASE SUMMARY")
    print("=" * 60)
    print(f"Makes:      {len(ALL_MAKES)} ({len(EUROPEAN_MAKES)} European, {len(ASIAN_MAKES)} Asian, {len(AMERICAN_MAKES)} American)")
    print(f"Platforms:  {len(PLATFORMS)}")
    print(f"Engines:    {len(ENGINES)}")
    print(f"Models:     {len(MODELS)}")
    print(f"DTC Freq:   {len(DTC_FREQUENCIES)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
