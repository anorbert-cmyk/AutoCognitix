"""Seed vehicle makes and models with European manufacturers.

Revision ID: 009_seed_vehicle_makes_models
Revises: 008_add_fk_constraints
Create Date: 2026-02-09 14:00:00.000000

This migration seeds the vehicle_makes and vehicle_models tables with
initial data for European manufacturers. Idempotent - skips if data exists.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '009_seed_vehicle_makes_models'
down_revision: str = '008_add_fk_constraints'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# European vehicle manufacturers (Top 20)
MAKES = [
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

MODELS = [
    {"id": "vw_golf", "name": "Golf", "make_id": "volkswagen", "year_start": 1974},
    {"id": "vw_passat", "name": "Passat", "make_id": "volkswagen", "year_start": 1973},
    {"id": "vw_tiguan", "name": "Tiguan", "make_id": "volkswagen", "year_start": 2007},
    {"id": "vw_polo", "name": "Polo", "make_id": "volkswagen", "year_start": 1975},
    {"id": "bmw_3series", "name": "3 Series", "make_id": "bmw", "year_start": 1975},
    {"id": "bmw_5series", "name": "5 Series", "make_id": "bmw", "year_start": 1972},
    {"id": "bmw_x3", "name": "X3", "make_id": "bmw", "year_start": 2003},
    {"id": "bmw_x5", "name": "X5", "make_id": "bmw", "year_start": 1999},
    {"id": "mb_cclass", "name": "C-Class", "make_id": "mercedes", "year_start": 1993},
    {"id": "mb_eclass", "name": "E-Class", "make_id": "mercedes", "year_start": 1993},
    {"id": "mb_glc", "name": "GLC", "make_id": "mercedes", "year_start": 2015},
    {"id": "audi_a3", "name": "A3", "make_id": "audi", "year_start": 1996},
    {"id": "audi_a4", "name": "A4", "make_id": "audi", "year_start": 1994},
    {"id": "audi_a6", "name": "A6", "make_id": "audi", "year_start": 1994},
    {"id": "audi_q5", "name": "Q5", "make_id": "audi", "year_start": 2008},
    {"id": "skoda_octavia", "name": "Octavia", "make_id": "skoda", "year_start": 1996},
    {"id": "skoda_fabia", "name": "Fabia", "make_id": "skoda", "year_start": 1999},
    {"id": "skoda_superb", "name": "Superb", "make_id": "skoda", "year_start": 2001},
    {"id": "renault_clio", "name": "Clio", "make_id": "renault", "year_start": 1990},
    {"id": "renault_megane", "name": "Megane", "make_id": "renault", "year_start": 1995},
    {"id": "peugeot_208", "name": "208", "make_id": "peugeot", "year_start": 2012},
    {"id": "peugeot_308", "name": "308", "make_id": "peugeot", "year_start": 2007},
    {"id": "peugeot_3008", "name": "3008", "make_id": "peugeot", "year_start": 2009},
    {"id": "citroen_c3", "name": "C3", "make_id": "citroen", "year_start": 2002},
    {"id": "citroen_c4", "name": "C4", "make_id": "citroen", "year_start": 2004},
    {"id": "fiat_500", "name": "500", "make_id": "fiat", "year_start": 2007},
    {"id": "fiat_panda", "name": "Panda", "make_id": "fiat", "year_start": 1980},
    {"id": "volvo_xc60", "name": "XC60", "make_id": "volvo", "year_start": 2008},
    {"id": "volvo_xc90", "name": "XC90", "make_id": "volvo", "year_start": 2002},
    {"id": "volvo_v60", "name": "V60", "make_id": "volvo", "year_start": 2010},
    {"id": "opel_astra", "name": "Astra", "make_id": "opel", "year_start": 1991},
    {"id": "opel_corsa", "name": "Corsa", "make_id": "opel", "year_start": 1982},
]


def upgrade() -> None:
    conn = op.get_bind()

    # Seed makes (idempotent)
    existing = conn.execute(sa.text("SELECT COUNT(*) FROM vehicle_makes")).scalar()
    if existing == 0:
        for make in MAKES:
            conn.execute(
                sa.text(
                    "INSERT INTO vehicle_makes (id, name, country) VALUES (:id, :name, :country) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                make,
            )

    # Seed models (idempotent)
    existing = conn.execute(sa.text("SELECT COUNT(*) FROM vehicle_models")).scalar()
    if existing == 0:
        for model in MODELS:
            conn.execute(
                sa.text(
                    "INSERT INTO vehicle_models (id, name, make_id, year_start) "
                    "VALUES (:id, :name, :make_id, :year_start) "
                    "ON CONFLICT (id) DO NOTHING"
                ),
                model,
            )


def downgrade() -> None:
    conn = op.get_bind()
    for model in MODELS:
        conn.execute(sa.text("DELETE FROM vehicle_models WHERE id = :id"), {"id": model["id"]})
    for make in MAKES:
        conn.execute(sa.text("DELETE FROM vehicle_makes WHERE id = :id"), {"id": make["id"]})
