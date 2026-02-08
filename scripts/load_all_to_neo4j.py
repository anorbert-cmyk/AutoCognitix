#!/usr/bin/env python3
"""
Load ALL data sources into Neo4j Aura.

This script consolidates data from:
- NHTSA Complaints (31K+)
- NHTSA Recalls (821+)
- Back4App Vehicles (63 makes, 1145 models)
- dtcdb DTC codes (467)
- OBDb parsed vehicles (732)

And creates a unified graph with relationships.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from neo4j import AsyncGraphDatabase
from tqdm import tqdm

# Neo4j Aura connection - Load from environment variables
# Set these in your .env file or export them before running
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# Validate required credentials
if not NEO4J_URI or not NEO4J_PASSWORD:
    print("Error: NEO4J_URI and NEO4J_PASSWORD environment variables are required")
    print("Usage: NEO4J_URI=neo4j+s://xxx.databases.neo4j.io NEO4J_PASSWORD=xxx python load_all_to_neo4j.py")
    print("Or set them in your .env file")
    sys.exit(1)

DATA_DIR = Path(__file__).parent.parent / "data"


def extract_dtc_codes(text: str) -> list[str]:
    """Extract DTC codes from text (P0XXX, C0XXX, B0XXX, U0XXX pattern)."""
    if not text:
        return []
    pattern = r'\b[PCBU][0-9A-F]{4}\b'
    return list(set(re.findall(pattern, text.upper())))


class Neo4jLoader:
    def __init__(self):
        self.driver = None
        self.stats = {
            "vehicles": 0,
            "complaints": 0,
            "recalls": 0,
            "dtc_codes": 0,
            "relationships": 0
        }

    async def connect(self):
        """Connect to Neo4j Aura."""
        self.driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        # Verify connection
        async with self.driver.session() as session:
            result = await session.run("RETURN 1 AS test")
            await result.single()
        print(f"✅ Connected to Neo4j Aura")

    async def close(self):
        """Close connection."""
        if self.driver:
            await self.driver.close()

    async def create_indexes(self):
        """Create indexes for better performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.make, v.model, v.year)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Complaint) ON (c.odi_number)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Recall) ON (r.campaign_number)",
            "CREATE INDEX IF NOT EXISTS FOR (d:DTC) ON (d.code)",
        ]
        async with self.driver.session() as session:
            for idx in indexes:
                try:
                    await session.run(idx)
                except Exception as e:
                    print(f"  Index warning: {e}")
        print("✅ Indexes created")

    async def load_vehicles(self):
        """Load vehicles from Back4App data."""
        vehicles_file = DATA_DIR / "vehicles" / "back4app_vehicles.json"
        if not vehicles_file.exists():
            print("⚠️ No Back4App vehicles file found")
            return

        with open(vehicles_file) as f:
            data = json.load(f)

        makes = data.get("makes", [])
        total = sum(len(m.get("models", [])) for m in makes)

        print(f"Loading {total} vehicle models...")

        async with self.driver.session() as session:
            for make_data in tqdm(makes, desc="Makes"):
                make_name = make_data.get("name", "").upper()
                for model_data in make_data.get("models", []):
                    model_name = model_data.get("name", "")
                    years = model_data.get("years", [])

                    for year in years:
                        await session.run("""
                            MERGE (v:Vehicle {make: $make, model: $model, year: $year})
                            SET v.updated_at = datetime()
                        """, make=make_name, model=model_name, year=year)
                        self.stats["vehicles"] += 1

        print(f"✅ Loaded {self.stats['vehicles']} vehicles")

    async def load_complaints(self):
        """Load NHTSA complaints."""
        complaints_file = DATA_DIR / "nhtsa" / "complaints.json"
        if not complaints_file.exists():
            print("⚠️ No complaints file found")
            return

        with open(complaints_file) as f:
            data = json.load(f)

        complaints = data.get("complaints", data if isinstance(data, list) else [])
        print(f"Loading {len(complaints)} complaints...")

        async with self.driver.session() as session:
            batch_size = 100
            for i in tqdm(range(0, len(complaints), batch_size), desc="Complaints"):
                batch = complaints[i:i+batch_size]

                for complaint in batch:
                    odi_number = complaint.get("odi_number") or complaint.get("ODI_ID")
                    if not odi_number:
                        continue

                    make = (complaint.get("make") or complaint.get("MAKETXT", "")).upper()
                    model = complaint.get("model") or complaint.get("MODELTXT", "")
                    year = complaint.get("model_year") or complaint.get("YEARTXT")
                    summary = complaint.get("summary") or complaint.get("CDESCR", "")
                    component = complaint.get("component") or complaint.get("COMPDESC", "")

                    # Extract DTC codes from summary
                    dtc_codes = extract_dtc_codes(summary)

                    try:
                        year_int = int(year) if year else None
                    except:
                        year_int = None

                    # Create Complaint node
                    await session.run("""
                        MERGE (c:Complaint {odi_number: $odi_number})
                        SET c.make = $make,
                            c.model = $model,
                            c.year = $year,
                            c.summary = $summary,
                            c.component = $component,
                            c.crash = $crash,
                            c.fire = $fire,
                            c.injuries = $injuries,
                            c.deaths = $deaths,
                            c.updated_at = datetime()
                    """,
                        odi_number=str(odi_number),
                        make=make,
                        model=model,
                        year=year_int,
                        summary=summary[:5000] if summary else "",  # Limit size
                        component=component,
                        crash=complaint.get("crash") or complaint.get("CRASH", "N") == "Y",
                        fire=complaint.get("fire") or complaint.get("FIRE", "N") == "Y",
                        injuries=int(complaint.get("injuries") or complaint.get("INJURED", 0) or 0),
                        deaths=int(complaint.get("deaths") or complaint.get("DEATHS", 0) or 0)
                    )

                    # Link to Vehicle if exists
                    if make and model and year_int:
                        await session.run("""
                            MATCH (c:Complaint {odi_number: $odi_number})
                            MERGE (v:Vehicle {make: $make, model: $model, year: $year})
                            MERGE (v)-[:HAS_COMPLAINT]->(c)
                        """, odi_number=str(odi_number), make=make, model=model, year=year_int)
                        self.stats["relationships"] += 1

                    # Link to DTC codes
                    for dtc_code in dtc_codes:
                        await session.run("""
                            MATCH (c:Complaint {odi_number: $odi_number})
                            MERGE (d:DTC {code: $code})
                            MERGE (d)-[:MENTIONED_IN]->(c)
                        """, odi_number=str(odi_number), code=dtc_code)
                        self.stats["relationships"] += 1

                    self.stats["complaints"] += 1

        print(f"✅ Loaded {self.stats['complaints']} complaints")

    async def load_recalls(self):
        """Load NHTSA recalls."""
        recalls_file = DATA_DIR / "nhtsa" / "recalls.json"
        if not recalls_file.exists():
            print("⚠️ No recalls file found")
            return

        with open(recalls_file) as f:
            data = json.load(f)

        recalls = data.get("recalls", data if isinstance(data, list) else [])
        print(f"Loading {len(recalls)} recalls...")

        async with self.driver.session() as session:
            for recall in tqdm(recalls, desc="Recalls"):
                campaign = recall.get("campaign_number") or recall.get("NHTSACampaignNumber")
                if not campaign:
                    continue

                make = (recall.get("make") or recall.get("Make", "")).upper()
                model = recall.get("model") or recall.get("Model", "")
                year = recall.get("model_year") or recall.get("ModelYear")
                summary = recall.get("summary") or recall.get("Summary", "")
                component = recall.get("component") or recall.get("Component", "")
                remedy = recall.get("remedy") or recall.get("Remedy", "")

                try:
                    year_int = int(year) if year else None
                except:
                    year_int = None

                # Create Recall node
                await session.run("""
                    MERGE (r:Recall {campaign_number: $campaign})
                    SET r.make = $make,
                        r.model = $model,
                        r.year = $year,
                        r.summary = $summary,
                        r.component = $component,
                        r.remedy = $remedy,
                        r.manufacturer = $manufacturer,
                        r.updated_at = datetime()
                """,
                    campaign=campaign,
                    make=make,
                    model=model,
                    year=year_int,
                    summary=summary[:5000] if summary else "",
                    component=component,
                    remedy=remedy[:2000] if remedy else "",
                    manufacturer=recall.get("manufacturer") or recall.get("Manufacturer", "")
                )

                # Link to Vehicle
                if make and model and year_int:
                    await session.run("""
                        MATCH (r:Recall {campaign_number: $campaign})
                        MERGE (v:Vehicle {make: $make, model: $model, year: $year})
                        MERGE (v)-[:HAS_RECALL]->(r)
                    """, campaign=campaign, make=make, model=model, year=year_int)
                    self.stats["relationships"] += 1

                # Link to Component
                if component:
                    await session.run("""
                        MATCH (r:Recall {campaign_number: $campaign})
                        MERGE (comp:Component {name: $component})
                        MERGE (comp)-[:AFFECTED_BY]->(r)
                    """, campaign=campaign, component=component)
                    self.stats["relationships"] += 1

                self.stats["recalls"] += 1

        print(f"✅ Loaded {self.stats['recalls']} recalls")

    async def load_dtcdb_codes(self):
        """Load DTC codes from dtcdb."""
        dtcdb_file = DATA_DIR / "dtc" / "dtcdb_codes.json"
        if not dtcdb_file.exists():
            print("⚠️ No dtcdb codes file found")
            return

        with open(dtcdb_file) as f:
            data = json.load(f)

        # Handle nested structure with "codes" key
        codes = data.get("codes", data if isinstance(data, list) else [])

        print(f"Loading {len(codes)} DTC codes from dtcdb...")

        async with self.driver.session() as session:
            for code_data in tqdm(codes, desc="DTC Codes"):
                code = code_data.get("code", "")
                if not code:
                    continue

                await session.run("""
                    MERGE (d:DTC {code: $code})
                    SET d.description_en = $description,
                        d.category = $category,
                        d.subcategory = $subcategory,
                        d.source = 'dtcdb',
                        d.updated_at = datetime()
                """,
                    code=code,
                    description=code_data.get("description", ""),
                    category=code_data.get("category", ""),
                    subcategory=code_data.get("subcategory", "")
                )
                self.stats["dtc_codes"] += 1

        print(f"✅ Loaded {self.stats['dtc_codes']} DTC codes from dtcdb")

    async def print_stats(self):
        """Print final statistics."""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (n)
                WITH labels(n) AS labels, count(n) AS count
                RETURN labels, count
                ORDER BY count DESC
            """)
            records = await result.data()

        print("\n" + "=" * 60)
        print("NEO4J AURA STATISTICS")
        print("=" * 60)

        total_nodes = 0
        for record in records:
            labels = record["labels"]
            count = record["count"]
            label_str = ":".join(labels) if labels else "Unknown"
            print(f"  {label_str}: {count:,}")
            total_nodes += count

        print("-" * 60)
        print(f"  TOTAL NODES: {total_nodes:,}")
        print(f"  RELATIONSHIPS CREATED: {self.stats['relationships']:,}")
        print("=" * 60)


async def main():
    print("=" * 60)
    print("LOADING ALL DATA TO NEO4J AURA")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    loader = Neo4jLoader()

    try:
        await loader.connect()
        await loader.create_indexes()

        # Load all data sources
        await loader.load_vehicles()
        await loader.load_dtcdb_codes()
        await loader.load_complaints()
        await loader.load_recalls()

        # Print final stats
        await loader.print_stats()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await loader.close()


if __name__ == "__main__":
    asyncio.run(main())
