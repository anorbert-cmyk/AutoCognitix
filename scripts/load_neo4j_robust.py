#!/usr/bin/env python3
"""
Robust Neo4j Batch Loader with Checkpoint Support
- Batch processing (500 records/transaction)
- Retry logic with exponential backoff
- Checkpoint saving for resume capability
- Progress tracking with tqdm
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase
from tqdm import tqdm

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://ae7124c9.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Validate required credentials
if not NEO4J_PASSWORD:
    print("❌ Error: NEO4J_PASSWORD environment variable is required")
    print("Usage: NEO4J_PASSWORD=xxx python load_neo4j_robust.py")
    sys.exit(1)

BATCH_SIZE = 500
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, will be exponentially increased

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_FILE = SCRIPT_DIR / "checkpoints" / "neo4j_checkpoint.json"


class CheckpointManager:
    """Manages checkpoint state for resumable loading."""

    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.state = self._load()

    def _load(self) -> dict:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {
            "vehicles_loaded": False,
            "dtc_loaded": False,
            "complaints_loaded": {},  # year -> last_index
            "recalls_loaded": False,
            "relationships_created": False,
            "last_updated": None
        }

    def save(self):
        self.state["last_updated"] = datetime.now().isoformat()
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def mark_complete(self, key: str, value: Any = True):
        self.state[key] = value
        self.save()


class RobustNeo4jLoader:
    """Robust batch loader for Neo4j with retry and checkpoint support."""

    def __init__(self):
        self.driver = None
        self.checkpoint = CheckpointManager(CHECKPOINT_FILE)
        self.stats = {
            "vehicles": 0,
            "dtc": 0,
            "complaints": 0,
            "recalls": 0,
            "relationships": 0
        }

    async def connect(self):
        """Establish connection to Neo4j."""
        self.driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            max_connection_lifetime=300,
            max_connection_pool_size=50,
            connection_acquisition_timeout=60
        )
        # Test connection
        async with self.driver.session() as session:
            await session.run("RETURN 1")
        print("✅ Connected to Neo4j Aura")

    async def close(self):
        """Close the driver connection."""
        if self.driver:
            await self.driver.close()

    async def execute_with_retry(self, query: str, params: dict = None, retries: int = MAX_RETRIES):
        """Execute a query with retry logic."""
        last_error = None
        for attempt in range(retries):
            try:
                async with self.driver.session() as session:
                    result = await session.run(query, params or {})
                    summary = await result.consume()
                    return summary
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    print(f"  ⚠️ Retry {attempt + 1}/{retries} after {delay}s: {str(e)[:50]}")
                    await asyncio.sleep(delay)
                    # Reconnect
                    try:
                        await self.close()
                        await self.connect()
                    except:
                        pass
        raise last_error

    async def create_indexes(self):
        """Create indexes for better performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (v:Vehicle) ON (v.make, v.model, v.year)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Complaint) ON (c.odi_id)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Recall) ON (r.campaign_number)",
            "CREATE INDEX IF NOT EXISTS FOR (d:DTC) ON (d.code)",
        ]
        for idx in indexes:
            try:
                await self.execute_with_retry(idx)
            except Exception as e:
                print(f"  Index warning: {str(e)[:50]}")
        print("✅ Indexes created/verified")

    async def load_vehicles(self):
        """Load vehicle data from Back4App export."""
        if self.checkpoint.state["vehicles_loaded"]:
            print("⏭️  Vehicles already loaded, skipping...")
            return

        vehicles_file = DATA_DIR / "vehicles" / "back4app_vehicles.json"
        if not vehicles_file.exists():
            print(f"⚠️  Vehicles file not found: {vehicles_file}")
            return

        with open(vehicles_file) as f:
            data = json.load(f)

        makes = data.get("makes", [])
        total_models = sum(len(m.get("models", [])) for m in makes)
        print(f"Loading {total_models} vehicle models from {len(makes)} makes...")

        with tqdm(total=total_models, desc="Vehicles") as pbar:
            for make_data in makes:
                make = make_data.get("name", "Unknown")
                models = make_data.get("models", [])

                # Batch insert models
                for i in range(0, len(models), BATCH_SIZE):
                    batch = models[i:i + BATCH_SIZE]
                    params = {
                        "make": make,
                        "models": [{"name": m.get("name", ""), "year": m.get("year")} for m in batch]
                    }

                    query = """
                    UNWIND $models AS model
                    MERGE (v:Vehicle {make: $make, model: model.name})
                    SET v.year = model.year
                    """

                    await self.execute_with_retry(query, params)
                    pbar.update(len(batch))
                    self.stats["vehicles"] += len(batch)

        self.checkpoint.mark_complete("vehicles_loaded")
        print(f"✅ Loaded {self.stats['vehicles']} vehicles")

    async def load_dtc_codes(self):
        """Load DTC codes from merged file and dtcdb."""
        if self.checkpoint.state["dtc_loaded"]:
            print("⏭️  DTC codes already loaded, skipping...")
            return

        all_codes = []

        # Load merged DTC codes
        merged_file = DATA_DIR / "dtc" / "all_codes_merged.json"
        if merged_file.exists():
            with open(merged_file) as f:
                data = json.load(f)
                codes = data.get("codes", data) if isinstance(data, dict) else data
                if isinstance(codes, list):
                    all_codes.extend(codes)
                    print(f"  Loaded {len(codes)} codes from all_codes_merged.json")

        # Load dtcdb codes
        dtcdb_file = DATA_DIR / "dtc" / "dtcdb_codes.json"
        if dtcdb_file.exists():
            with open(dtcdb_file) as f:
                data = json.load(f)
                codes = data.get("codes", [])
                all_codes.extend(codes)
                print(f"  Loaded {len(codes)} codes from dtcdb_codes.json")

        if not all_codes:
            print("⚠️  No DTC codes found")
            return

        # Deduplicate by code
        seen = set()
        unique_codes = []
        for c in all_codes:
            code = c.get("code", "")
            if code and code not in seen:
                seen.add(code)
                unique_codes.append(c)

        print(f"Loading {len(unique_codes)} unique DTC codes...")

        with tqdm(total=len(unique_codes), desc="DTC Codes") as pbar:
            for i in range(0, len(unique_codes), BATCH_SIZE):
                batch = unique_codes[i:i + BATCH_SIZE]
                params = {"codes": batch}

                query = """
                UNWIND $codes AS dtc
                MERGE (d:DTC {code: dtc.code})
                SET d.description = dtc.description,
                    d.category = dtc.category,
                    d.subcategory = dtc.subcategory
                """

                await self.execute_with_retry(query, params)
                pbar.update(len(batch))
                self.stats["dtc"] += len(batch)

        self.checkpoint.mark_complete("dtc_loaded")
        print(f"✅ Loaded {self.stats['dtc']} DTC codes")

    async def load_complaints(self):
        """Load NHTSA complaints from yearly JSON files."""
        complaints_dir = DATA_DIR / "nhtsa" / "complaints"
        if not complaints_dir.exists():
            print(f"⚠️  Complaints directory not found: {complaints_dir}")
            return

        # Find all complaint files
        complaint_files = sorted(complaints_dir.glob("*.json"))
        if not complaint_files:
            print("⚠️  No complaint files found")
            return

        print(f"Found {len(complaint_files)} complaint files")

        for file_path in complaint_files:
            year = file_path.stem.split("_")[-1] if "_" in file_path.stem else file_path.stem

            # Check checkpoint for this year
            last_index = self.checkpoint.state["complaints_loaded"].get(year, 0)

            with open(file_path) as f:
                data = json.load(f)
                # Handle both formats: list or dict with 'complaints' key
                if isinstance(data, dict):
                    complaints = data.get("complaints", [])
                else:
                    complaints = data

            if last_index >= len(complaints):
                print(f"⏭️  {file_path.name}: already loaded, skipping...")
                continue

            remaining = complaints[last_index:]
            print(f"Loading {len(remaining)} complaints from {file_path.name} (starting at {last_index})...")

            with tqdm(total=len(remaining), desc=f"Complaints {year}") as pbar:
                for i in range(0, len(remaining), BATCH_SIZE):
                    batch = remaining[i:i + BATCH_SIZE]
                    params = {"complaints": [
                        {
                            "odi_id": str(c.get("ODI_ID", c.get("CMPLID", i + last_index))),
                            "make": c.get("MAKETXT", c.get("make", "")),
                            "model": c.get("MODELTXT", c.get("model", "")),
                            "year": c.get("YEARTXT", c.get("year", "")),
                            "component": c.get("COMPNAME", c.get("component", "")),
                            "description": c.get("CDESCR", c.get("description", ""))[:5000],
                            "crash": c.get("CRASH", "N") == "Y",
                            "fire": c.get("FIRE", "N") == "Y",
                            "injuries": int(c.get("INJURED", 0) or 0),
                            "deaths": int(c.get("DEATHS", 0) or 0),
                            "date_received": c.get("DATEA", c.get("date", ""))
                        }
                        for c in batch
                    ]}

                    query = """
                    UNWIND $complaints AS c
                    MERGE (comp:Complaint {odi_id: c.odi_id})
                    SET comp.make = c.make,
                        comp.model = c.model,
                        comp.year = c.year,
                        comp.component = c.component,
                        comp.description = c.description,
                        comp.crash = c.crash,
                        comp.fire = c.fire,
                        comp.injuries = c.injuries,
                        comp.deaths = c.deaths,
                        comp.date_received = c.date_received
                    """

                    await self.execute_with_retry(query, params)
                    pbar.update(len(batch))
                    self.stats["complaints"] += len(batch)

                    # Save checkpoint every batch
                    self.checkpoint.state["complaints_loaded"][year] = last_index + i + len(batch)
                    self.checkpoint.save()

        print(f"✅ Loaded {self.stats['complaints']} complaints")

    async def load_recalls(self):
        """Load NHTSA recalls from JSON files."""
        if self.checkpoint.state["recalls_loaded"]:
            print("⏭️  Recalls already loaded, skipping...")
            return

        recalls_dir = DATA_DIR / "nhtsa" / "recalls"
        if not recalls_dir.exists():
            print(f"⚠️  Recalls directory not found: {recalls_dir}")
            return

        # Collect all recalls
        all_recalls = []
        for file_path in recalls_dir.glob("*.json"):
            with open(file_path) as f:
                data = json.load(f)
                recalls = data if isinstance(data, list) else data.get("recalls", [])
                all_recalls.extend(recalls)

        if not all_recalls:
            print("⚠️  No recalls found")
            return

        print(f"Loading {len(all_recalls)} recalls...")

        with tqdm(total=len(all_recalls), desc="Recalls") as pbar:
            for i in range(0, len(all_recalls), BATCH_SIZE):
                batch = all_recalls[i:i + BATCH_SIZE]
                params = {"recalls": [
                    {
                        "campaign_number": r.get("NHTSACampaignNumber", r.get("campaign_number", f"UNK_{i}")),
                        "make": r.get("Make", r.get("make", "")),
                        "model": r.get("Model", r.get("model", "")),
                        "year": str(r.get("ModelYear", r.get("year", ""))),
                        "component": r.get("Component", r.get("component", "")),
                        "summary": (r.get("Summary", r.get("summary", "")) or "")[:5000],
                        "consequence": (r.get("Consequence", r.get("consequence", "")) or "")[:2000],
                        "remedy": (r.get("Remedy", r.get("remedy", "")) or "")[:2000],
                        "report_date": r.get("ReportReceivedDate", r.get("date", ""))
                    }
                    for r in batch
                ]}

                query = """
                UNWIND $recalls AS r
                MERGE (rec:Recall {campaign_number: r.campaign_number})
                SET rec.make = r.make,
                    rec.model = r.model,
                    rec.year = r.year,
                    rec.component = r.component,
                    rec.summary = r.summary,
                    rec.consequence = r.consequence,
                    rec.remedy = r.remedy,
                    rec.report_date = r.report_date
                """

                await self.execute_with_retry(query, params)
                pbar.update(len(batch))
                self.stats["recalls"] += len(batch)

        self.checkpoint.mark_complete("recalls_loaded")
        print(f"✅ Loaded {self.stats['recalls']} recalls")

    async def create_relationships(self):
        """Create relationships between nodes."""
        if self.checkpoint.state["relationships_created"]:
            print("⏭️  Relationships already created, skipping...")
            return

        print("Creating relationships...")

        # Vehicle -> Complaint relationships
        print("  Creating Vehicle-Complaint relationships...")
        query = """
        MATCH (c:Complaint)
        WHERE c.make IS NOT NULL AND c.model IS NOT NULL
        MATCH (v:Vehicle {make: c.make, model: c.model})
        MERGE (v)-[:HAS_COMPLAINT]->(c)
        """
        await self.execute_with_retry(query)

        # Vehicle -> Recall relationships
        print("  Creating Vehicle-Recall relationships...")
        query = """
        MATCH (r:Recall)
        WHERE r.make IS NOT NULL AND r.model IS NOT NULL
        MATCH (v:Vehicle {make: r.make, model: r.model})
        MERGE (v)-[:HAS_RECALL]->(r)
        """
        await self.execute_with_retry(query)

        # DTC mentioned in Complaint
        print("  Creating DTC-Complaint relationships...")
        query = """
        MATCH (d:DTC), (c:Complaint)
        WHERE c.description CONTAINS d.code
        MERGE (d)-[:MENTIONED_IN]->(c)
        """
        await self.execute_with_retry(query)

        self.checkpoint.mark_complete("relationships_created")
        print("✅ Relationships created")

    async def run(self, resume: bool = True):
        """Run the full loading process."""
        print("=" * 60)
        print("ROBUST NEO4J BATCH LOADER")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Resume mode: {resume}")
        if resume and self.checkpoint.state["last_updated"]:
            print(f"Last checkpoint: {self.checkpoint.state['last_updated']}")
        print()

        try:
            await self.connect()
            await self.create_indexes()

            await self.load_vehicles()
            await self.load_dtc_codes()
            await self.load_complaints()
            await self.load_recalls()
            await self.create_relationships()

            print()
            print("=" * 60)
            print("LOADING COMPLETE")
            print("=" * 60)
            print(f"Vehicles: {self.stats['vehicles']:,}")
            print(f"DTC Codes: {self.stats['dtc']:,}")
            print(f"Complaints: {self.stats['complaints']:,}")
            print(f"Recalls: {self.stats['recalls']:,}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Progress saved to checkpoint. Run with --resume to continue.")
            raise
        finally:
            await self.close()


async def main():
    resume = "--resume" in sys.argv or "-r" in sys.argv
    loader = RobustNeo4jLoader()
    await loader.run(resume=resume)


if __name__ == "__main__":
    asyncio.run(main())
