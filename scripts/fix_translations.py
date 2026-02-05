#!/usr/bin/env python3
"""
Fix Hungarian Translations in AutoCognitix

This script applies the translation glossary to correct mistranslated DTC codes.
Common mistranslations fixed:
- "erdő" (forest) -> "föld/test" (ground) - from "ground" mistranslation
- "hűtő" (cooler/refrigerator) -> "akkumulátor" (battery) - in battery context
- Various other automotive term corrections

Usage:
    python scripts/fix_translations.py --dry-run     # Preview changes
    python scripts/fix_translations.py --apply       # Apply to cache file
    python scripts/fix_translations.py --database    # Also update databases
"""

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# File paths
GLOSSARY_PATH = PROJECT_ROOT / "data" / "translation_glossary.json"
TRANSLATION_CACHE_PATH = PROJECT_ROOT / "data" / "dtc_codes" / "translation_cache.json"
BACKUP_PATH = PROJECT_ROOT / "data" / "dtc_codes" / "translation_cache.backup.json"


class TranslationFixer:
    """Applies glossary-based corrections to Hungarian translations."""

    def __init__(self):
        self.glossary: Dict[str, Dict[str, str]] = {}
        self.fix_patterns: List[Tuple[re.Pattern, str, str]] = []
        self.stats = defaultdict(int)

    def load_glossary(self) -> bool:
        """
        Load the translation glossary from JSON file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.glossary = data.get("terms", {})
            logger.info(f"Loaded glossary with {len(self.glossary)} terms")
            return True

        except FileNotFoundError:
            logger.error(f"Glossary file not found: {GLOSSARY_PATH}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in glossary: {e}")
            return False

    def build_fix_patterns(self) -> None:
        """Build regex patterns for common mistranslations."""
        # Pattern format: (regex_pattern, replacement, description)

        # 1. Fix "erdő" (forest) - should be "föld" (ground) or "test" (chassis ground)
        self.fix_patterns.extend([
            # "erdőhöz rövidítés" -> "testre rövidítés" (short to ground)
            (re.compile(r'\berdőhöz\s+rövidítés\b', re.IGNORECASE),
             'testre rövidítés', 'short to ground'),
            (re.compile(r'\berdőhoz\s+rövidítés\b', re.IGNORECASE),
             'testre rövidítés', 'short to ground'),
            # "az erdőhez" -> "a testre" or "a földre"
            (re.compile(r'\baz\s+erdőhez\b', re.IGNORECASE),
             'a testre', 'to ground'),
            (re.compile(r'\baz\s+erdőhöz\b', re.IGNORECASE),
             'a testre', 'to ground'),
            # "erdőre rövidítés" -> "testre rövidítés"
            (re.compile(r'\berdőre\s+rövidítés\b', re.IGNORECASE),
             'testre rövidítés', 'short to ground'),
            # Standalone "erdő" at word boundaries in circuit context
            (re.compile(r'\berdő\b(?=\s*(hálózat|körülmény|hiba|áramkör))', re.IGNORECASE),
             'test', 'ground'),
            # "erdőzítés" -> "földelés" (grounding)
            (re.compile(r'\berdőzítés[ea]?\b', re.IGNORECASE),
             'földelés', 'grounding'),
            # "erdhez" -> "testre" (to ground)
            (re.compile(r'\berdhez\b', re.IGNORECASE),
             'testre', 'to ground'),
            # "kerek erdőhöz" -> "rövidzár testre"
            (re.compile(r'\bkerek\s+erdőhöz\b', re.IGNORECASE),
             'rövidzár testre', 'short to ground'),
        ])

        # 2. Fix "hűtő" in battery context (should be "akkumulátor")
        self.fix_patterns.extend([
            # "a hűtőhoz" in circuit context -> "az akkumulátorhoz"
            (re.compile(r'\ba\s+hűtőhoz\b(?=.*(?:rövidítés|áramkör|hálózat))', re.IGNORECASE),
             'az akkumulátorhoz', 'to battery'),
            (re.compile(r'\ba\s+hűtőhöz\b(?=.*(?:rövidítés|áramkör|hálózat))', re.IGNORECASE),
             'az akkumulátorhoz', 'to battery'),
            # "akárnyomás a hűtőhoz" -> "rövidzár az akkumulátorra"
            (re.compile(r'\bakárnyomás\s+a\s+hűtő(hoz|höz)\b', re.IGNORECASE),
             'rövidzár az akkumulátorra', 'short to battery'),
            # "akarszánt a hűtőhez" -> "rövidzár az akkumulátorra"
            (re.compile(r'\bakarszánt\s+a\s+hűtőhöz\b', re.IGNORECASE),
             'rövidzár az akkumulátorra', 'short to battery'),
            # "hűtőre kortslutva" -> "akkumulátorra rövidítve"
            (re.compile(r'\bhűtőre\s+kortslutva\b', re.IGNORECASE),
             'akkumulátorra rövidítve', 'shorted to battery'),
            # "kerek bateriához" -> "rövidzár az akkumulátorra"
            (re.compile(r'\bkerek\s+bateriához\b', re.IGNORECASE),
             'rövidzár az akkumulátorra', 'short to battery'),
            # "batteriához rövidítés" -> "akkumulátorra rövidítés"
            (re.compile(r'\bbatter?iához\s+rövidítés\b', re.IGNORECASE),
             'akkumulátorra rövidítés', 'short to battery'),
            # "szélre erdőzítése" -> "zárlat a tápfeszültségre" (short to power)
            (re.compile(r'\bszélre\s+erdőzítése\b', re.IGNORECASE),
             'zárlat a tápfeszültségre', 'short to power'),
        ])

        # 3. Fix circuit-related terms
        self.fix_patterns.extend([
            # "nyomkör" -> "áramkör" (circuit)
            (re.compile(r'\bnyomkör\b', re.IGNORECASE),
             'áramkör', 'circuit'),
            # "kórtérvétel" -> "áramkör" (circuit)
            (re.compile(r'\bkórtérvétel\b', re.IGNORECASE),
             'áramkör', 'circuit'),
            # "kольota" (Cyrillic mixed) -> "áramkör" (circuit)
            (re.compile(r'\bkольota\b', re.IGNORECASE),
             'áramkör', 'circuit'),
            # "circuits" (English leftover) -> "áramkör"
            (re.compile(r'\b-?circuits?\b', re.IGNORECASE),
             ' áramkör', 'circuit'),
            # "cirkusza" -> "áramkör"
            (re.compile(r'\bcirkusz[aá]\b', re.IGNORECASE),
             'áramkör', 'circuit'),
            # "keringés" in circuit context -> "áramkör hiba"
            (re.compile(r'\bkeringés\b(?=\s*$)', re.IGNORECASE),
             'áramkör hiba', 'circuit malfunction'),
        ])

        # 4. Fix "nyitott" / "üres" for open circuit
        self.fix_patterns.extend([
            # "sáv nyitott" -> "áramkör szakadt"
            (re.compile(r'\bsáv\s+nyitott\b', re.IGNORECASE),
             'áramkör szakadt', 'open circuit'),
            # "kablója nyitott" -> "áramkör szakadt"
            (re.compile(r'\b(kablója|kabló)\s+nyitott\b', re.IGNORECASE),
             'áramkör szakadt', 'open circuit'),
            # "üres" at end (meaning open) -> "szakadt"
            (re.compile(r'\bnyomkör\s+üres\b', re.IGNORECASE),
             'áramkör szakadt', 'open circuit'),
        ])

        # 5. Fix sensor-related terms
        self.fix_patterns.extend([
            # "szenszor" -> "szenzor"
            (re.compile(r'\bszenszor\b', re.IGNORECASE),
             'szenzor', 'sensor'),
            # "senzor" -> "szenzor"
            (re.compile(r'\bsenzor\b', re.IGNORECASE),
             'szenzor', 'sensor'),
        ])

        # 6. Fix common gibberish translations
        self.fix_patterns.extend([
            # "kúzdulása" -> "hibája" (fault)
            (re.compile(r'\bkúzdulás[aá]\b', re.IGNORECASE),
             'hibája', 'fault'),
            # "hivatala" -> "hibája" (fault)
            (re.compile(r'\bhivatala\b', re.IGNORECASE),
             'hibája', 'fault'),
            # "hibáság" -> "hiba" (fault)
            (re.compile(r'\bhibáság\b', re.IGNORECASE),
             'hiba', 'fault'),
            # "kilépett" -> "szakadt" (open)
            (re.compile(r'\bkilépett\b', re.IGNORECASE),
             'szakadt', 'open'),
        ])

        # 7. Fix "föld" / "test" variants
        self.fix_patterns.extend([
            # "a földre" in circuit context
            (re.compile(r'\ba\s+földre\b', re.IGNORECASE),
             'a testre', 'to ground'),
            # "földhöz csatlakoztatva" -> "testre rövidítve"
            (re.compile(r'\bföldhöz\s+csatlakoztatva\b', re.IGNORECASE),
             'testre rövidítve', 'shorted to ground'),
            # "kurzszt a földhöz" -> "rövidzár a testre"
            (re.compile(r'\bkurzszt\s+a\s+földhöz\b', re.IGNORECASE),
             'rövidzár a testre', 'short to ground'),
        ])

        # 8. Fix "battery" variants
        self.fix_patterns.extend([
            # "akkumulátorhoz csatlakoztatva" -> "akkumulátorra rövidítve"
            (re.compile(r'\bakkumulátorhoz\s+csatlakoztatva\b', re.IGNORECASE),
             'akkumulátorra rövidítve', 'shorted to battery'),
            # "kurzszt a gyarmatkhoz" -> "rövidzár az akkumulátorra"
            (re.compile(r'\bkurzszt\s+a\s+gyarmatkhoz\b', re.IGNORECASE),
             'rövidzár az akkumulátorra', 'short to battery'),
            # "gyarmatkhoz" alone -> "akkumulátorra"
            (re.compile(r'\bgyarmatk?hoz\b', re.IGNORECASE),
             'akkumulátorra', 'to battery'),
        ])

        # 9. Fix component names
        self.fix_patterns.extend([
            # "relais" -> "relé"
            (re.compile(r'\brelais\b', re.IGNORECASE),
             'relé', 'relay'),
            # "lampa" -> "lámpa"
            (re.compile(r'\blampa\b', re.IGNORECASE),
             'lámpa', 'lamp'),
            # "kabló" -> "kábel"
            (re.compile(r'\bkabló\b', re.IGNORECASE),
             'kábel', 'cable'),
            # "kablója" -> "kábele"
            (re.compile(r'\bkablója\b', re.IGNORECASE),
             'kábele', 'cable'),
        ])

        logger.info(f"Built {len(self.fix_patterns)} fix patterns")

    def fix_translation(self, text: str) -> Tuple[str, List[str]]:
        """
        Apply all fix patterns to a translation.

        Args:
            text: Original Hungarian translation text.

        Returns:
            Tuple of (corrected_text, list_of_changes_made).
        """
        if not text:
            return text, []

        changes = []
        corrected = text

        for pattern, replacement, description in self.fix_patterns:
            if pattern.search(corrected):
                new_text = pattern.sub(replacement, corrected)
                if new_text != corrected:
                    changes.append(f"{description}: '{pattern.pattern}' -> '{replacement}'")
                    corrected = new_text

        return corrected, changes

    def load_translations(self) -> Optional[Dict[str, str]]:
        """
        Load translation cache from JSON file.

        Returns:
            Dictionary of DTC code -> Hungarian translation, or None on error.
        """
        try:
            with open(TRANSLATION_CACHE_PATH, "r", encoding="utf-8") as f:
                translations = json.load(f)

            logger.info(f"Loaded {len(translations)} translations from cache")
            return translations

        except FileNotFoundError:
            logger.error(f"Translation cache not found: {TRANSLATION_CACHE_PATH}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in translation cache: {e}")
            return None

    def save_translations(self, translations: Dict[str, str]) -> bool:
        """
        Save corrected translations to cache file.

        Args:
            translations: Dictionary of DTC code -> Hungarian translation.

        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            # Create backup first
            if TRANSLATION_CACHE_PATH.exists():
                import shutil
                shutil.copy2(TRANSLATION_CACHE_PATH, BACKUP_PATH)
                logger.info(f"Created backup at {BACKUP_PATH}")

            with open(TRANSLATION_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(translations)} translations to cache")
            return True

        except Exception as e:
            logger.error(f"Failed to save translations: {e}")
            return False

    def fix_all_translations(
        self,
        translations: Dict[str, str],
        dry_run: bool = True
    ) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str, List[str]]]]:
        """
        Apply fixes to all translations.

        Args:
            translations: Original translations dictionary.
            dry_run: If True, don't modify original dict.

        Returns:
            Tuple of (corrected_translations, changes_dict).
            changes_dict maps DTC code -> (original, corrected, list_of_changes).
        """
        corrected = translations.copy() if dry_run else translations
        changes = {}

        for code, original_text in translations.items():
            fixed_text, change_list = self.fix_translation(original_text)

            if fixed_text != original_text:
                changes[code] = (original_text, fixed_text, change_list)
                if not dry_run:
                    corrected[code] = fixed_text
                self.stats["fixed"] += 1
            else:
                self.stats["unchanged"] += 1

        self.stats["total"] = len(translations)
        return corrected, changes

    def update_postgres(self, changes: Dict[str, Tuple[str, str, List[str]]]) -> int:
        """
        Update corrected translations in PostgreSQL.

        Args:
            changes: Dictionary of changes to apply.

        Returns:
            Number of records updated.
        """
        try:
            from sqlalchemy import create_engine, text
            from scripts.utils import get_sync_db_url

            engine = create_engine(get_sync_db_url())
            updated = 0

            with engine.connect() as conn:
                for code, (original, corrected, _) in changes.items():
                    result = conn.execute(
                        text("""
                            UPDATE dtc_codes
                            SET description_hu = :description_hu,
                                updated_at = NOW()
                            WHERE code = :code
                        """),
                        {"code": code, "description_hu": corrected}
                    )
                    if result.rowcount > 0:
                        updated += 1

                conn.commit()

            logger.info(f"Updated {updated} records in PostgreSQL")
            return updated

        except ImportError as e:
            logger.warning(f"Could not import database dependencies: {e}")
            return 0
        except Exception as e:
            logger.error(f"PostgreSQL update failed: {e}")
            return 0

    def update_neo4j(self, changes: Dict[str, Tuple[str, str, List[str]]]) -> int:
        """
        Update corrected translations in Neo4j.

        Args:
            changes: Dictionary of changes to apply.

        Returns:
            Number of nodes updated.
        """
        try:
            from neomodel import config, db
            from backend.app.core.config import settings

            # Configure Neo4j connection
            config.DATABASE_URL = settings.NEO4J_URI
            config.AUTH = (settings.NEO4J_USER, settings.NEO4J_PASSWORD)

            updated = 0

            for code, (original, corrected, _) in changes.items():
                query = """
                    MATCH (d:DTCCode {code: $code})
                    SET d.description_hu = $description_hu
                    RETURN d.code
                """
                results, _ = db.cypher_query(
                    query,
                    {"code": code, "description_hu": corrected}
                )
                if results:
                    updated += 1

            logger.info(f"Updated {updated} nodes in Neo4j")
            return updated

        except ImportError as e:
            logger.warning(f"Could not import Neo4j dependencies: {e}")
            return 0
        except Exception as e:
            logger.error(f"Neo4j update failed: {e}")
            return 0

    def print_report(
        self,
        changes: Dict[str, Tuple[str, str, List[str]]],
        verbose: bool = False
    ) -> None:
        """
        Print a report of all changes.

        Args:
            changes: Dictionary of changes made.
            verbose: If True, show all changes; otherwise show summary.
        """
        print("\n" + "=" * 80)
        print("TRANSLATION FIX REPORT")
        print("=" * 80)

        print(f"\nStatistics:")
        print(f"  Total translations: {self.stats['total']}")
        print(f"  Fixed:              {self.stats['fixed']}")
        print(f"  Unchanged:          {self.stats['unchanged']}")
        print(f"  Fix rate:           {self.stats['fixed'] / max(self.stats['total'], 1) * 100:.1f}%")

        # Count fix types
        fix_counts = defaultdict(int)
        for code, (original, corrected, change_list) in changes.items():
            for change in change_list:
                # Extract the description part (before the colon)
                fix_type = change.split(":")[0] if ":" in change else change
                fix_counts[fix_type] += 1

        print(f"\nFixes by type:")
        for fix_type, count in sorted(fix_counts.items(), key=lambda x: -x[1]):
            print(f"  {fix_type}: {count}")

        if verbose:
            print(f"\nDetailed changes:")
            for code in sorted(changes.keys()):
                original, corrected, change_list = changes[code]
                print(f"\n  {code}:")
                print(f"    Before: {original[:100]}...")
                print(f"    After:  {corrected[:100]}...")
                print(f"    Fixes:  {', '.join(change_list)}")
        else:
            # Show sample of changes
            sample_size = min(10, len(changes))
            if sample_size > 0:
                print(f"\nSample changes (showing {sample_size} of {len(changes)}):")
                for i, code in enumerate(sorted(changes.keys())[:sample_size]):
                    original, corrected, change_list = changes[code]
                    print(f"\n  {code}:")
                    print(f"    Before: {original[:80]}...")
                    print(f"    After:  {corrected[:80]}...")

        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fix Hungarian translations using glossary"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply fixes to the cache file"
    )
    parser.add_argument(
        "--database",
        action="store_true",
        help="Also update PostgreSQL and Neo4j databases"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Default to dry-run if no action specified
    if not args.apply and not args.database:
        args.dry_run = True

    fixer = TranslationFixer()

    # Load glossary
    if not fixer.load_glossary():
        sys.exit(1)

    # Build fix patterns
    fixer.build_fix_patterns()

    # Load translations
    translations = fixer.load_translations()
    if translations is None:
        sys.exit(1)

    # Apply fixes
    corrected, changes = fixer.fix_all_translations(
        translations,
        dry_run=args.dry_run
    )

    # Print report
    fixer.print_report(changes, verbose=args.verbose)

    if not changes:
        print("\nNo changes needed - translations are already correct.")
        return

    if args.dry_run:
        print("\nDry run complete. Use --apply to save changes.")
        return

    # Save to cache file
    if args.apply:
        if fixer.save_translations(corrected):
            print(f"\nSaved {fixer.stats['fixed']} corrected translations to cache.")
        else:
            print("\nFailed to save translations.")
            sys.exit(1)

    # Update databases
    if args.database:
        print("\nUpdating databases...")
        pg_count = fixer.update_postgres(changes)
        neo4j_count = fixer.update_neo4j(changes)
        print(f"  PostgreSQL: {pg_count} records updated")
        print(f"  Neo4j: {neo4j_count} nodes updated")


if __name__ == "__main__":
    main()
