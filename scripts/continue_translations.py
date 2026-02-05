#!/usr/bin/env python3
"""
Continue Hungarian Translations for DTC Codes.

This script continues the translation process for untranslated DTC codes
in PostgreSQL, then updates Neo4j and Qdrant.

Features:
- Queries PostgreSQL for untranslated codes directly
- Uses multiple LLM providers (DeepSeek, Groq, OpenRouter, Kimi)
- Parallel API calls for efficiency
- Progress checkpointing
- Quality validation of translations
- Updates all three databases (PostgreSQL, Neo4j, Qdrant)

Usage:
    python scripts/continue_translations.py --stats              # Show current stats
    python scripts/continue_translations.py --translate          # Continue translations
    python scripts/continue_translations.py --translate --limit 500  # Translate 500 codes
    python scripts/continue_translations.py --sync-neo4j         # Sync to Neo4j
    python scripts/continue_translations.py --reindex-qdrant     # Reindex Qdrant
    python scripts/continue_translations.py --all                # Do everything
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
TRANSLATION_CACHE = DATA_DIR / "translation_cache.json"
CHECKPOINT_FILE = DATA_DIR / "translation_checkpoint.json"

# =============================================================================
# LLM Provider Configuration
# =============================================================================
PROVIDERS = {
    "anthropic": {
        "api_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-3-5-haiku-20241022",
        "env_key": "ANTHROPIC_API_KEY",
        "rate_limit_delay": 0.3,
        "batch_size": 40,
        "is_anthropic": True,
    },
    "deepseek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "rate_limit_delay": 0.5,
        "batch_size": 30,
    },
    "groq": {
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "rate_limit_delay": 1.0,
        "batch_size": 20,
    },
    "openrouter": {
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "google/gemini-2.0-flash-001",
        "env_key": "OPENROUTER_API_KEY",
        "rate_limit_delay": 0.5,
        "batch_size": 30,
    },
    "kimi": {
        "api_url": "https://api.moonshot.cn/v1/chat/completions",
        "model": "moonshot-v1-8k",
        "env_key": "KIMI_API_KEY",
        "rate_limit_delay": 1.0,
        "batch_size": 20,
    },
    "mistral": {
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
        "rate_limit_delay": 0.5,
        "batch_size": 25,
    },
}

# Hungarian character validation pattern
HUNGARIAN_CHARS = re.compile(r'[a-zA-Z0-9\s\-\.\,\(\)\/\#\@\:\;\!\?\%\&\*\+\=\'\"]+')
HUNGARIAN_SPECIAL = set('aeiouAEIOUbcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')
HUNGARIAN_ACCENTED = set('aeiouAEIOUbcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ')

MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 100


def get_sync_db_url() -> str:
    """Get synchronous database URL."""
    from backend.app.core.config import settings
    url = settings.DATABASE_URL
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("postgresql+asyncpg://", "postgresql://")
    return url


def get_available_provider() -> Optional[Tuple[str, str]]:
    """
    Find an available LLM provider with a valid API key.

    Returns:
        Tuple of (provider_name, api_key) or None if no provider available.
    """
    # Priority order - anthropic first since it's already configured
    priority = ["anthropic", "groq", "deepseek", "openrouter", "mistral", "kimi"]

    for provider in priority:
        env_key = PROVIDERS[provider]["env_key"]
        api_key = os.environ.get(env_key)
        if api_key and api_key != "your_anthropic_api_key_here":
            logger.info(f"Using provider: {provider}")
            return provider, api_key

    return None


def load_translation_cache() -> Dict[str, str]:
    """Load translation cache from file."""
    if not TRANSLATION_CACHE.exists():
        return {}

    with open(TRANSLATION_CACHE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_translation_cache(cache: Dict[str, str]) -> None:
    """Save translation cache to file."""
    TRANSLATION_CACHE.parent.mkdir(parents=True, exist_ok=True)

    with open(TRANSLATION_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_checkpoint() -> Dict[str, Any]:
    """Load translation checkpoint."""
    if not CHECKPOINT_FILE.exists():
        return {"last_code": None, "translated_count": 0, "timestamp": None}

    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(last_code: str, translated_count: int) -> None:
    """Save translation checkpoint."""
    checkpoint = {
        "last_code": last_code,
        "translated_count": translated_count,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def validate_translation(text: str, original: str) -> Tuple[bool, str]:
    """
    Validate a Hungarian translation for quality.

    Args:
        text: The translated Hungarian text.
        original: The original English text.

    Returns:
        Tuple of (is_valid, reason).
    """
    if not text:
        return False, "Empty translation"

    # Check minimum length (should be at least 10% of original)
    if len(text) < max(10, len(original) * 0.1):
        return False, "Translation too short"

    # Check maximum length (should not be more than 3x original)
    if len(text) > len(original) * 3:
        return False, "Translation too long"

    # Check for common bad translations
    bad_patterns = [
        r'\berdo\b',  # "erdő" (forest) instead of "föld" (ground)
        r'\bhuto\b.*akku',  # "hűtő" instead of "akkumulátor"
        r'[^\x00-\x7F\u00C0-\u017F\u0150\u0151\u0170\u0171]',  # Non-Latin/Hungarian chars
    ]

    for pattern in bad_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, f"Contains suspicious pattern: {pattern}"

    # Check for Hungarian accented characters (should have some)
    hungarian_accents = set('áéíóöőúüűÁÉÍÓÖŐÚÜŰ')
    has_accents = any(c in hungarian_accents for c in text)

    # Most Hungarian automotive texts should have at least some accents
    if len(text) > 30 and not has_accents:
        # This is a warning, not a failure
        logger.debug(f"Translation might be missing accents: {text[:50]}...")

    return True, "OK"


def create_translation_prompt(descriptions: List[Tuple[str, str]]) -> str:
    """Create a prompt for batch translation."""
    prompt = """Te egy szakerto autoszerelo vagy, aki angol nyelvu OBD-II hibakodokat fordit magyarra.

FONTOS SZABALYOK:
1. Hasznalj pontos magyar autoipari terminologiat
2. A "ground" szo MINDIG "test" vagy "fold" legyen, SOHA NEM "erdo"
3. A "battery" szo MINDIG "akkumulator" legyen, SOHA NEM "huto"
4. Az "open circuit" MINDIG "aramkor szakadas" vagy "szakadt aramkor"
5. A "short to ground" MINDIG "testre zaras" vagy "rovidtar a testre"
6. A "short to battery" MINDIG "akkumulatorra zaras" vagy "rovidtar az akkumulatorra"
7. Tartsd meg a roviditeseket (MAF, ECU, TPS, PCM stb.)
8. A "circuit" szo MINDIG "aramkor" legyen

Forditsd le az alabbi hibakod-leirasokat. Valaszolj JSON formatumban:
{"KOD1": "magyar forditas1", "KOD2": "magyar forditas2", ...}

Hibakodok:
"""

    for code, desc in descriptions:
        prompt += f"\n{code}: {desc}"

    prompt += "\n\nJSON valasz (CSAK a JSON, semmi mas):"

    return prompt


async def translate_batch(
    client: httpx.AsyncClient,
    descriptions: List[Tuple[str, str]],
    api_key: str,
    provider: str,
    retry_count: int = 0,
) -> Dict[str, str]:
    """
    Translate a batch of descriptions using an LLM provider.

    Args:
        client: HTTP client instance.
        descriptions: List of (code, description) tuples.
        api_key: API key for the provider.
        provider: Provider name.
        retry_count: Current retry attempt.

    Returns:
        Dictionary mapping codes to Hungarian translations.
    """
    config = PROVIDERS[provider]
    prompt = create_translation_prompt(descriptions)
    is_anthropic = config.get("is_anthropic", False)

    if is_anthropic:
        # Anthropic Messages API format
        payload = {
            "model": config["model"],
            "max_tokens": 4000,
            "system": "Te egy preciz fordito vagy, aki autoipari szakkifejezeseket fordit magyarra. Mindig JSON formatumban valaszolj, semmi mast ne irj.",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        # OpenAI-compatible format
        payload = {
            "model": config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": "Te egy preciz fordito vagy, aki autoipari szakkifejezeseket fordit magyarra. Mindig JSON formatumban valaszolj, semmi mast ne irj."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.2,
            "max_tokens": 4000,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    # OpenRouter requires additional headers
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/AutoCognitix"
        headers["X-Title"] = "AutoCognitix DTC Translator"

    try:
        response = await client.post(
            config["api_url"],
            json=payload,
            headers=headers,
            timeout=120.0,
        )

        if response.status_code == 429:
            wait_time = 15 * (retry_count + 1)
            logger.warning(f"Rate limited by {provider}, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            if retry_count < MAX_RETRIES:
                return await translate_batch(
                    client, descriptions, api_key, provider, retry_count + 1
                )
            return {}

        if response.status_code != 200:
            logger.error(f"{provider} API error: {response.status_code} - {response.text[:200]}")
            if retry_count < MAX_RETRIES:
                await asyncio.sleep(5)
                return await translate_batch(
                    client, descriptions, api_key, provider, retry_count + 1
                )
            return {}

        result = response.json()

        # Extract content based on provider
        if is_anthropic:
            content = result.get("content", [{}])[0].get("text", "")
        else:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON from response
        try:
            # Find JSON in response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                translations = json.loads(json_str)

                # Validate translations
                validated = {}
                for code, translation in translations.items():
                    original = next((d[1] for d in descriptions if d[0] == code), "")
                    is_valid, reason = validate_translation(translation, original)
                    if is_valid:
                        validated[code] = translation
                    else:
                        logger.debug(f"Invalid translation for {code}: {reason}")

                return validated
            else:
                logger.warning(f"No JSON found in {provider} response")
                return {}

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error from {provider}: {e}")
            return {}

    except httpx.RequestError as e:
        logger.error(f"{provider} request error: {e}")
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(5 * (retry_count + 1))
            return await translate_batch(
                client, descriptions, api_key, provider, retry_count + 1
            )
        return {}


def get_untranslated_codes(limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """
    Get untranslated DTC codes from PostgreSQL.

    Args:
        limit: Maximum number of codes to fetch.

    Returns:
        List of (code, description_en) tuples.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(get_sync_db_url())

    query = """
        SELECT code, description_en
        FROM dtc_codes
        WHERE (description_hu IS NULL OR description_hu = '')
        AND description_en IS NOT NULL
        AND description_en != ''
        ORDER BY code
    """

    if limit:
        query += f" LIMIT {limit}"

    with engine.connect() as conn:
        result = conn.execute(text(query))
        codes = [(row[0], row[1]) for row in result]

    return codes


def get_translation_stats() -> Dict[str, Any]:
    """Get current translation statistics from PostgreSQL."""
    from sqlalchemy import create_engine, text

    engine = create_engine(get_sync_db_url())

    with engine.connect() as conn:
        # Total codes
        total = conn.execute(text("SELECT COUNT(*) FROM dtc_codes")).scalar()

        # Translated codes
        translated = conn.execute(text(
            "SELECT COUNT(*) FROM dtc_codes WHERE description_hu IS NOT NULL AND description_hu != ''"
        )).scalar()

        # Untranslated codes
        untranslated = total - translated

        # By category
        by_category = conn.execute(text("""
            SELECT category,
                   COUNT(*) as total,
                   COUNT(CASE WHEN description_hu IS NOT NULL AND description_hu != '' THEN 1 END) as translated
            FROM dtc_codes
            GROUP BY category
            ORDER BY category
        """)).fetchall()

    # Cache size
    cache = load_translation_cache()

    return {
        "total": total,
        "translated": translated,
        "untranslated": untranslated,
        "percentage": f"{(translated / total * 100):.1f}%" if total > 0 else "0%",
        "cache_size": len(cache),
        "by_category": [{"category": r[0], "total": r[1], "translated": r[2]} for r in by_category],
    }


def update_postgres_translations(translations: Dict[str, str]) -> int:
    """
    Update PostgreSQL with new translations.

    Args:
        translations: Dictionary mapping codes to Hungarian translations.

    Returns:
        Number of records updated.
    """
    if not translations:
        return 0

    from sqlalchemy import create_engine, text

    engine = create_engine(get_sync_db_url())
    updated = 0

    with engine.connect() as conn:
        for code, translation in translations.items():
            result = conn.execute(
                text("""
                    UPDATE dtc_codes
                    SET description_hu = :description_hu,
                        updated_at = NOW()
                    WHERE code = :code
                    AND (description_hu IS NULL OR description_hu = '')
                """),
                {"code": code, "description_hu": translation}
            )
            if result.rowcount > 0:
                updated += 1

        conn.commit()

    return updated


def sync_neo4j_translations() -> int:
    """
    Sync translations from PostgreSQL to Neo4j.

    Returns:
        Number of nodes updated.
    """
    from sqlalchemy import create_engine, text
    from neomodel import config as neo_config, db
    from backend.app.core.config import settings

    # Configure Neo4j
    neo_uri = settings.NEO4J_URI
    if neo_uri.startswith("bolt://"):
        neo_uri = neo_uri.replace("bolt://", "")
    neo_config.DATABASE_URL = f"bolt://{settings.NEO4J_USER}:{settings.NEO4J_PASSWORD}@{neo_uri}"

    # Get translations from PostgreSQL
    engine = create_engine(get_sync_db_url())

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT code, description_hu
            FROM dtc_codes
            WHERE description_hu IS NOT NULL AND description_hu != ''
        """))
        translations = {row[0]: row[1] for row in result}

    logger.info(f"Syncing {len(translations)} translations to Neo4j...")

    updated = 0
    batch_size = 100
    codes = list(translations.keys())

    for i in tqdm(range(0, len(codes), batch_size), desc="Syncing to Neo4j"):
        batch_codes = codes[i:i + batch_size]

        for code in batch_codes:
            try:
                query = """
                    MATCH (d:DTCNode {code: $code})
                    WHERE d.description_hu IS NULL OR d.description_hu = ''
                    SET d.description_hu = $description_hu
                    RETURN d.code
                """
                results, _ = db.cypher_query(
                    query,
                    {"code": code, "description_hu": translations[code]}
                )
                if results:
                    updated += 1
            except Exception as e:
                logger.debug(f"Neo4j update error for {code}: {e}")

    logger.info(f"Updated {updated} Neo4j nodes")
    return updated


def reindex_qdrant() -> int:
    """
    Re-index Qdrant with updated Hungarian translations.

    Returns:
        Number of vectors indexed.
    """
    from sqlalchemy import create_engine, text
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    import uuid

    # Import embedder
    from scripts.index_qdrant import embed_batch, EMBEDDING_DIMENSION

    logger.info("Re-indexing Qdrant with Hungarian translations...")

    # Get all translated codes from PostgreSQL
    engine = create_engine(get_sync_db_url())

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT code, description_en, description_hu, category, severity, system,
                   symptoms, possible_causes, diagnostic_steps, related_codes, is_generic
            FROM dtc_codes
            WHERE description_hu IS NOT NULL AND description_hu != ''
        """))
        codes = result.fetchall()

    if not codes:
        logger.warning("No translated codes found for indexing")
        return 0

    logger.info(f"Found {len(codes)} translated codes for indexing")

    # Connect to Qdrant
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )

    collection_name = "dtc_embeddings_hu"

    # Recreate collection
    try:
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=qdrant_models.Distance.COSINE,
        ),
    )

    # Prepare data
    ids = []
    texts = []
    payloads = []

    for row in codes:
        code = row[0]
        ids.append(f"dtc_{code}")
        texts.append(row[2])  # description_hu
        payloads.append({
            "code": code,
            "description_hu": row[2],
            "description_en": row[1],
            "category": row[3],
            "severity": row[4],
            "system": row[5] or "",
            "symptoms": row[6] or [],
            "possible_causes": row[7] or [],
            "diagnostic_steps": row[8] or [],
            "related_codes": row[9] or [],
            "is_generic": row[10],
        })

    # Generate embeddings in batches
    batch_size = 10
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embed_batch(batch_texts, preprocess=False)
        all_embeddings.extend(batch_embeddings)

    # Upsert to Qdrant
    logger.info(f"Upserting {len(ids)} vectors to Qdrant...")

    for i in tqdm(range(0, len(ids), batch_size), desc="Indexing to Qdrant"):
        batch_ids = ids[i:i + batch_size]
        batch_vectors = all_embeddings[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]

        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, id_)),
                vector=vector,
                payload=payload,
            )
            for id_, vector, payload in zip(batch_ids, batch_vectors, batch_payloads)
        ]

        client.upsert(
            collection_name=collection_name,
            points=points,
        )

    logger.info(f"Indexed {len(ids)} vectors to Qdrant")
    return len(ids)


async def translate_all(
    provider: str,
    api_key: str,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Translate all pending codes.

    Args:
        provider: LLM provider name.
        api_key: API key.
        limit: Maximum codes to translate.

    Returns:
        Tuple of (translated_count, failed_count).
    """
    config = PROVIDERS[provider]
    batch_size = config["batch_size"]
    rate_limit_delay = config["rate_limit_delay"]

    # Load cache
    cache = load_translation_cache()

    # Get untranslated codes
    pending = get_untranslated_codes(limit)

    # Filter out cached codes
    pending = [(code, desc) for code, desc in pending if code not in cache]

    logger.info(f"Found {len(pending)} codes to translate using {provider}")

    if not pending:
        logger.info("No codes to translate")
        return 0, 0

    translated_count = 0
    failed_count = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i in tqdm(range(0, len(pending), batch_size), desc=f"Translating ({provider})"):
            batch = pending[i:i + batch_size]

            translations = await translate_batch(client, batch, api_key, provider)

            # Update cache and counts
            for code, desc in batch:
                if code in translations:
                    cache[code] = translations[code]
                    translated_count += 1
                else:
                    failed_count += 1

            # Update PostgreSQL with this batch
            if translations:
                pg_updated = update_postgres_translations(translations)
                logger.debug(f"Updated {pg_updated} PostgreSQL records")

            # Save checkpoint
            if (i // batch_size + 1) % 5 == 0:
                save_translation_cache(cache)
                save_checkpoint(batch[-1][0], translated_count)

            # Report progress every 500 codes
            if translated_count > 0 and translated_count % 500 == 0:
                logger.info(f"Progress: {translated_count} translated, {failed_count} failed")

            await asyncio.sleep(rate_limit_delay)

    # Final save
    save_translation_cache(cache)
    save_checkpoint(pending[-1][0] if pending else None, translated_count)

    logger.info(f"Translation complete: {translated_count} translated, {failed_count} failed")
    return translated_count, failed_count


def print_stats() -> None:
    """Print current translation statistics."""
    stats = get_translation_stats()

    print("\n" + "=" * 70)
    print("HUNGARIAN TRANSLATION STATISTICS")
    print("=" * 70)
    print(f"\nPostgreSQL Database:")
    print(f"  Total DTC codes:    {stats['total']:,}")
    print(f"  Translated:         {stats['translated']:,} ({stats['percentage']})")
    print(f"  Untranslated:       {stats['untranslated']:,}")
    print(f"  Cache size:         {stats['cache_size']:,}")

    print(f"\nBy Category:")
    for cat in stats['by_category']:
        pct = (cat['translated'] / cat['total'] * 100) if cat['total'] > 0 else 0
        print(f"  {cat['category']:<12} {cat['translated']:>4}/{cat['total']:<4} ({pct:.1f}%)")

    # Target progress
    target = 0.80  # 80%
    current = stats['translated'] / stats['total'] if stats['total'] > 0 else 0
    needed = int(stats['total'] * target) - stats['translated']

    print(f"\nTarget: 80% ({int(stats['total'] * target):,} codes)")
    if needed > 0:
        print(f"Need to translate: {needed:,} more codes")
    else:
        print("TARGET ACHIEVED!")

    print("=" * 70)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Continue Hungarian translations for DTC codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current translation statistics",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Continue translating pending codes",
    )
    parser.add_argument(
        "--sync-neo4j",
        action="store_true",
        help="Sync translations to Neo4j",
    )
    parser.add_argument(
        "--reindex-qdrant",
        action="store_true",
        help="Re-index Qdrant with translations",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Do everything: translate, sync Neo4j, reindex Qdrant",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum codes to translate",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default=None,
        help="Force specific LLM provider",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to stats if nothing specified
    if not any([args.stats, args.translate, args.sync_neo4j, args.reindex_qdrant, args.all]):
        args.stats = True

    # Show stats
    if args.stats or args.all:
        print_stats()

    # Translate
    if args.translate or args.all:
        # Find available provider
        if args.provider:
            api_key = os.environ.get(PROVIDERS[args.provider]["env_key"])
            if not api_key:
                logger.error(f"No API key found for {args.provider}")
                sys.exit(1)
            provider = args.provider
        else:
            result = get_available_provider()
            if not result:
                logger.error("No LLM provider available. Please set one of these environment variables:")
                for name, config in PROVIDERS.items():
                    print(f"  {config['env_key']}")
                sys.exit(1)
            provider, api_key = result

        translated, failed = await translate_all(provider, api_key, args.limit)
        print(f"\nTranslation complete: {translated} translated, {failed} failed")

        # Show updated stats
        print_stats()

    # Sync Neo4j
    if args.sync_neo4j or args.all:
        try:
            updated = sync_neo4j_translations()
            print(f"\nNeo4j sync complete: {updated} nodes updated")
        except Exception as e:
            logger.error(f"Neo4j sync failed: {e}")
            if not args.all:
                raise

    # Reindex Qdrant
    if args.reindex_qdrant or args.all:
        try:
            indexed = reindex_qdrant()
            print(f"\nQdrant reindex complete: {indexed} vectors indexed")
        except Exception as e:
            logger.error(f"Qdrant reindex failed: {e}")
            if not args.all:
                raise

    # Final summary
    if args.all:
        print("\n" + "=" * 70)
        print("ALL OPERATIONS COMPLETE")
        print("=" * 70)
        print_stats()


if __name__ == "__main__":
    asyncio.run(main())
