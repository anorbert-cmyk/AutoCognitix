#!/usr/bin/env python3
"""
Hungarian Translation Script for DTC Codes using DeepSeek API.

This script translates English DTC descriptions to Hungarian using the
DeepSeek API (free tier available).

API Documentation: https://platform.deepseek.com/docs
Model: deepseek-chat (free tier: 1M tokens/month)

Usage:
    python scripts/translate_to_hungarian.py --translate   # Translate all pending
    python scripts/translate_to_hungarian.py --batch 100   # Translate in batches
    python scripts/translate_to_hungarian.py --verify      # Verify translations
    python scripts/translate_to_hungarian.py --stats       # Show translation stats
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm

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

# Data paths
DATA_DIR = PROJECT_ROOT / "data" / "dtc_codes"
CODES_FILE = DATA_DIR / "mytrile_codes.json"
BACKUP_FILE = DATA_DIR / "mytrile_codes_backup.json"
TRANSLATION_CACHE = DATA_DIR / "translation_cache.json"

# =============================================================================
# LLM Provider Configuration
# =============================================================================
# Supported providers (all OpenAI-compatible):
#   - deepseek: 5M free tokens, then $0.14/M (BEST VALUE)
#   - gemini: Free tier via OpenRouter (BEST FREE)
#   - groq: Free tier with rate limits (FASTEST)
#   - openrouter: 18+ free models (MOST FLEXIBLE)
#   - mistral: Free tier 1B tokens/month (EU COMPLIANT)
#   - ollama: Local, unlimited (PRIVATE)

PROVIDERS = {
    "deepseek": {
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "signup_url": "https://platform.deepseek.com/sign_up",
    },
    "gemini": {
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "nvidia/nemotron-nano-9b-v2:free",  # Free NVIDIA model
        "env_key": "OPENROUTER_API_KEY",
        "signup_url": "https://openrouter.ai",
    },
    "groq": {
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "signup_url": "https://console.groq.com",
    },
    "openrouter": {
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openrouter/auto",  # Auto-select best free model
        "env_key": "OPENROUTER_API_KEY",
        "signup_url": "https://openrouter.ai",
    },
    "mistral": {
        "api_url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-small-latest",
        "env_key": "MISTRAL_API_KEY",
        "signup_url": "https://console.mistral.ai",
    },
    "ollama": {
        "api_url": "http://localhost:11434/v1/chat/completions",
        "model": "llama3.2",  # or "jobautomation/OpenEuroLLM-Hungarian"
        "env_key": None,  # No API key needed for local
        "signup_url": "https://ollama.com",
    },
    "kimi": {
        "api_url": "https://api.moonshot.cn/v1/chat/completions",
        "model": "moonshot-v1-8k",
        "env_key": "KIMI_API_KEY",
        "signup_url": "https://platform.moonshot.ai",
    },
}

# Rate limiting
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
BATCH_SIZE = 20  # codes per API call
MAX_RETRIES = 3


def get_api_key(provider: str = "deepseek") -> Optional[str]:
    """
    Get API key from environment for the specified provider.

    Args:
        provider: Provider name (deepseek, gemini, groq, openrouter, mistral, ollama, kimi)

    Returns:
        API key string or None if not found.
    """
    if provider not in PROVIDERS:
        logger.error(f"Unknown provider: {provider}. Available: {', '.join(PROVIDERS.keys())}")
        return None

    env_key = PROVIDERS[provider]["env_key"]

    # Ollama doesn't need an API key
    if env_key is None:
        return "ollama-local"

    api_key = os.environ.get(env_key)

    if not api_key:
        signup_url = PROVIDERS[provider]["signup_url"]
        logger.error(f"No API key found for {provider}. Set {env_key} environment variable.")
        logger.info(f"Get your free API key at: {signup_url}")

    return api_key


def list_providers() -> None:
    """Print available providers and their status."""
    print("\n" + "=" * 70)
    print("AVAILABLE LLM PROVIDERS FOR HUNGARIAN TRANSLATION")
    print("=" * 70)

    for name, config in PROVIDERS.items():
        env_key = config["env_key"]
        has_key = "✅" if (env_key is None or os.environ.get(env_key)) else "❌"
        signup = config["signup_url"]
        model = config["model"]

        print(f"\n{has_key} {name.upper()}")
        print(f"   Model: {model}")
        print(f"   Signup: {signup}")
        if env_key:
            print(f"   Env var: {env_key}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS:")
    print("  1. DeepSeek - Best value (5M free tokens, then $0.14/M)")
    print("  2. Gemini via OpenRouter - Best free tier")
    print("  3. Groq - Fastest inference")
    print("  4. Ollama - Free, private, local")
    print("=" * 70)


def load_codes(file_path: Path) -> List[Dict[str, Any]]:
    """Load DTC codes from JSON file."""
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("codes", [])


def save_codes(codes: List[Dict[str, Any]], file_path: Path) -> None:
    """Save DTC codes to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "count": len(codes),
        },
        "codes": codes,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(codes)} codes to {file_path}")


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


def create_translation_prompt(descriptions: List[Tuple[str, str]]) -> str:
    """
    Create a prompt for batch translation.

    Args:
        descriptions: List of (code, description) tuples.

    Returns:
        Formatted prompt string.
    """
    prompt = """Te egy szakértő autószerelő vagy, aki angol nyelvű OBD-II hibakódokat fordít magyarra.

Fordítsd le az alábbi autódiagnosztikai hibakód-leírásokat angolról magyarra.
Használj szakmai, de érthető magyar terminológiát.
Tartsd meg a rövidítéseket (MAF, ECU, TPS stb.) de zárójelben add meg a magyar megfelelőjüket.

Válaszolj JSON formátumban, ahol a kulcs a hibakód, az érték pedig a magyar fordítás.

Hibakódok fordításra:
"""

    for code, desc in descriptions:
        prompt += f"\n{code}: {desc}"

    prompt += "\n\nJSON válasz:"

    return prompt


async def translate_batch(
    client: httpx.AsyncClient,
    descriptions: List[Tuple[str, str]],
    api_key: str,
    provider: str = "deepseek",
    retry_count: int = 0,
) -> Dict[str, str]:
    """
    Translate a batch of descriptions using any supported LLM provider.

    Args:
        client: HTTP client instance.
        descriptions: List of (code, description) tuples.
        api_key: API key for the provider.
        provider: Provider name (deepseek, gemini, groq, openrouter, mistral, ollama, kimi).
        retry_count: Current retry attempt.

    Returns:
        Dictionary mapping codes to Hungarian translations.
    """
    if provider not in PROVIDERS:
        logger.error(f"Unknown provider: {provider}")
        return {}

    config = PROVIDERS[provider]
    prompt = create_translation_prompt(descriptions)

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": "Te egy precíz fordító vagy, aki autóipari szakkifejezéseket fordít magyarra. Mindig JSON formátumban válaszolj."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    headers = {
        "Content-Type": "application/json",
    }

    # Add authorization header (except for Ollama local)
    if api_key and api_key != "ollama-local":
        headers["Authorization"] = f"Bearer {api_key}"

    # OpenRouter requires additional headers
    if provider in ["gemini", "openrouter"]:
        headers["HTTP-Referer"] = "https://github.com/AutoCognitix"
        headers["X-Title"] = "AutoCognitix DTC Translator"

    try:
        response = await client.post(
            config["api_url"],
            json=payload,
            headers=headers,
            timeout=90.0,  # Longer timeout for slower providers
        )

        if response.status_code == 429:
            wait_time = 10 * (retry_count + 1)
            logger.warning(f"Rate limited by {provider}, waiting {wait_time}s...")
            await asyncio.sleep(wait_time)
            if retry_count < MAX_RETRIES:
                return await translate_batch(
                    client, descriptions, api_key, provider, retry_count + 1
                )
            return {}

        if response.status_code != 200:
            logger.error(f"{provider} API error: {response.status_code} - {response.text[:200]}")
            return {}

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON from response
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                translations = json.loads(json_str)
                return translations
            else:
                logger.warning(f"No JSON found in {provider} response: {content[:200]}")
                return parse_translations_fallback(content, descriptions)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error from {provider}: {e}")
            return parse_translations_fallback(content, descriptions)

    except httpx.RequestError as e:
        logger.error(f"{provider} request error: {e}")
        if retry_count < MAX_RETRIES:
            await asyncio.sleep(5 * (retry_count + 1))
            return await translate_batch(
                client, descriptions, api_key, provider, retry_count + 1
            )
        return {}


# Legacy function for backward compatibility
async def translate_batch_deepseek(
    client: httpx.AsyncClient,
    descriptions: List[Tuple[str, str]],
    api_key: str,
    retry_count: int = 0,
) -> Dict[str, str]:
    """Legacy wrapper for DeepSeek translation."""
    return await translate_batch(client, descriptions, api_key, "deepseek", retry_count)


async def translate_batch_kimi(
    client: httpx.AsyncClient,
    descriptions: List[Tuple[str, str]],
    api_key: str,
    retry_count: int = 0,
) -> Dict[str, str]:
    """
    Translate a batch of descriptions using Kimi API (alternative).

    Args:
        client: HTTP client instance.
        descriptions: List of (code, description) tuples.
        api_key: Kimi API key.
        retry_count: Current retry attempt.

    Returns:
        Dictionary mapping codes to Hungarian translations.
    """
    config = PROVIDERS["kimi"]
    prompt = create_translation_prompt(descriptions)

    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": "Te egy precíz fordító vagy, aki autóipari szakkifejezéseket fordít magyarra. Mindig JSON formátumban válaszolj."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = await client.post(
            config["api_url"],
            json=payload,
            headers=headers,
            timeout=60.0,
        )

        if response.status_code != 200:
            logger.error(f"Kimi API error: {response.status_code}")
            return {}

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            return {}

        except json.JSONDecodeError:
            return parse_translations_fallback(content, descriptions)

    except httpx.RequestError as e:
        logger.error(f"Kimi request error: {e}")
        return {}


def parse_translations_fallback(
    content: str,
    descriptions: List[Tuple[str, str]],
    max_content_size: int = 100000,
) -> Dict[str, str]:
    """
    Parse translations from non-JSON response.

    Args:
        content: Response content.
        descriptions: Original descriptions.
        max_content_size: Maximum content size to process (ReDoS prevention).

    Returns:
        Dictionary mapping codes to translations.
    """
    import re

    translations = {}

    # Prevent ReDoS by limiting content size
    if not content or len(content) > max_content_size:
        logger.warning(f"Content too large for fallback parsing: {len(content) if content else 0}")
        return translations

    for code, _ in descriptions:
        # Use more specific pattern to avoid ReDoS
        # Match code followed by separator and text until newline or next code
        pattern = rf'{re.escape(code)}\s*[:\-]\s*([^\n]+)'
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            translation = match.group(1).strip().strip('"').strip("'")
            # Validate translation quality
            if translation and len(translation) > 4 and len(translation) < 500:
                translations[code] = translation

    return translations


async def translate_all(
    codes: List[Dict[str, Any]],
    provider: str = "deepseek",
    batch_size: int = BATCH_SIZE,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Translate all pending DTC codes using any supported LLM provider.

    Args:
        codes: List of DTC code dictionaries.
        provider: API provider (deepseek, gemini, groq, openrouter, mistral, ollama, kimi).
        batch_size: Number of codes per API call.
        limit: Maximum codes to translate (None = all).

    Returns:
        Tuple of (translated_count, failed_count).
    """
    api_key = get_api_key(provider)

    if not api_key:
        return 0, 0

    # Load translation cache
    cache = load_translation_cache()

    # Filter codes needing translation
    pending = []
    for code in codes:
        code_str = code.get("code", "")
        desc_en = code.get("description_en", "")

        # Skip if already translated or cached
        if code.get("description_hu") or code_str in cache:
            continue

        if desc_en:
            pending.append((code_str, desc_en))

    if limit:
        pending = pending[:limit]

    logger.info(f"Translating {len(pending)} codes using {provider} ({PROVIDERS[provider]['model']})...")

    if not pending:
        logger.info("No codes to translate")
        return 0, 0

    translated_count = 0
    failed_count = 0

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i in tqdm(range(0, len(pending), batch_size), desc=f"Translating ({provider})"):
            batch = pending[i:i + batch_size]

            # Use unified translate_batch function for all providers
            translations = await translate_batch(client, batch, api_key, provider)

            for code_str, _ in batch:
                if code_str in translations:
                    cache[code_str] = translations[code_str]
                    translated_count += 1
                else:
                    failed_count += 1

            # Save cache every 5 batches (more frequent saves)
            if (i // batch_size + 1) % 5 == 0:
                save_translation_cache(cache)

            await asyncio.sleep(RATE_LIMIT_DELAY)

    # Save final cache
    save_translation_cache(cache)

    logger.info(f"Translated: {translated_count}, Failed: {failed_count}")
    return translated_count, failed_count


def apply_translations(codes: List[Dict[str, Any]]) -> int:
    """
    Apply cached translations to codes.

    Args:
        codes: List of DTC code dictionaries.

    Returns:
        Number of codes updated.
    """
    cache = load_translation_cache()

    if not cache:
        logger.info("No translations in cache")
        return 0

    updated = 0

    for code in codes:
        code_str = code.get("code", "")

        if code_str in cache and not code.get("description_hu"):
            code["description_hu"] = cache[code_str]
            code["translation_status"] = "completed"
            updated += 1

    logger.info(f"Applied {updated} translations from cache")
    return updated


def update_postgres_translations() -> int:
    """Update PostgreSQL with cached translations."""
    from backend.app.db.postgres.models import DTCCode
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    cache = load_translation_cache()

    if not cache:
        return 0

    from backend.app.core.config import settings
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    engine = create_engine(db_url)
    updated = 0

    with Session(engine) as session:
        for code_str, translation in tqdm(cache.items(), desc="Updating PostgreSQL"):
            dtc = session.query(DTCCode).filter_by(code=code_str).first()

            if dtc and not dtc.description_hu:
                dtc.description_hu = translation
                updated += 1

        session.commit()

    logger.info(f"Updated {updated} PostgreSQL records")
    return updated


def update_neo4j_translations() -> int:
    """Update Neo4j with cached translations."""
    from backend.app.db.neo4j_models import DTCNode

    cache = load_translation_cache()

    if not cache:
        return 0

    updated = 0

    for code_str, translation in tqdm(cache.items(), desc="Updating Neo4j"):
        node = DTCNode.nodes.get_or_none(code=code_str)

        if node and not node.description_hu:
            node.description_hu = translation
            node.save()
            updated += 1

    logger.info(f"Updated {updated} Neo4j nodes")
    return updated


def get_stats(codes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get translation statistics."""
    total = len(codes)
    translated = sum(1 for c in codes if c.get("description_hu"))
    pending = total - translated

    cache = load_translation_cache()

    return {
        "total_codes": total,
        "translated": translated,
        "pending": pending,
        "percentage": f"{(translated / total * 100):.1f}%" if total > 0 else "0%",
        "cache_size": len(cache),
    }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Translate DTC descriptions to Hungarian using various free LLM APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python translate_to_hungarian.py --list-providers        # Show available providers
  python translate_to_hungarian.py --translate --provider gemini --limit 100
  python translate_to_hungarian.py --translate --provider deepseek
  python translate_to_hungarian.py --translate --provider ollama  # Local, free
  python translate_to_hungarian.py --apply --update-db     # Apply translations to DBs

Environment Variables:
  DEEPSEEK_API_KEY    - DeepSeek API key (https://platform.deepseek.com)
  OPENROUTER_API_KEY  - OpenRouter API key for Gemini (https://openrouter.ai)
  GROQ_API_KEY        - Groq API key (https://console.groq.com)
  MISTRAL_API_KEY     - Mistral API key (https://console.mistral.ai)
        """
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate pending codes",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply cached translations to codes file",
    )
    parser.add_argument(
        "--update-db",
        action="store_true",
        help="Update databases with cached translations",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show translation statistics",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available LLM providers and their status",
    )
    parser.add_argument(
        "--provider",
        choices=list(PROVIDERS.keys()),
        default="deepseek",
        help="LLM provider (default: deepseek). Use --list-providers to see all options",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for API calls (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum codes to translate",
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

    # Handle --list-providers separately
    if args.list_providers:
        list_providers()
        return

    # Default to --stats if nothing specified
    if not (args.translate or args.apply or args.update_db or args.stats):
        args.stats = True

    try:
        # Load codes
        codes = load_codes(CODES_FILE)

        if not codes:
            # Try alternative files
            alt_files = [
                DATA_DIR / "generic_codes.json",
                DATA_DIR / "klavkarr_codes.json",
            ]
            for alt_file in alt_files:
                codes = load_codes(alt_file)
                if codes:
                    logger.info(f"Loaded codes from {alt_file}")
                    break

        if not codes:
            logger.error("No codes found. Run import scripts first.")
            sys.exit(1)

        # Show stats
        if args.stats:
            stats = get_stats(codes)

            print("\n" + "=" * 60)
            print("TRANSLATION STATISTICS")
            print("=" * 60)
            print(f"Total codes: {stats['total_codes']}")
            print(f"Translated: {stats['translated']} ({stats['percentage']})")
            print(f"Pending: {stats['pending']}")
            print(f"Cache size: {stats['cache_size']}")
            print("=" * 60)

        # Translate
        if args.translate:
            translated, failed = await translate_all(
                codes,
                provider=args.provider,
                batch_size=args.batch,
                limit=args.limit,
            )

            print(f"\nTranslation complete: {translated} translated, {failed} failed")

        # Apply translations
        if args.apply:
            # Backup first
            save_codes(codes, BACKUP_FILE)

            applied = apply_translations(codes)
            save_codes(codes, CODES_FILE)

            print(f"\nApplied {applied} translations")

        # Update databases
        if args.update_db:
            pg_updated = update_postgres_translations()
            neo_updated = update_neo4j_translations()

            print(f"\nDatabase updates: PostgreSQL={pg_updated}, Neo4j={neo_updated}")

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
