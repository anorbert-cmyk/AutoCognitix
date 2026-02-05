#!/usr/bin/env python3
"""
Export OpenAPI schema from FastAPI application.

This script generates the OpenAPI JSON specification from the AutoCognitix
backend API and saves it to the docs directory.

Usage:
    python scripts/export_openapi.py [--output PATH] [--format FORMAT]

Options:
    --output, -o    Output file path (default: docs/openapi.json)
    --format, -f    Output format: json or yaml (default: json)
    --pretty, -p    Pretty print JSON output (default: True)

Examples:
    # Export to default location
    python scripts/export_openapi.py

    # Export to custom location
    python scripts/export_openapi.py -o /path/to/openapi.json

    # Export as YAML
    python scripts/export_openapi.py -f yaml -o docs/openapi.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add backend to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
BACKEND_DIR = SCRIPT_DIR.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


def export_openapi(output_path: str, format: str = "json", pretty: bool = True) -> None:
    """
    Export the OpenAPI schema from the FastAPI application.

    Args:
        output_path: Path to save the OpenAPI specification
        format: Output format ('json' or 'yaml')
        pretty: Whether to format the output for readability
    """
    # Set environment variables to prevent database connections during import
    os.environ.setdefault("DATABASE_URL", "postgresql://dummy:dummy@localhost/dummy")
    os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    os.environ.setdefault("JWT_SECRET_KEY", "dummy-secret-key-for-openapi-export")
    os.environ.setdefault("ANTHROPIC_API_KEY", "dummy-key")

    # Import FastAPI app
    try:
        from app.main import app
    except ImportError as e:
        print(f"Error: Could not import FastAPI app: {e}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)

    # Get OpenAPI schema
    openapi_schema = app.openapi()

    # Add additional metadata
    openapi_schema["info"]["x-logo"] = {
        "url": "https://autocognitix.hu/logo.png",
        "altText": "AutoCognitix Logo"
    }

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    if format.lower() == "yaml":
        try:
            import yaml
            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    openapi_schema,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except ImportError:
            print("Error: PyYAML is required for YAML output. Install with: pip install pyyaml")
            sys.exit(1)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
            else:
                json.dump(openapi_schema, f, ensure_ascii=False)

    print(f"OpenAPI schema exported to: {output_path}")

    # Print summary
    paths_count = len(openapi_schema.get("paths", {}))
    schemas_count = len(openapi_schema.get("components", {}).get("schemas", {}))
    print(f"  - Paths: {paths_count}")
    print(f"  - Schemas: {schemas_count}")
    print(f"  - Version: {openapi_schema.get('info', {}).get('version', 'unknown')}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export OpenAPI schema from AutoCognitix API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-o", "--output",
        default=str(SCRIPT_DIR.parent / "docs" / "openapi.json"),
        help="Output file path (default: docs/openapi.json)",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "-p", "--pretty",
        action="store_true",
        default=True,
        help="Pretty print output (default: True)",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Disable pretty printing (compact output)",
    )

    args = parser.parse_args()

    pretty = args.pretty and not args.no_pretty

    export_openapi(args.output, args.format, pretty)


if __name__ == "__main__":
    main()
