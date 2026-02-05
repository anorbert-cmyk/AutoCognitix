# AutoCognitix CLI Documentation

A comprehensive command-line interface for the AutoCognitix vehicle diagnostic platform.

## Installation

### Quick Install (Development)

```bash
# Navigate to the project root
cd /path/to/AutoCognitix

# Install in development mode
pip install -e scripts/cli/
```

### Install Dependencies Only

```bash
pip install click rich httpx pyyaml
```

### Run Without Installation

```bash
# Run directly with Python
python scripts/cli/autocognitix_cli.py --help
```

## Quick Start

```bash
# Check available commands
autocognitix --help

# Search for DTC codes
autocognitix dtc search "oxygen sensor"

# Get detailed info about a specific code
autocognitix dtc info P0420

# Run diagnosis with codes and symptoms
autocognitix diagnose P0171 P0174 --symptoms "Motor nehezen indul hidegben"

# Decode a VIN
autocognitix vehicle decode 1HGCM82633A123456

# View database statistics
autocognitix stats
```

## Commands Reference

### Global Options

All commands support these global options:

| Option | Short | Description |
|--------|-------|-------------|
| `--verbose` | `-v` | Increase verbosity (-v for info, -vv for debug) |
| `--json` | | Output results in JSON format |
| `--config` | | Path to custom config file |
| `--version` | | Show version and exit |
| `--help` | | Show help message |

### diagnose

Run vehicle diagnosis based on DTC codes and symptoms.

```bash
# Basic diagnosis with DTC codes
autocognitix diagnose P0171 P0174

# With symptom description (Hungarian)
autocognitix diagnose P0171 P0174 --symptoms "Motor nehezen indul hidegben"

# With vehicle information
autocognitix diagnose P0420 \
    --symptoms "Fogyasztas novekedett" \
    --make Volkswagen \
    --model Golf \
    --year 2018

# Use backend API instead of local data
autocognitix diagnose P0171 --use-api

# JSON output for scripting
autocognitix diagnose P0171 --json
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--symptoms` | `-s` | Symptom description (Hungarian recommended) |
| `--make` | `-m` | Vehicle make (e.g., Volkswagen) |
| `--model` | | Vehicle model (e.g., Golf) |
| `--year` | `-y` | Vehicle year |
| `--use-api` | | Use backend API instead of local data |

### dtc

DTC code lookup and search commands.

#### dtc search

Search DTC codes by description or code prefix.

```bash
# Search by description
autocognitix dtc search "oxygen sensor"

# Search by code prefix
autocognitix dtc search P01

# Filter by category
autocognitix dtc search "sensor" --category powertrain

# Limit results
autocognitix dtc search "motor" --limit 10

# JSON output
autocognitix dtc search "fuel" --json
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--category` | `-c` | Filter by category (powertrain, body, chassis, network) |
| `--limit` | `-l` | Maximum results (default: 20) |
| `--use-api` | | Use backend API |

#### dtc info

Get detailed information about a specific DTC code.

```bash
# Basic info
autocognitix dtc info P0420

# With verbose output
autocognitix dtc info P0171 -v

# JSON output
autocognitix dtc info P0300 --json
```

**Output includes:**
- English and Hungarian descriptions
- Category and system
- Severity level
- Common symptoms
- Possible causes
- Diagnostic steps
- Related codes

#### dtc related

Find DTC codes related to a specific code.

```bash
# Find related codes
autocognitix dtc related P0171

# Limit results
autocognitix dtc related P0300 --limit 5

# JSON output
autocognitix dtc related P0420 --json
```

### vehicle

Vehicle-related commands.

#### vehicle decode

Decode a VIN (Vehicle Identification Number) using the NHTSA database.

```bash
# Decode VIN
autocognitix vehicle decode 1HGCM82633A123456

# European VIN
autocognitix vehicle decode WVWZZZ3CZWE123456

# JSON output
autocognitix vehicle decode 1HGCM82633A123456 --json
```

**Output includes:**
- Make, Model, Year
- Body class and vehicle type
- Manufacturing country
- Engine specifications
- Transmission and drive type

### translate

Translate automotive terms to Hungarian.

```bash
# Translate a term
autocognitix translate "Mass Air Flow Sensor"

# Translate DTC description
autocognitix translate P0171

# JSON output
autocognitix translate "Catalytic Converter" --json
```

### stats

Show database statistics.

```bash
# Show all statistics
autocognitix stats

# JSON output for scripting
autocognitix stats --json
```

**Output includes:**
- Total DTC codes
- Category breakdown
- Severity distribution
- Translation coverage
- Data sources

### config

Manage CLI configuration.

```bash
# Show current configuration
autocognitix config --show

# Set API URL
autocognitix config --set api_url http://localhost:8000/api/v1

# Set default language
autocognitix config --set language en

# Enable API mode by default
autocognitix config --set use_api true

# Reset to defaults
autocognitix config --reset
```

**Configuration options:**

| Key | Default | Description |
|-----|---------|-------------|
| `api_url` | `http://localhost:8000/api/v1` | Backend API URL |
| `language` | `hu` | Default language (hu or en) |
| `default_limit` | `20` | Default search result limit |
| `verbose` | `0` | Default verbosity level |
| `use_api` | `false` | Use API by default |

Configuration is stored in `~/.autocognitix.yaml`.

## Examples

### Diagnostic Workflow

```bash
# 1. Check what codes you have
autocognitix dtc info P0171
autocognitix dtc info P0174

# 2. Find related codes
autocognitix dtc related P0171

# 3. Run full diagnosis
autocognitix diagnose P0171 P0174 \
    --symptoms "Motor alapjaraton vibral, fogyasztas novekedett" \
    --make Volkswagen \
    --model Passat \
    --year 2017

# 4. If you have a VIN, decode it first
autocognitix vehicle decode WVWZZZ3CZWE123456
```

### Scripting with JSON Output

```bash
# Search and process with jq
autocognitix dtc search "oxygen" --json | jq '.codes[].code'

# Get diagnosis result
autocognitix diagnose P0171 --json > diagnosis_result.json

# Statistics for monitoring
autocognitix stats --json | jq '.translation_percentage'
```

### Batch Processing

```bash
# Process multiple codes
for code in P0171 P0174 P0300 P0420; do
    echo "=== $code ==="
    autocognitix dtc info $code
    echo
done

# Search multiple terms
for term in "oxygen" "fuel" "misfire"; do
    autocognitix dtc search "$term" --limit 5
done
```

### Integration with Backend API

```bash
# Configure API endpoint
autocognitix config --set api_url http://api.autocognitix.com/api/v1
autocognitix config --set use_api true

# Now all commands will use the API
autocognitix dtc search "oxygen sensor"
autocognitix diagnose P0171 --symptoms "Motor vibral"
```

## Verbose Output

Use `-v` or `-vv` for more detailed output:

```bash
# Info level logging
autocognitix -v dtc search "oxygen"

# Debug level logging
autocognitix -vv diagnose P0171 --symptoms "Motor vibral"
```

## Output Formats

### Table Format (Default)

Human-readable tables with colored output for severity levels:
- Critical: Red
- High: Orange
- Medium: Yellow
- Low: Green

### JSON Format

Use `--json` flag for machine-readable output:

```bash
autocognitix dtc info P0420 --json
```

Output:
```json
{
  "code": "P0420",
  "description_en": "Catalyst System Efficiency Below Threshold (Bank 1)",
  "description_hu": "Katalizator rendszer hatekonysaga a kuszobertek alatt (1. bank)",
  "category": "powertrain",
  "severity": "medium",
  "symptoms": [...],
  "possible_causes": [...],
  "diagnostic_steps": [...]
}
```

## Troubleshooting

### "DTC code not found"

The code may not be in the local database. Try:
1. Use `--use-api` to search the backend API
2. Check if the code format is correct (P/B/C/U + 4 digits)
3. Search with partial code: `autocognitix dtc search P04`

### "API Error"

1. Check if the backend is running
2. Verify the API URL: `autocognitix config --show`
3. Check network connectivity

### "No translation found"

1. The term may not be in the translation cache
2. Try searching for similar terms
3. Use English descriptions as fallback

### Configuration Issues

Reset to defaults:
```bash
autocognitix config --reset
```

Or manually delete the config file:
```bash
rm ~/.autocognitix.yaml
```

## Requirements

- Python 3.9+
- click >= 8.0.0
- rich >= 13.0.0
- httpx >= 0.24.0
- pyyaml >= 6.0

## Support

For issues and feature requests, please visit:
https://github.com/autocognitix/autocognitix/issues
