#!/usr/bin/env python3
"""Generate detailed OBDb download report."""

import json
from pathlib import Path
from datetime import datetime, timezone

data_dir = Path(__file__).parent.parent / "data" / "obdb"
signalsets_dir = data_dir / "signalsets"

# Analyze all signalsets
files = list(signalsets_dir.glob("*.json"))

repos_with_signals = []
repos_empty = []
all_signals = []
unique_pids = set()
unique_units = set()

for f in files:
    try:
        data = json.load(open(f))
        commands = data.get("commands", [])

        file_signals = []
        if isinstance(commands, list):
            for cmd in commands:
                if isinstance(cmd, dict):
                    cmd_info = cmd.get("cmd", {})
                    pid = None
                    if isinstance(cmd_info, dict):
                        for service, pid_val in cmd_info.items():
                            pid = f"{service}:{pid_val}"
                            unique_pids.add(pid)
                            break

                    for sig in cmd.get("signals", []):
                        if isinstance(sig, dict):
                            fmt = sig.get("fmt", {})
                            unit = fmt.get("unit", "") if isinstance(fmt, dict) else ""
                            if unit:
                                unique_units.add(unit)
                            file_signals.append({
                                "id": sig.get("id", ""),
                                "name": sig.get("name", ""),
                                "path": sig.get("path", ""),
                                "pid": pid,
                                "unit": unit,
                            })

        if file_signals:
            repos_with_signals.append({
                "name": f.stem,
                "signal_count": len(file_signals),
                "signals": file_signals
            })
            all_signals.extend(file_signals)
        else:
            repos_empty.append(f.stem)
    except Exception as e:
        pass

# Aggregate by category/path
paths = {}
for sig in all_signals:
    path = sig.get("path", "").split(".")[0] if sig.get("path") else "Unknown"
    paths[path] = paths.get(path, 0) + 1

# Create comprehensive summary
summary = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "overview": {
        "total_repos_downloaded": len(files),
        "repos_with_signals": len(repos_with_signals),
        "repos_empty": len(repos_empty),
        "total_signals": len(all_signals),
        "unique_pids": len(unique_pids),
        "unique_units": len(unique_units),
    },
    "signal_categories": dict(sorted(paths.items(), key=lambda x: -x[1])[:20]),
    "units_available": sorted(list(unique_units)),
    "top_repos_by_signals": [
        {"name": r["name"], "signals": r["signal_count"]}
        for r in sorted(repos_with_signals, key=lambda x: -x["signal_count"])[:30]
    ],
    "comparison_with_original": {
        "previous_vehicle_count": 191,
        "new_vehicle_count": len(repos_with_signals),
        "increase": len(repos_with_signals) - 191,
        "increase_percentage": f"{((len(repos_with_signals) - 191) / 191) * 100:.1f}%" if repos_with_signals else "N/A"
    }
}

# Save enhanced summary
with open(data_dir / "detailed_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Print report
print("=" * 70)
print("OBDB DOWNLOAD COMPLETE - DETAILED REPORT")
print("=" * 70)
print(f"Generated: {summary['generated_at']}")
print()
print("OVERVIEW")
print("-" * 70)
print(f"Total repos downloaded:      {summary['overview']['total_repos_downloaded']}")
print(f"Repos with signal data:      {summary['overview']['repos_with_signals']}")
print(f"Empty repos (placeholders):  {summary['overview']['repos_empty']}")
print(f"Total OBD signals:           {summary['overview']['total_signals']:,}")
print(f"Unique PIDs:                 {summary['overview']['unique_pids']:,}")
print(f"Unique measurement units:    {summary['overview']['unique_units']}")
print()
print("COMPARISON WITH PREVIOUS DATA")
print("-" * 70)
print(f"Previous:  {summary['comparison_with_original']['previous_vehicle_count']} vehicles")
print(f"Current:   {summary['comparison_with_original']['new_vehicle_count']} vehicles with data")
if summary['comparison_with_original']['increase'] >= 0:
    print(f"Increase:  +{summary['comparison_with_original']['increase']} ({summary['comparison_with_original']['increase_percentage']})")
else:
    print(f"Change:    {summary['comparison_with_original']['increase']}")
print()
print("SIGNAL CATEGORIES (Top 15)")
print("-" * 70)
for cat, count in list(summary['signal_categories'].items())[:15]:
    print(f"  {cat:30} {count:6,} signals")
print()
print("MEASUREMENT UNITS AVAILABLE")
print("-" * 70)
print(f"  {', '.join(summary['units_available'][:15])}")
if len(summary['units_available']) > 15:
    print(f"  ... and {len(summary['units_available']) - 15} more")
print()
print("TOP 15 REPOS BY SIGNAL COUNT")
print("-" * 70)
for repo in summary['top_repos_by_signals'][:15]:
    print(f"  {repo['name']:40} {repo['signals']:5} signals")
print("=" * 70)
print()
print("Files saved:")
print(f"  - data/obdb/signalsets/ ({len(files)} JSON files)")
print(f"  - data/obdb/parsed_vehicles.json")
print(f"  - data/obdb/detailed_summary.json")
print(f"  - data/obdb/download_summary.json")
print(f"  - data/obdb/metadata/all_repos.json")
print(f"  - data/obdb/metadata/download_state.json")
