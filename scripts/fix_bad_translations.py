#!/usr/bin/env python3
"""
Hibás fordítások javító script.
A gépi fordítás által generált értelmetlen fordításokat javítja.
"""

import json
from pathlib import Path
from datetime import datetime

# Hibás fordítás -> helyes fordítás mapping
# Az angol eredeti alapján készített helyes fordítások
TRANSLATION_FIXES = {
    # B1201-B1204: Fuel Sender (üzemanyag szintjelző)
    "B1201": {
        "description_en": "Fuel Sender Circuit Failure",
        "description_hu": "Üzemanyag szintjelző áramkör hiba"
    },
    "B1202": {
        "description_en": "Fuel Sender Circuit Open",
        "description_hu": "Üzemanyag szintjelző áramkör szakadás"
    },
    "B1203": {
        "description_en": "Fuel Sender Circuit Short To Battery",
        "description_hu": "Üzemanyag szintjelző áramkör rövidzár az akkumulátorra"
    },
    "B1204": {
        "description_en": "Fuel Sender Circuit Short To Ground",
        "description_hu": "Üzemanyag szintjelző áramkör rövidzár a testre"
    },

    # B1205-B1212: EIC Switch (műszerfal kapcsoló)
    "B1205": {
        "description_en": "EIC Switch-1 Assembly Circuit Failure",
        "description_hu": "EIC 1-es kapcsoló áramkör hiba"
    },
    "B1206": {
        "description_en": "EIC Switch-1 Assembly Circuit Open",
        "description_hu": "EIC 1-es kapcsoló áramkör szakadás"
    },
    "B1207": {
        "description_en": "EIC Switch-1 Assembly Circuit Short To Battery",
        "description_hu": "EIC 1-es kapcsoló áramkör rövidzár az akkumulátorra"
    },
    "B1208": {
        "description_en": "EIC Switch-1 Assembly Circuit Short To Ground",
        "description_hu": "EIC 1-es kapcsoló áramkör rövidzár a testre"
    },
    "B1209": {
        "description_en": "EIC Switch-2 Assembly Circuit Failure",
        "description_hu": "EIC 2-es kapcsoló áramkör hiba"
    },
    "B1210": {
        "description_en": "EIC Switch-2 Assembly Circuit Open",
        "description_hu": "EIC 2-es kapcsoló áramkör szakadás"
    },
    "B1211": {
        "description_en": "EIC Switch-2 Assembly Circuit Short To Battery",
        "description_hu": "EIC 2-es kapcsoló áramkör rövidzár az akkumulátorra"
    },
    "B1212": {
        "description_en": "EIC Switch-2 Assembly Circuit Short To Ground",
        "description_hu": "EIC 2-es kapcsoló áramkör rövidzár a testre"
    },

    # B1215-B1218: Egyéb
    "B1215": {
        "description_en": "Running Board Lamp Relay Circuit Short To Battery",
        "description_hu": "Futólámpa relé áramkör rövidzár az akkumulátorra"
    },
    "B1216": {
        "description_en": "Emergency & Road Side Assistance Switch Circuit Short To Ground",
        "description_hu": "Vészhelyzeti és útszéli segítség kapcsoló áramkör rövidzár a testre"
    },
    "B1218": {
        "description_en": "Fuel Tank Pressure Sensor Circuit Short To Battery",
        "description_hu": "Üzemanyagtartály nyomásérzékelő áramkör rövidzár az akkumulátorra"
    },

    # B1911, B1916, B1919, B1922, B1925: Air Conditioning (klíma)
    "B1911": {
        "description_en": "Air Conditioning Recirculation Actuator Feedback Circuit Short To Battery",
        "description_hu": "Klíma keringtető motor visszajelző áramkör rövidzár az akkumulátorra"
    },
    "B1916": {
        "description_en": "Main Air Conditioning Clutch Circuit Short To Battery",
        "description_hu": "Klíma kompresszor kuplung áramkör rövidzár az akkumulátorra"
    },
    "B1919": {
        "description_en": "Air Conditioning Reminder Lamp Circuit Short To Battery",
        "description_hu": "Klíma emlékeztető lámpa áramkör rövidzár az akkumulátorra"
    },
    "B1922": {
        "description_en": "Main Air Conditioning Safety Monitoring Output Circuit Short To Battery",
        "description_hu": "Klíma biztonsági figyelő kimenet áramkör rövidzár az akkumulátorra"
    },
    "B1925": {
        "description_en": "Main Air Conditioning Circuit Short To Battery",
        "description_hu": "Fő klíma áramkör rövidzár az akkumulátorra"
    },
}


def fix_translations(data: dict) -> tuple[dict, int]:
    """Javítja a hibás fordításokat."""
    if 'codes' not in data:
        return data, 0

    fixed_count = 0
    for item in data['codes']:
        code = item.get('code', '')
        if code in TRANSLATION_FIXES:
            fix = TRANSLATION_FIXES[code]
            old_hu = item.get('description_hu', '')
            item['description_hu'] = fix['description_hu']
            item['translation_status'] = 'fixed'
            print(f"  [{code}] Javítva:")
            print(f"    Régi: {old_hu}")
            print(f"    Új:   {fix['description_hu']}")
            fixed_count += 1

    return data, fixed_count


def main():
    """Fő belépési pont."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'dtc_codes'
    input_file = data_dir / 'all_codes_merged.json'

    print(f"Adatfájl betöltése: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Betöltve: {len(data.get('codes', []))} kód")
    print("")

    print("Hibás fordítások javítása...")
    print("-" * 60)
    fixed_data, fixed_count = fix_translations(data)
    print("-" * 60)
    print(f"\nÖsszesen javítva: {fixed_count} fordítás")

    if fixed_count > 0:
        # Metadata frissítése
        fixed_data['metadata']['last_translation_fix'] = datetime.now().isoformat()

        # Mentés
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
        print(f"Mentve: {input_file}")

    print("\nKész!")


if __name__ == '__main__':
    main()
