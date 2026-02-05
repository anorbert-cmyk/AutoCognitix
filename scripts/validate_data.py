#!/usr/bin/env python3
"""
DTC adatvalidációs script
Ellenőrzi a DTC kódok minőségét és konzisztenciáját.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Any

# Magyar karakterek a validációhoz
HUNGARIAN_CHARS = set("áéíóöőúüűÁÉÍÓÖŐÚÜŰ")

# DTC kód formátum regex
DTC_PATTERN = re.compile(r'^[PBCU]\d{4}$')

# Gibberish detektáláshoz használt minták
GIBBERISH_PATTERNS = [
    r'akárnyomás',  # hibás fordítás
    r'erdőhez',     # teljesen rossz kontextus
    r'hűtőhoz',     # hibás fordítás
    r'benzinfüggő', # rossz fordítás (fuel sender != benzinfüggő)
    r'kommutátor',  # switch != kommutátor autós kontextusban
]

# Elvárt mezők
REQUIRED_FIELDS = ['code', 'description_en', 'category']
OPTIONAL_FIELDS = ['description_hu', 'severity', 'system', 'is_generic',
                   'symptoms', 'possible_causes', 'diagnostic_steps',
                   'related_codes', 'sources', 'source', 'manufacturer',
                   'translation_status']

# Érvényes kategóriák
VALID_CATEGORIES = {'powertrain', 'body', 'chassis', 'network'}

# Érvényes súlyossági szintek
VALID_SEVERITIES = {'critical', 'high', 'medium', 'low', None}


class ValidationReport:
    """Validációs riport generátor."""

    def __init__(self):
        self.errors: list[dict] = []
        self.warnings: list[dict] = []
        self.stats: dict[str, Any] = {}
        self.fixable_issues: list[dict] = []

    def add_error(self, code: str, field: str, message: str):
        self.errors.append({
            'code': code,
            'field': field,
            'message': message,
            'severity': 'ERROR'
        })

    def add_warning(self, code: str, field: str, message: str):
        self.warnings.append({
            'code': code,
            'field': field,
            'message': message,
            'severity': 'WARNING'
        })

    def add_fixable(self, code: str, issue_type: str, details: dict):
        self.fixable_issues.append({
            'code': code,
            'issue_type': issue_type,
            'details': details
        })

    def generate_report(self) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append("DTC ADATVALIDÁCIÓS RIPORT")
        lines.append(f"Generálva: {datetime.now().isoformat()}")
        lines.append("=" * 80)
        lines.append("")

        # Statisztikák
        lines.append("STATISZTIKÁK:")
        lines.append("-" * 40)
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Hibák
        lines.append(f"HIBÁK ({len(self.errors)} db):")
        lines.append("-" * 40)
        if self.errors:
            for err in self.errors[:50]:  # Max 50 hiba megjelenítése
                lines.append(f"  [{err['code']}] {err['field']}: {err['message']}")
            if len(self.errors) > 50:
                lines.append(f"  ... és még {len(self.errors) - 50} hiba")
        else:
            lines.append("  Nincs hiba!")
        lines.append("")

        # Figyelmeztetések
        lines.append(f"FIGYELMEZTETÉSEK ({len(self.warnings)} db):")
        lines.append("-" * 40)
        if self.warnings:
            for warn in self.warnings[:50]:
                lines.append(f"  [{warn['code']}] {warn['field']}: {warn['message']}")
            if len(self.warnings) > 50:
                lines.append(f"  ... és még {len(self.warnings) - 50} figyelmeztetés")
        else:
            lines.append("  Nincs figyelmeztetés!")
        lines.append("")

        # Javítható problémák
        lines.append(f"JAVÍTHATÓ PROBLÉMÁK ({len(self.fixable_issues)} db):")
        lines.append("-" * 40)
        if self.fixable_issues:
            issue_types = Counter(i['issue_type'] for i in self.fixable_issues)
            for issue_type, count in issue_types.items():
                lines.append(f"  {issue_type}: {count} db")
        else:
            lines.append("  Nincs javítható probléma!")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


def validate_dtc_format(code: str) -> bool:
    """Ellenőrzi a DTC kód formátumát."""
    return bool(DTC_PATTERN.match(code))


def get_category_from_code(code: str) -> str:
    """Visszaadja a várt kategóriát a kód alapján."""
    prefix_map = {
        'P': 'powertrain',
        'B': 'body',
        'C': 'chassis',
        'U': 'network'
    }
    return prefix_map.get(code[0], 'unknown')


def is_gibberish_translation(text: str) -> tuple[bool, str]:
    """Ellenőrzi, hogy a fordítás gibberish-e."""
    if not text:
        return False, ""

    text_lower = text.lower()

    for pattern in GIBBERISH_PATTERNS:
        if re.search(pattern, text_lower):
            return True, f"Gyanús minta: '{pattern}'"

    # Túl sok speciális karakter ellenőrzése
    special_ratio = sum(1 for c in text if not c.isalnum() and c not in ' -.,') / max(len(text), 1)
    if special_ratio > 0.3:
        return True, "Túl sok speciális karakter"

    return False, ""


def validate_json_structure(data: dict) -> list[str]:
    """Ellenőrzi a JSON struktúra konzisztenciáját."""
    issues = []

    if 'metadata' not in data:
        issues.append("Hiányzó 'metadata' mező")
    else:
        required_meta = ['generated_at', 'total_codes']
        for field in required_meta:
            if field not in data['metadata']:
                issues.append(f"Hiányzó metadata mező: {field}")

    if 'codes' not in data:
        issues.append("Hiányzó 'codes' mező")
    elif not isinstance(data['codes'], list):
        issues.append("A 'codes' nem lista típusú")

    return issues


def validate_codes(data: dict) -> ValidationReport:
    """Fő validációs függvény."""
    report = ValidationReport()

    # JSON struktúra ellenőrzése
    structure_issues = validate_json_structure(data)
    for issue in structure_issues:
        report.add_error("STRUCTURE", "json", issue)

    if 'codes' not in data:
        return report

    codes = data['codes']
    code_counter = Counter(item.get('code', '') for item in codes)

    # Statisztikák
    report.stats['Összes kód'] = len(codes)
    report.stats['Egyedi kódok'] = len(code_counter)
    report.stats['Duplikált kódok'] = sum(1 for c, cnt in code_counter.items() if cnt > 1)

    translated_count = 0
    empty_description_count = 0
    gibberish_count = 0
    invalid_format_count = 0
    category_mismatch_count = 0

    seen_codes = set()
    duplicates = {}

    for idx, item in enumerate(codes):
        code = item.get('code', f'UNKNOWN_{idx}')

        # DTC formátum ellenőrzés
        if not validate_dtc_format(code):
            report.add_error(code, 'code', f"Érvénytelen DTC formátum: '{code}'")
            invalid_format_count += 1
            continue

        # Duplikátum ellenőrzés
        if code in seen_codes:
            if code not in duplicates:
                duplicates[code] = []
            duplicates[code].append(idx)
            report.add_fixable(code, 'duplicate', {'indices': [idx]})
        seen_codes.add(code)

        # Kötelező mezők ellenőrzése
        for field in REQUIRED_FIELDS:
            if field not in item or item[field] is None:
                report.add_error(code, field, f"Hiányzó kötelező mező: {field}")

        # Angol leírás ellenőrzése
        desc_en = item.get('description_en', '')
        if not desc_en or not desc_en.strip():
            report.add_error(code, 'description_en', "Üres angol leírás")
            empty_description_count += 1

        # Magyar leírás ellenőrzése
        desc_hu = item.get('description_hu', '')
        if desc_hu and desc_hu.strip():
            translated_count += 1

            # Gibberish ellenőrzés
            is_gibberish, reason = is_gibberish_translation(desc_hu)
            if is_gibberish:
                report.add_warning(code, 'description_hu', f"Gyanús fordítás: {reason}")
                report.add_fixable(code, 'gibberish_translation', {
                    'description_hu': desc_hu,
                    'reason': reason
                })
                gibberish_count += 1

        # Kategória ellenőrzés
        category = item.get('category', '')
        if category:
            if category not in VALID_CATEGORIES:
                report.add_warning(code, 'category', f"Ismeretlen kategória: {category}")
            else:
                expected_category = get_category_from_code(code)
                if category != expected_category:
                    report.add_warning(code, 'category',
                        f"Kategória eltérés: '{category}' != várt '{expected_category}'")
                    category_mismatch_count += 1

        # Súlyosság ellenőrzés
        severity = item.get('severity')
        if severity and severity not in VALID_SEVERITIES:
            report.add_warning(code, 'severity', f"Ismeretlen súlyosság: {severity}")

        # Lista típusú mezők ellenőrzése
        list_fields = ['symptoms', 'possible_causes', 'diagnostic_steps', 'related_codes', 'sources']
        for field in list_fields:
            if field in item and not isinstance(item[field], list):
                report.add_error(code, field, f"'{field}' nem lista típusú")

        # Kapcsolódó kódok validálása
        related = item.get('related_codes', [])
        if related:
            for rel_code in related:
                if not validate_dtc_format(rel_code):
                    report.add_warning(code, 'related_codes',
                        f"Érvénytelen kapcsolódó kód: '{rel_code}'")

    # További statisztikák
    report.stats['Fordított kódok'] = translated_count
    report.stats['Fordítás %'] = f"{translated_count / len(codes) * 100:.1f}%" if codes else "0%"
    report.stats['Üres leírások'] = empty_description_count
    report.stats['Gyanús fordítások'] = gibberish_count
    report.stats['Érvénytelen formátum'] = invalid_format_count
    report.stats['Kategória eltérések'] = category_mismatch_count

    # Duplikátumok részletezése
    for code, indices in duplicates.items():
        report.add_error(code, 'duplicate', f"Duplikált kód, {len(indices) + 1} előfordulás")

    return report


def fix_duplicates(data: dict) -> tuple[dict, int]:
    """Eltávolítja a duplikált kódokat, megtartva a legteljesebbet."""
    if 'codes' not in data:
        return data, 0

    codes = data['codes']
    code_map: dict[str, list[dict]] = {}

    # Csoportosítás kód szerint
    for item in codes:
        code = item.get('code', '')
        if code not in code_map:
            code_map[code] = []
        code_map[code].append(item)

    # Legjobb verzió kiválasztása
    fixed_codes = []
    removed_count = 0

    for code, items in code_map.items():
        if len(items) == 1:
            fixed_codes.append(items[0])
        else:
            # Összefésülés: a legteljesebb adatokat megtartjuk
            merged = items[0].copy()
            for item in items[1:]:
                for key, value in item.items():
                    if key not in merged or merged[key] is None or merged[key] == '' or merged[key] == []:
                        merged[key] = value
                    elif isinstance(value, list) and isinstance(merged.get(key), list):
                        # Listák összefésülése (egyedi elemek)
                        merged[key] = list(set(merged[key] + value))

            # Sources összegyűjtése
            all_sources = set()
            for item in items:
                if 'sources' in item:
                    all_sources.update(item['sources'])
                if 'source' in item and item['source']:
                    all_sources.add(item['source'])
            merged['sources'] = sorted(list(all_sources))

            fixed_codes.append(merged)
            removed_count += len(items) - 1

    # Kód szerinti rendezés
    fixed_codes.sort(key=lambda x: x.get('code', ''))

    data['codes'] = fixed_codes
    data['metadata']['total_codes'] = len(fixed_codes)

    return data, removed_count


def main():
    """Fő belépési pont."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data' / 'dtc_codes'
    input_file = data_dir / 'all_codes_merged.json'

    print(f"Adatfájl betöltése: {input_file}")

    if not input_file.exists():
        print(f"HIBA: A fájl nem található: {input_file}")
        sys.exit(1)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Betöltve: {len(data.get('codes', []))} kód")
    print("")

    # Validáció
    print("Validáció futtatása...")
    report = validate_codes(data)

    # Riport generálás
    report_text = report.generate_report()
    print(report_text)

    # Riport mentése
    report_file = data_dir / 'validation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nRiport mentve: {report_file}")

    # Javítások alkalmazása, ha vannak duplikátumok
    duplicates = [i for i in report.fixable_issues if i['issue_type'] == 'duplicate']
    if duplicates:
        print(f"\n{len(duplicates)} duplikátum található. Javítás...")
        fixed_data, removed = fix_duplicates(data)

        # Mentés
        output_file = data_dir / 'all_codes_merged.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)

        print(f"Eltávolítva: {removed} duplikátum")
        print(f"Javított fájl mentve: {output_file}")

        # Újravalidálás
        print("\nÚjravalidálás...")
        new_report = validate_codes(fixed_data)
        new_report_text = new_report.generate_report()

        new_report_file = data_dir / 'validation_report_after_fix.txt'
        with open(new_report_file, 'w', encoding='utf-8') as f:
            f.write(new_report_text)
        print(f"Újravalidációs riport mentve: {new_report_file}")

    # JSON riport is
    json_report = {
        'timestamp': datetime.now().isoformat(),
        'stats': report.stats,
        'error_count': len(report.errors),
        'warning_count': len(report.warnings),
        'fixable_count': len(report.fixable_issues),
        'errors': report.errors[:100],  # Max 100 részlet
        'warnings': report.warnings[:100],
        'gibberish_translations': [
            i for i in report.fixable_issues
            if i['issue_type'] == 'gibberish_translation'
        ][:50]
    }

    json_report_file = data_dir / 'validation_report.json'
    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)
    print(f"JSON riport mentve: {json_report_file}")

    # Visszatérési kód
    if report.errors:
        print(f"\n⚠ Validáció befejezve hibákkal ({len(report.errors)} hiba)")
        sys.exit(1)
    else:
        print("\n✓ Validáció sikeres!")
        sys.exit(0)


if __name__ == '__main__':
    main()
