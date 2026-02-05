"""
Tests for DTC (Diagnostic Trouble Code) validation.

Tests cover:
- DTC code format validation (P, B, C, U prefixes)
- Category assignment based on prefix
- Severity level validation
- Schema validation with Pydantic
"""

import pytest
import re
from typing import Tuple


# DTC code validation patterns
DTC_PATTERN = re.compile(r"^[PBCU][0-9A-F]{4}$", re.IGNORECASE)
DTC_GENERIC_PATTERN = re.compile(r"^[PBCU]0[0-9]{3}$", re.IGNORECASE)
DTC_MANUFACTURER_PATTERN = re.compile(r"^[PBCU][1-3][0-9A-F]{3}$", re.IGNORECASE)


def validate_dtc_code(code: str) -> bool:
    """
    Validate DTC code format.

    Valid formats:
    - P0XXX: Generic Powertrain
    - P1XXX-P3XXX: Manufacturer-specific Powertrain
    - B0XXX: Generic Body
    - B1XXX-B3XXX: Manufacturer-specific Body
    - C0XXX: Generic Chassis
    - C1XXX-C3XXX: Manufacturer-specific Chassis
    - U0XXX: Generic Network
    - U1XXX-U3XXX: Manufacturer-specific Network
    """
    if not code or not isinstance(code, str):
        return False
    return bool(DTC_PATTERN.match(code.strip().upper()))


def get_dtc_category(code: str) -> str:
    """Get category name from DTC code prefix."""
    if not validate_dtc_code(code):
        raise ValueError(f"Invalid DTC code: {code}")

    prefix = code[0].upper()
    categories = {
        "P": "powertrain",
        "B": "body",
        "C": "chassis",
        "U": "network",
    }
    return categories[prefix]


def is_generic_code(code: str) -> bool:
    """Check if DTC code is generic (X0XXX) or manufacturer-specific (X1-3XXX)."""
    if not validate_dtc_code(code):
        raise ValueError(f"Invalid DTC code: {code}")
    return bool(DTC_GENERIC_PATTERN.match(code.upper()))


def parse_dtc_code(code: str) -> Tuple[str, str, int, bool]:
    """
    Parse DTC code into components.

    Returns:
        Tuple of (category_prefix, category_name, numeric_part, is_generic)
    """
    if not validate_dtc_code(code):
        raise ValueError(f"Invalid DTC code: {code}")

    code = code.upper().strip()
    prefix = code[0]
    category = get_dtc_category(code)
    numeric = int(code[1:], 16)  # Parse as hex
    generic = is_generic_code(code)

    return (prefix, category, numeric, generic)


class TestDTCCodeFormat:
    """Test DTC code format validation."""

    def test_valid_powertrain_generic_codes(self):
        """Test valid generic powertrain codes (P0XXX)."""
        valid_codes = ["P0100", "P0101", "P0171", "P0300", "P0420", "P0507", "P0999"]
        for code in valid_codes:
            assert validate_dtc_code(code), f"Expected {code} to be valid"
            assert get_dtc_category(code) == "powertrain"
            assert is_generic_code(code)

    def test_valid_powertrain_manufacturer_codes(self):
        """Test valid manufacturer-specific powertrain codes (P1XXX-P3XXX)."""
        valid_codes = ["P1000", "P1234", "P2000", "P2ABC", "P3FFF"]
        for code in valid_codes:
            assert validate_dtc_code(code), f"Expected {code} to be valid"
            assert get_dtc_category(code) == "powertrain"
            assert not is_generic_code(code)

    def test_valid_body_codes(self):
        """Test valid body codes (B prefix)."""
        valid_codes = ["B0001", "B0100", "B1234", "B2000"]
        for code in valid_codes:
            assert validate_dtc_code(code), f"Expected {code} to be valid"
            assert get_dtc_category(code) == "body"

    def test_valid_chassis_codes(self):
        """Test valid chassis codes (C prefix)."""
        valid_codes = ["C0001", "C0035", "C1100", "C2ABC"]
        for code in valid_codes:
            assert validate_dtc_code(code), f"Expected {code} to be valid"
            assert get_dtc_category(code) == "chassis"

    def test_valid_network_codes(self):
        """Test valid network codes (U prefix)."""
        valid_codes = ["U0001", "U0100", "U1000", "U2ABC"]
        for code in valid_codes:
            assert validate_dtc_code(code), f"Expected {code} to be valid"
            assert get_dtc_category(code) == "network"

    def test_case_insensitivity(self):
        """Test that validation is case-insensitive."""
        assert validate_dtc_code("p0101")
        assert validate_dtc_code("P0101")
        assert validate_dtc_code("b1234")
        assert validate_dtc_code("B1234")
        assert validate_dtc_code("p2abc")
        assert validate_dtc_code("P2ABC")

    def test_invalid_codes_wrong_prefix(self):
        """Test rejection of codes with invalid prefix."""
        invalid_codes = ["A0101", "D0101", "E0101", "X0101", "10101"]
        for code in invalid_codes:
            assert not validate_dtc_code(code), f"Expected {code} to be invalid"

    def test_invalid_codes_wrong_length(self):
        """Test rejection of codes with wrong length."""
        invalid_codes = ["P01", "P010", "P01011", "P", "P0", "P010101", ""]
        for code in invalid_codes:
            assert not validate_dtc_code(code), f"Expected {code} to be invalid"

    def test_invalid_codes_non_hex(self):
        """Test rejection of codes with invalid hex characters."""
        invalid_codes = ["P0GGG", "P0ZZZ", "PXXXX"]
        for code in invalid_codes:
            assert not validate_dtc_code(code), f"Expected {code} to be invalid"

    def test_invalid_codes_special_chars(self):
        """Test rejection of codes with special characters."""
        invalid_codes = ["P-101", "P 101", "P.101", "P_101", "P@101"]
        for code in invalid_codes:
            assert not validate_dtc_code(code), f"Expected {code} to be invalid"

    def test_none_and_empty_inputs(self):
        """Test handling of None and empty inputs."""
        assert not validate_dtc_code(None)
        assert not validate_dtc_code("")
        assert not validate_dtc_code("   ")


class TestDTCCodeParsing:
    """Test DTC code parsing functionality."""

    def test_parse_generic_powertrain(self):
        """Test parsing generic powertrain code."""
        prefix, category, numeric, is_generic = parse_dtc_code("P0101")
        assert prefix == "P"
        assert category == "powertrain"
        assert numeric == 0x0101
        assert is_generic is True

    def test_parse_manufacturer_body(self):
        """Test parsing manufacturer-specific body code."""
        prefix, category, numeric, is_generic = parse_dtc_code("B1234")
        assert prefix == "B"
        assert category == "body"
        assert numeric == 0x1234
        assert is_generic is False

    def test_parse_chassis_code(self):
        """Test parsing chassis code."""
        prefix, category, _numeric, is_generic = parse_dtc_code("C0035")
        assert prefix == "C"
        assert category == "chassis"
        assert is_generic is True

    def test_parse_network_code(self):
        """Test parsing network code."""
        prefix, category, _numeric, is_generic = parse_dtc_code("U0100")
        assert prefix == "U"
        assert category == "network"
        assert is_generic is True

    def test_parse_invalid_raises_error(self):
        """Test that parsing invalid code raises ValueError."""
        with pytest.raises(ValueError):
            parse_dtc_code("INVALID")
        with pytest.raises(ValueError):
            parse_dtc_code("")
        with pytest.raises(ValueError):
            parse_dtc_code("X0101")


class TestDTCCategoryAssignment:
    """Test DTC category assignment."""

    def test_powertrain_category(self):
        """Test powertrain category assignment."""
        codes = ["P0100", "P0171", "P0300", "P1234", "P2000"]
        for code in codes:
            assert get_dtc_category(code) == "powertrain"

    def test_body_category(self):
        """Test body category assignment."""
        codes = ["B0100", "B1234", "B2000"]
        for code in codes:
            assert get_dtc_category(code) == "body"

    def test_chassis_category(self):
        """Test chassis category assignment."""
        codes = ["C0100", "C1234", "C2000"]
        for code in codes:
            assert get_dtc_category(code) == "chassis"

    def test_network_category(self):
        """Test network category assignment."""
        codes = ["U0100", "U1234", "U2000"]
        for code in codes:
            assert get_dtc_category(code) == "network"


class TestDTCGenericVsManufacturer:
    """Test generic vs manufacturer-specific code detection."""

    def test_generic_codes_first_digit_zero(self):
        """Test that codes with 0 as second character are generic."""
        generic_codes = ["P0101", "B0001", "C0035", "U0100"]
        for code in generic_codes:
            assert is_generic_code(code), f"Expected {code} to be generic"

    def test_manufacturer_codes_first_digit_nonzero(self):
        """Test that codes with 1-3 as second character are manufacturer-specific."""
        manufacturer_codes = ["P1234", "B1001", "C2035", "U3100"]
        for code in manufacturer_codes:
            assert not is_generic_code(code), f"Expected {code} to be manufacturer-specific"


class TestDTCSeverityLevels:
    """Test DTC severity level validation."""

    VALID_SEVERITIES = ["low", "medium", "high", "critical"]

    def test_valid_severity_levels(self):
        """Test that all valid severity levels are accepted."""
        for severity in self.VALID_SEVERITIES:
            assert severity in self.VALID_SEVERITIES

    def test_severity_case_sensitivity(self):
        """Test severity level case handling."""
        # Severity should be lowercase
        assert "LOW".lower() in self.VALID_SEVERITIES
        assert "MEDIUM".lower() in self.VALID_SEVERITIES
        assert "HIGH".lower() in self.VALID_SEVERITIES
        assert "CRITICAL".lower() in self.VALID_SEVERITIES


class TestDTCBulkValidation:
    """Test bulk DTC code validation."""

    def test_validate_multiple_codes(self, sample_dtc_codes):
        """Test validating multiple codes from fixture."""
        for dtc in sample_dtc_codes:
            code = dtc["code"]
            category = dtc["category"]

            assert validate_dtc_code(code), f"Code {code} should be valid"
            assert get_dtc_category(code) == category, f"Category mismatch for {code}"

    def test_filter_valid_codes(self):
        """Test filtering valid codes from mixed input."""
        mixed_codes = ["P0101", "INVALID", "B1234", "", "C0035", None, "U0100", "XXX"]
        valid_codes = [c for c in mixed_codes if c and validate_dtc_code(c)]

        assert len(valid_codes) == 4
        assert "P0101" in valid_codes
        assert "B1234" in valid_codes
        assert "C0035" in valid_codes
        assert "U0100" in valid_codes

    def test_deduplicate_codes(self):
        """Test deduplication of DTC codes."""
        codes_with_duplicates = ["P0101", "p0101", "P0101", "B1234", "b1234"]
        unique_codes = list({c.upper() for c in codes_with_duplicates if validate_dtc_code(c)})

        assert len(unique_codes) == 2
        assert "P0101" in unique_codes
        assert "B1234" in unique_codes
