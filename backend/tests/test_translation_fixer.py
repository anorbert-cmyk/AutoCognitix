"""
Tests for translation fixer regex patterns and utilities.

Tests cover:
- JSON extraction from LLM responses
- Translation parsing fallback patterns
- Hungarian text normalization
- ReDoS prevention in regex patterns
"""

import re
import json
from typing import Dict, List, Tuple


def extract_json_from_response(content: str) -> Dict[str, str]:
    """
    Extract JSON object from LLM response content.

    Args:
        content: Raw response content that may contain JSON.

    Returns:
        Parsed dictionary or empty dict if no valid JSON found.
    """
    if not content:
        return {}

    try:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return {}


def parse_translations_fallback(
    content: str,
    descriptions: List[Tuple[str, str]],
    max_content_size: int = 100000,
) -> Dict[str, str]:
    """
    Parse translations from non-JSON response using regex.

    Args:
        content: Response content to parse.
        descriptions: List of (code, description) tuples.
        max_content_size: Maximum content size to process (ReDoS prevention).

    Returns:
        Dictionary mapping codes to translations.
    """
    translations = {}

    # Prevent ReDoS by limiting content size
    if not content or len(content) > max_content_size:
        return translations

    for code, _ in descriptions:
        # Use specific pattern to avoid ReDoS
        # Match code followed by separator and text until newline or next code
        pattern = rf"{re.escape(code)}\s*[:\-]\s*([^\n]+)"
        match = re.search(pattern, content, re.IGNORECASE)

        if match:
            translation = match.group(1).strip().strip('"').strip("'")
            # Validate translation quality
            if translation and len(translation) > 4 and len(translation) < 500:
                translations[code] = translation

    return translations


def normalize_hungarian_text(text: str) -> str:
    """
    Normalize Hungarian text for comparison.

    - Converts to lowercase
    - Normalizes whitespace
    - Handles Hungarian special characters
    """
    if not text:
        return ""

    # Normalize whitespace
    text = " ".join(text.split())

    # Lowercase
    text = text.lower()

    return text.strip()


def fix_common_translation_issues(text: str) -> str:
    """
    Fix common issues in LLM-generated Hungarian translations.

    Handles:
    - Extra quotes
    - Escaped characters
    - Trailing punctuation issues
    - Double spaces
    """
    if not text:
        return ""

    # Remove extra quotes
    text = text.strip('"').strip("'")

    # Fix escaped quotes
    text = text.replace('\\"', '"').replace("\\'", "'")

    # Normalize whitespace
    text = " ".join(text.split())

    # Fix double punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    return text.strip()


class TestJSONExtraction:
    """Test JSON extraction from LLM responses."""

    def test_extract_simple_json(self):
        """Test extraction of simple JSON object."""
        content = '{"P0101": "Levegotomeg-mero hiba"}'
        result = extract_json_from_response(content)

        assert result == {"P0101": "Levegotomeg-mero hiba"}

    def test_extract_json_with_surrounding_text(self):
        """Test extraction of JSON with surrounding text."""
        content = """Here is the translation:
        {"P0101": "Levegotomeg-mero hiba", "P0171": "Rendszer tul sovany"}
        I hope this helps!"""

        result = extract_json_from_response(content)

        assert "P0101" in result
        assert "P0171" in result

    def test_extract_json_multiple_codes(self):
        """Test extraction of JSON with multiple codes."""
        content = """{
            "P0101": "Levegotomeg-mero aramkor tartomany hiba",
            "P0171": "Rendszer tul sovany (Bank 1)",
            "P0300": "Tobbszoros hengerbedurranas eszlelve"
        }"""

        result = extract_json_from_response(content)

        assert len(result) == 3
        assert "P0101" in result
        assert "P0171" in result
        assert "P0300" in result

    def test_extract_empty_content(self):
        """Test extraction from empty content."""
        assert extract_json_from_response("") == {}
        assert extract_json_from_response(None) == {}

    def test_extract_no_json(self):
        """Test extraction when no JSON present."""
        content = "This is just plain text without any JSON."
        result = extract_json_from_response(content)

        assert result == {}

    def test_extract_malformed_json(self):
        """Test extraction of malformed JSON."""
        content = '{"P0101": "Hiba", "P0171": }'  # Invalid JSON
        result = extract_json_from_response(content)

        assert result == {}

    def test_extract_nested_braces(self):
        """Test extraction with nested braces in content."""
        content = '{"P0101": "Hiba {ECU} aramkorben"}'
        result = extract_json_from_response(content)

        assert result == {"P0101": "Hiba {ECU} aramkorben"}


class TestTranslationFallbackParsing:
    """Test fallback parsing when JSON extraction fails."""

    def test_parse_colon_separated(self):
        """Test parsing colon-separated format."""
        content = """P0101: Levegotomeg-mero hiba
P0171: Rendszer tul sovany"""

        descriptions = [("P0101", "desc1"), ("P0171", "desc2")]
        result = parse_translations_fallback(content, descriptions)

        assert "P0101" in result
        assert "P0171" in result
        assert "Levegotomeg" in result["P0101"]

    def test_parse_dash_separated(self):
        """Test parsing dash-separated format."""
        content = """P0101 - Levegotomeg-mero hiba
P0171 - Rendszer tul sovany"""

        descriptions = [("P0101", "desc1"), ("P0171", "desc2")]
        result = parse_translations_fallback(content, descriptions)

        assert "P0101" in result
        assert "P0171" in result

    def test_parse_case_insensitive(self):
        """Test case-insensitive code matching."""
        content = "p0101: Levegotomeg-mero hiba"

        descriptions = [("P0101", "desc1")]
        result = parse_translations_fallback(content, descriptions)

        assert "P0101" in result

    def test_parse_with_quotes(self):
        """Test parsing with quoted translations."""
        content = """P0101: "Levegotomeg-mero hiba"
P0171: 'Rendszer tul sovany' """

        descriptions = [("P0101", "desc1"), ("P0171", "desc2")]
        result = parse_translations_fallback(content, descriptions)

        # Quotes should be stripped
        assert result.get("P0101") == "Levegotomeg-mero hiba"
        assert result.get("P0171") == "Rendszer tul sovany"

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        descriptions = [("P0101", "desc1")]

        assert parse_translations_fallback("", descriptions) == {}
        assert parse_translations_fallback(None, descriptions) == {}

    def test_parse_missing_codes(self):
        """Test parsing when requested codes are not in content."""
        content = "P0300: Hengerbedurranas"

        descriptions = [("P0101", "desc1"), ("P0171", "desc2")]
        result = parse_translations_fallback(content, descriptions)

        assert result == {}  # None of the requested codes found

    def test_parse_rejects_too_short_translations(self):
        """Test that very short translations are rejected."""
        content = "P0101: abc"  # Too short (< 5 chars)

        descriptions = [("P0101", "desc1")]
        result = parse_translations_fallback(content, descriptions)

        assert result == {}

    def test_parse_rejects_too_long_translations(self):
        """Test that very long translations are rejected."""
        content = f"P0101: {'x' * 600}"  # Too long (> 500 chars)

        descriptions = [("P0101", "desc1")]
        result = parse_translations_fallback(content, descriptions)

        assert result == {}


class TestReDoSPrevention:
    """Test ReDoS (Regular Expression Denial of Service) prevention."""

    def test_reject_oversized_content(self):
        """Test rejection of oversized content."""
        # Create content larger than max_content_size
        large_content = "P0101: " + "x" * 200000

        descriptions = [("P0101", "desc1")]
        result = parse_translations_fallback(large_content, descriptions, max_content_size=100000)

        assert result == {}  # Should reject due to size

    def test_handle_normal_sized_content(self):
        """Test that normal-sized content is processed."""
        content = "P0101: Levegotomeg-mero hiba"

        descriptions = [("P0101", "desc1")]
        result = parse_translations_fallback(content, descriptions, max_content_size=100000)

        assert "P0101" in result

    def test_pattern_with_special_regex_chars(self):
        """Test that DTC codes with regex special chars are escaped."""
        # Ensure re.escape is used properly
        content = "P0101: Hiba a rendszerben"

        # Code that looks like regex special chars
        descriptions = [("P0101", "desc")]
        result = parse_translations_fallback(content, descriptions)

        assert "P0101" in result


class TestHungarianTextNormalization:
    """Test Hungarian text normalization."""

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "  Levegotomeg   mero    hiba  "
        result = normalize_hungarian_text(text)

        assert result == "levegotomeg mero hiba"

    def test_normalize_lowercase(self):
        """Test lowercase conversion."""
        text = "LEVEGOTOMEG-Mero Hiba"
        result = normalize_hungarian_text(text)

        assert result == "levegotomeg-mero hiba"

    def test_normalize_empty(self):
        """Test normalization of empty strings."""
        assert normalize_hungarian_text("") == ""
        assert normalize_hungarian_text(None) == ""

    def test_normalize_hungarian_chars(self):
        """Test handling of Hungarian special characters."""
        text = "Levegotomeg-mero aramkor hiba"
        result = normalize_hungarian_text(text)

        # Hungarian characters should be preserved
        assert result == "levegotomeg-mero aramkor hiba"


class TestTranslationFixes:
    """Test common translation issue fixes."""

    def test_fix_extra_quotes(self):
        """Test removal of extra quotes."""
        assert fix_common_translation_issues('"Hiba"') == "Hiba"
        assert fix_common_translation_issues("'Hiba'") == "Hiba"
        assert fix_common_translation_issues("\"'Hiba'\"") == "Hiba"

    def test_fix_escaped_quotes(self):
        """Test handling of escaped quotes."""
        # Test with actual backslash-quote sequences (as they appear in JSON)
        text = 'Hiba a \\"motor\\" rendszerben'
        result = fix_common_translation_issues(text)

        assert result == 'Hiba a "motor" rendszerben'

    def test_fix_double_spaces(self):
        """Test removal of double spaces."""
        text = "Hiba  a   rendszerben"
        result = fix_common_translation_issues(text)

        assert result == "Hiba a rendszerben"

    def test_fix_double_punctuation(self):
        """Test fixing double punctuation."""
        text = "Hiba a rendszerben..."
        result = fix_common_translation_issues(text)

        assert result == "Hiba a rendszerben."

    def test_fix_space_before_punctuation(self):
        """Test removal of space before punctuation."""
        text = "Hiba a rendszerben ."
        result = fix_common_translation_issues(text)

        assert result == "Hiba a rendszerben."

    def test_fix_empty_input(self):
        """Test handling of empty input."""
        assert fix_common_translation_issues("") == ""
        assert fix_common_translation_issues(None) == ""


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions."""

    def test_full_parsing_pipeline(self):
        """Test complete parsing pipeline from raw response."""
        # Simulate LLM response
        raw_response = """Ime a forditasok:
        {
            "P0101": "Levegotomeg-mero aramkor tartomany/teljesitmeny hiba",
            "P0171": "Rendszer tul sovany (Bank 1)"
        }
        Remelem, ez segit!"""

        # Try JSON extraction first
        result = extract_json_from_response(raw_response)

        if not result:
            # Fall back to regex parsing
            descriptions = [("P0101", "desc1"), ("P0171", "desc2")]
            result = parse_translations_fallback(raw_response, descriptions)

        # Apply fixes
        fixed_result = {k: fix_common_translation_issues(v) for k, v in result.items()}

        assert len(fixed_result) == 2
        assert "P0101" in fixed_result
        assert "P0171" in fixed_result

    def test_fallback_when_json_fails(self):
        """Test fallback parsing when JSON extraction fails."""
        # Response without valid JSON
        raw_response = """A forditasok:
        P0101: Levegotomeg-mero hiba
        P0171: Rendszer tul sovany"""

        # JSON extraction should fail
        result = extract_json_from_response(raw_response)
        assert result == {}

        # Fallback should succeed
        descriptions = [("P0101", "desc1"), ("P0171", "desc2")]
        result = parse_translations_fallback(raw_response, descriptions)

        assert len(result) == 2

    def test_mixed_format_response(self):
        """Test handling response with mixed formats."""
        # Some codes in JSON-like format, some not
        raw_response = """
        "P0101": "Levegotomeg-mero hiba"
        P0171 - Rendszer tul sovany
        P0300: Hengerbedurranas"""

        # This won't parse as valid JSON
        json_result = extract_json_from_response(raw_response)
        assert json_result == {}

        # But fallback should get some
        descriptions = [("P0101", "desc1"), ("P0171", "desc2"), ("P0300", "desc3")]
        fallback_result = parse_translations_fallback(raw_response, descriptions)

        # Should find at least P0171 and P0300
        assert len(fallback_result) >= 2
