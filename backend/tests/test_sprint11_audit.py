"""Sprint 11 Audit Tests.

Verifies fixes applied during Sprint 11 by inspecting source code.
All tests are synchronous and use no external dependencies beyond pytest.
"""

from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "app"
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "src"


def _read_backend(relative_path: str) -> str:
    """Read a backend source file."""
    filepath = BACKEND_DIR / relative_path
    return filepath.read_text(encoding="utf-8")


def _read_frontend(relative_path: str) -> str:
    """Read a frontend source file."""
    filepath = FRONTEND_DIR / relative_path
    return filepath.read_text(encoding="utf-8")


# =============================================================================
# TestDiagnosisServiceFixes
# =============================================================================


class TestDiagnosisServiceFixes:
    """Verify fixes in diagnosis_service.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read_backend("services/diagnosis_service.py")

    def test_save_diagnosis_session_flushes(self):
        """_save_diagnosis_session must call db.flush() (commit handled by get_db dependency)."""
        # Find the _save_diagnosis_session method body
        method_start = self.source.find("async def _save_diagnosis_session")
        assert method_start > 0, "_save_diagnosis_session method not found"

        # Find the next method definition to delimit the body
        next_method = self.source.find("\n    async def ", method_start + 1)
        if next_method == -1:
            next_method = len(self.source)
        method_body = self.source[method_start:next_method]

        assert "await self.db.flush()" in method_body, (
            "_save_diagnosis_session must call await self.db.flush() "
            "(commit is handled by the get_db dependency auto-commit)"
        )

    def test_fallback_diagnosis_has_used_fallback_true(self):
        """_fallback_diagnosis must return used_fallback=True."""
        method_start = self.source.find("def _fallback_diagnosis")
        assert method_start > 0, "_fallback_diagnosis method not found"

        next_method = self.source.find("\n    # ====", method_start + 1)
        if next_method == -1:
            next_method = self.source.find("\n    async def ", method_start + 1)
        if next_method == -1:
            next_method = len(self.source)
        method_body = self.source[method_start:next_method]

        assert '"used_fallback": True' in method_body, (
            "_fallback_diagnosis must return used_fallback=True"
        )

    def test_fallback_diagnosis_has_all_required_keys(self):
        """_fallback_diagnosis must return safety_warnings, diagnostic_steps,
        processing_time_ms, and model_used keys."""
        method_start = self.source.find("def _fallback_diagnosis")
        assert method_start > 0, "_fallback_diagnosis method not found"

        next_method = self.source.find("\n    # ====", method_start + 1)
        if next_method == -1:
            next_method = self.source.find("\n    async def ", method_start + 1)
        if next_method == -1:
            next_method = len(self.source)
        method_body = self.source[method_start:next_method]

        required_keys = [
            '"safety_warnings"',
            '"diagnostic_steps"',
            '"processing_time_ms"',
            '"model_used"',
        ]
        for key in required_keys:
            assert key in method_body, f"_fallback_diagnosis return dict must contain {key}"

    def test_no_hasattr_on_repair_objects(self):
        """diagnosis_service.py should use getattr() instead of hasattr() on repair objects."""
        # hasattr(repair, ...) is fragile; getattr(repair, ..., default) is preferred
        assert "hasattr(repair," not in self.source, (
            "Found hasattr(repair, ...) in diagnosis_service.py - use getattr(repair, key, default) instead"
        )


# =============================================================================
# TestRAGServiceFixes
# =============================================================================


class TestRAGServiceFixes:
    """Verify fixes in rag_service.py."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read_backend("services/rag_service.py")

    def test_no_md5_usage(self):
        """rag_service.py must NOT use hashlib.md5 (weak hash)."""
        assert "hashlib.md5" not in self.source, (
            "Found hashlib.md5 in rag_service.py - use hashlib.sha256 instead"
        )

    def test_sha256_used_for_hashing(self):
        """rag_service.py must use hashlib.sha256 for content hashing."""
        assert "hashlib.sha256" in self.source, (
            "rag_service.py should use hashlib.sha256 for hashing"
        )

    def test_no_shadowed_lambda_k(self):
        """reciprocal_rank_fusion should not use lambda k: which shadows self.k."""
        # Find the reciprocal_rank_fusion method
        method_start = self.source.find("def reciprocal_rank_fusion")
        assert method_start > 0, "reciprocal_rank_fusion method not found"

        # Find end of method (next def at same or lower indent)
        next_def = self.source.find("\n    def ", method_start + 1)
        if next_def == -1:
            next_def = len(self.source)
        method_body = self.source[method_start:next_def]

        # lambda k: would shadow self.k (the RRF constant)
        assert "lambda k:" not in method_body, (
            "reciprocal_rank_fusion uses 'lambda k:' which shadows self.k - "
            "rename the lambda parameter (e.g., lambda item_key:)"
        )

    def test_no_redundant_global_rag_service(self):
        """rag_service.py should NOT have a module-level '_rag_service = None' variable.
        The singleton pattern uses RAGService._instance instead."""
        # Check for module-level _rag_service = None pattern (outside class)
        # This would be redundant with the singleton __new__ pattern
        import re

        # Match _rag_service = None at module level (not inside a class/function)
        matches = re.findall(r"^_rag_service\s*=\s*None", self.source, re.MULTILINE)
        assert len(matches) == 0, (
            "Found redundant module-level '_rag_service = None' - "
            "RAGService uses singleton __new__ pattern instead"
        )


# =============================================================================
# TestFrontendFixes
# =============================================================================


class TestFrontendFixes:
    """Verify frontend source code fixes (source inspection)."""

    def test_no_unsplash_source_url(self):
        """ResultPage.tsx must not reference source.unsplash.com (unreliable external dependency)."""
        result_page = _read_frontend("pages/ResultPage.tsx")
        assert "source.unsplash.com" not in result_page, (
            "ResultPage.tsx should not use source.unsplash.com URLs"
        )

    def test_clickable_divs_have_role_button(self):
        """DiagnosisPage.tsx clickable divs should have role='button' for accessibility."""
        diagnosis_page = _read_frontend("pages/DiagnosisPage.tsx")

        # If there are onClick handlers on div elements, they should have role="button"
        import re

        # Find div elements with onClick
        onclick_divs = re.findall(r"<div[^>]*onClick[^>]*>", diagnosis_page)

        if onclick_divs:
            # At least one clickable div should have role="button"
            has_role_button = any(
                'role="button"' in div or "role='button'" in div for div in onclick_divs
            )
            assert has_role_button, (
                "DiagnosisPage.tsx has clickable <div onClick=...> elements "
                "without role='button' - add role='button' for accessibility"
            )
