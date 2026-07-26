"""Regression tests for the Qdrant deletion guard in ``scripts/init_qdrant.py``.

Why these exist
---------------
``assert_safe_to_delete`` is the only thing standing between an operator flag and
the irreversible destruction of real vectors: ``dtc_embeddings_hu`` holds ~2,323
points and ``symptom_embeddings_hu`` ~117. Nothing in the application reads them
any more, but they are real, and ``COLLECTIONS`` does not bound the blast radius
- ``--drop --collection <name>`` resolves against the LIVE server, so the
unified ``autocognitix`` store is reachable too.

This project has already been bitten by shipping a guard nobody executes: the
frozen reference-vector check went in behind a ``skipif`` that was true in every
environment, so it ran precisely nowhere. ``scripts/`` is outside both
``testpaths`` and the coverage source, so a guard living there is invisible to CI
unless a test under ``backend/tests/`` reaches out and imports it - which is what
this module does, following the same ``SCRIPTS_DIR`` pattern as
``test_dtc_codes.py`` and ``test_dtc_extraction.py``.

No network: every client is a mock.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _load_script(name: str) -> ModuleType:
    """Import a standalone script from scripts/ without installing it."""
    path = SCRIPTS_DIR / f"{name}.py"
    if not path.exists():  # pragma: no cover - repo layout guard
        pytest.skip(f"script not found: {path}")
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pytest.importorskip("qdrant_client", reason="qdrant-client required to import the init script")
init_qdrant = _load_script("init_qdrant")


def _client(counts: dict, count_raises: bool = False) -> MagicMock:
    """A fake Qdrant client whose collections hold the given point counts.

    ``SimpleNamespace`` rather than ``MagicMock`` for the collection objects:
    ``name`` is a reserved MagicMock constructor kwarg that sets the mock's own
    name instead of an attribute, so ``MagicMock(name="x").name`` is NOT "x".
    A probe written that way silently reports every collection as absent, which
    looks exactly like the guard passing.
    """
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name=n) for n in counts]
    )
    if count_raises:
        client.count.side_effect = ConnectionError("qdrant unreachable")
    else:
        client.count.side_effect = lambda collection_name, exact: SimpleNamespace(
            count=counts[collection_name]
        )
    return client


# ---------------------------------------------------------------------------
# assert_safe_to_delete
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssertSafeToDelete:
    def test_empty_collection_is_allowed(self):
        init_qdrant.assert_safe_to_delete(
            _client({"symptom_embeddings_hu": 0}), "symptom_embeddings_hu"
        )

    def test_non_empty_collection_is_refused_and_the_count_is_named(self):
        with pytest.raises(RuntimeError, match=r"2,323 points"):
            init_qdrant.assert_safe_to_delete(
                _client({"dtc_embeddings_hu": 2323}), "dtc_embeddings_hu"
            )

    def test_force_overrides_a_non_empty_collection(self):
        init_qdrant.assert_safe_to_delete(
            _client({"dtc_embeddings_hu": 2323}), "dtc_embeddings_hu", force=True
        )

    def test_an_unreadable_count_fails_closed(self):
        """Not being able to see what you destroy is not permission to destroy it."""
        with pytest.raises(RuntimeError, match=r"could not read its point count"):
            init_qdrant.assert_safe_to_delete(
                _client({"dtc_embeddings_hu": 1}, count_raises=True), "dtc_embeddings_hu"
            )

    def test_force_does_not_bypass_an_unreadable_count(self):
        """--force means "destroy a known quantity", and here the quantity is unknown.

        The message must not tell the operator to retry with --force, because
        that produces a byte-identical refusal.
        """
        with pytest.raises(RuntimeError) as exc:
            init_qdrant.assert_safe_to_delete(
                _client({"dtc_embeddings_hu": 1}, count_raises=True),
                "dtc_embeddings_hu",
                force=True,
            )
        assert "--force does not bypass this" in str(exc.value)

    def test_a_null_count_is_unknown_not_empty(self):
        """`if count and not force` read None as 0 and unlocked the delete."""
        with pytest.raises(RuntimeError, match=r"no point count"):
            init_qdrant.assert_safe_to_delete(
                _client({"dtc_embeddings_hu": None}), "dtc_embeddings_hu"
            )

    def test_the_count_is_exact(self):
        """An approximate count can report 0 for a small non-empty collection."""
        client = _client({"dtc_embeddings_hu": 0})
        init_qdrant.assert_safe_to_delete(client, "dtc_embeddings_hu")
        assert client.count.call_args.kwargs["exact"] is True


# ---------------------------------------------------------------------------
# Wiring: every path that can delete must go through the guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoDeletePathBypassesTheGuard:
    def test_recreate_refuses_and_deletes_nothing(self):
        """--recreate is the path with no confirmation prompt at all."""
        client = _client({"dtc_embeddings_hu": 2323})
        config = {
            "name": "dtc_embeddings_hu",
            "vector_size": 768,
            "distance": "Cosine",
            "payload_schema": {},
        }
        with pytest.raises(RuntimeError):
            init_qdrant.create_collection(client, config, recreate=True)
        client.delete_collection.assert_not_called()
        client.create_collection.assert_not_called()

    def test_the_refusal_is_not_swallowed_into_a_false_return(self):
        """`except Exception -> return False` turns a refusal into a tally line.

        cmd_init treats False as "this one had a problem, carry on", which is the
        silent-failure shape this guard exists to prevent. It must propagate.
        """
        client = _client({"dtc_embeddings_hu": 2323})
        config = {
            "name": "dtc_embeddings_hu",
            "vector_size": 768,
            "distance": "Cosine",
            "payload_schema": {},
        }
        try:
            result = init_qdrant.create_collection(client, config, recreate=True)
        except RuntimeError:
            return  # propagated, correct
        pytest.fail(f"refusal decayed into a return value: {result!r}")

    def test_drop_all_is_all_or_nothing(self):
        """A half-finished drop leaves the store in a state nobody chose."""
        counts = {c["name"]: 0 for c in init_qdrant.COLLECTIONS}
        # Make the LAST candidate the non-empty one, so a naive loop would have
        # already deleted every other collection before refusing.
        counts[init_qdrant.COLLECTIONS[-1]["name"]] = 117
        client = _client(counts)
        init_qdrant.get_qdrant_client = lambda: client

        with pytest.raises(RuntimeError):
            init_qdrant.cmd_drop()
        client.delete_collection.assert_not_called()

    def test_drop_targets_a_free_form_name_and_still_guards_it(self):
        """--drop --collection resolves against the live server, not COLLECTIONS.

        So the unified `autocognitix` store - every vector the app actually reads
        - is reachable here. Nothing about the COLLECTIONS list bounds this; the
        point-count guard is the only thing that does.
        """
        client = _client({"autocognitix": 54652})
        init_qdrant.get_qdrant_client = lambda: client

        with pytest.raises(RuntimeError, match=r"54,652 points"):
            init_qdrant.cmd_drop("autocognitix")
        client.delete_collection.assert_not_called()

    def test_an_unreachable_qdrant_is_not_reported_as_nothing_to_drop(self):
        """`Dropped 0 collection(s)` + exit 0 hid an outage as an empty result."""
        client = MagicMock()
        client.get_collections.side_effect = ConnectionError("qdrant unreachable")
        init_qdrant.get_qdrant_client = lambda: client

        with pytest.raises(ConnectionError):
            init_qdrant.cmd_drop()
        client.delete_collection.assert_not_called()


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCliRefusesTheMisleadingCombination:
    def test_recreate_with_collection_is_rejected(self, monkeypatch):
        """--recreate never read --collection, so scoping it was a lie.

        Survivable while every deletion was unconditional; with --force it is
        not, because the operator reads `--recreate --collection X --force` as
        scoping the destruction to X and gets every collection instead.
        """
        monkeypatch.setattr(
            sys,
            "argv",
            ["init_qdrant.py", "--recreate", "--collection", "dtc_embeddings_hu", "--force"],
        )
        with pytest.raises(SystemExit) as exc:
            init_qdrant.main()
        assert exc.value.code == 2  # argparse usage error
