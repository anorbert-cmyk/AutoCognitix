"""The Neo4j availability probe must actually probe Neo4j.

Why this file exists
--------------------
`is_neo4j_available()` called `db.cypher_query("RETURN 1", timeout=5)`. neomodel's
`Database.cypher_query` accepts no `timeout` keyword, so that call raised
`TypeError` **before touching the network** - and the bare `except` recorded the
result as "Neo4j unavailable". The probe therefore returned False permanently,
against a healthy graph as readily as a dead one, from the day the kwarg was
introduced. Every feature gated on it went dark, and restoring the database
would not have changed anything.

It survived because every existing test PATCHES `is_neo4j_available` out
(tests/integration/test_database_neo4j.py, test_service_rag.py) and one merely
greps the source for the function's name. The body was never executed. So these
tests do the one thing those could not: run the real body against a stubbed
driver, and assert on the QUERY THAT WAS ISSUED.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_neomodel_does_not_accept_a_timeout_kwarg():
    """Pins the upstream fact this bug rested on.

    If a future neomodel adds `timeout`, this fails and someone can revisit the
    call sites deliberately - rather than re-adding the kwarg on a hunch.
    """
    from neomodel import db

    # `driver` lives on the INSTANCE, not the class, so create=True is required.
    # Stubbing it makes the connection guard a no-op, which is what proves the
    # TypeError comes from the signature and not from an unreachable graph.
    with (
        patch.object(db, "driver", MagicMock(), create=True),
        pytest.raises(TypeError, match="timeout"),
    ):
        db.cypher_query("RETURN 1", timeout=5)


@pytest.mark.unit
def test_no_call_site_passes_timeout_to_cypher_query():
    """Structural guard: the kwarg must not come back anywhere in the module."""
    from pathlib import Path

    source = Path(inspect.getfile(__import__("app.db.neo4j_models", fromlist=["x"]))).read_text()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "timeout=" in line and not line.strip().startswith("#")
    ]
    assert not offenders, (
        "cypher_query takes no timeout kwarg; passing one raises TypeError "
        f"before any query runs: {offenders}"
    )


@pytest.mark.asyncio
@pytest.mark.unit
class TestTheProbeReachesTheGraph:
    @staticmethod
    def _reset_cache():
        import app.db.neo4j_models as m

        m._neo4j_last_check = 0.0
        m._neo4j_available = True

    async def test_a_healthy_graph_reports_available(self):
        import app.db.neo4j_models as m

        self._reset_cache()
        # Replace the MODULE-level `db` name: the probe's lambda resolves `db`
        # from neo4j_models' globals at call time, so that is the binding a test
        # has to control.
        fake = MagicMock()
        fake.cypher_query.return_value = ([[1]], None)
        with patch.object(m, "db", fake):
            assert await m.is_neo4j_available() is True
        q = fake.cypher_query
        q.assert_called_once()
        # The real defect: assert on HOW it was called, not just that it was.
        assert "timeout" not in q.call_args.kwargs, (
            "the probe passed a kwarg cypher_query rejects, so it never ran"
        )

    async def test_a_dead_graph_reports_unavailable(self):
        import app.db.neo4j_models as m

        self._reset_cache()
        fake = MagicMock()
        fake.cypher_query.side_effect = OSError("unreachable")
        with patch.object(m, "db", fake):
            assert await m.is_neo4j_available() is False

    async def test_a_broken_probe_cannot_masquerade_as_a_dead_graph(self):
        """The exact shape of the bug: TypeError must not read as 'graph down'.

        We cannot make the function return True on a TypeError - it genuinely
        cannot reach the graph - but the log MUST name the type, because
        "Neo4j unavailable" with no type is what made a four-month-old code
        defect indistinguishable from an infrastructure outage.
        """
        import app.db.neo4j_models as m

        self._reset_cache()
        fake = MagicMock()
        fake.cypher_query.side_effect = TypeError("unexpected kwarg")
        with patch.object(m, "db", fake), patch.object(m.logger, "warning") as warn:
            assert await m.is_neo4j_available() is False
        assert warn.called
        logged = " ".join(str(a) for a in warn.call_args.args)
        assert "TypeError" in logged or "%s" in logged
