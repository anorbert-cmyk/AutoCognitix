"""Prometheus endpoint-label rules for the DTC routes.

Why this is its own module
--------------------------
Two path normalizers used to exist - ``app.core.metrics.MetricsMiddleware``
``._normalize_endpoint`` (the one ``app.main`` installs) and a dead duplicate in
``app/middleware/metrics.py``. They declared colliding Prometheus ``Info``
metrics, so they could never be imported into the same interpreter and neither
could import the other; a third, metric-free module was the only place the rule
could live exactly once.

The duplicate has since been deleted (guarded by
``tests/unit/test_dtc_codes.py::test_the_dead_metrics_duplicate_stays_deleted``),
so the collision is gone - but this module stays split out. Importing the rule
from ``app.core.metrics`` would drag the whole Prometheus registry into every
consumer, and keeping it metric-free is what lets a test import it next to
anything else. That matters after this project spent a release removing ten
divergent copies of the DTC pattern.

The cardinality problem this solves
-----------------------------------
``/api/v1/dtc/{code}`` is unauthenticated. Tightening DTC recognition to the SAE
J2012 rule (second character ``0-3``) correctly stopped folding vehicle makes
like ``CHEVROLET`` into the ``{dtc_code}`` label - but both normalizers fall
through to appending the RAW segment, so every DTC-SHAPED string that fails the
rule now opens its own Prometheus time series:

    P4AAA, P5AAA, P6AAA, ... ~= [PBCU] x [4-9A-F] x hex^3 ~= 164k label values

and that is only the shaped subset; an arbitrary segment works just as well. The
endpoint answers 400, but the middleware runs first, so a single unauthenticated
sprayer can grow the registry without bound - a memory-exhaustion vector against
the scrape target, not merely untidy dashboards.

The rule
--------
Recognition stays exactly as strict as the API validators (``is_valid_dtc_code``),
so a genuine code still gets ``{dtc_code}`` and the label keeps meaning "a real
code was requested". What changes is the fallthrough: in the ``{code}`` position
- the segment right after ``/dtc/`` - anything that is not a valid code and not
one of the literal sibling routes collapses to ``{invalid_dtc}`` instead of
entering the label verbatim. That caps this route family at two extra series
while keeping junk traffic separately visible and alertable.

Scoped to the position rather than the shape on purpose: a make named ``PACED``
under ``/api/v1/vehicles/`` is not the ``{code}`` parameter and keeps its own
label, exactly as the SAE tightening intended.
"""

from typing import Optional

from app.core.dtc_codes import is_valid_dtc_code

__all__ = [
    "DTC_PLACEHOLDER",
    "DTC_ROUTE_SEGMENT",
    "DTC_STATIC_SUBPATHS",
    "INVALID_DTC_PLACEHOLDER",
    "dtc_segment_label",
]

# Router prefix that owns the {code} path parameter (see app/api/v1/router.py).
DTC_ROUTE_SEGMENT = "dtc"

# Literal (non-parameter) routes registered directly under that prefix:
# GET /search, GET /categories/list, POST /bulk. Everything else in that
# position is user input. Kept honest by
# tests/unit/test_metrics.py::test_dtc_static_subpaths_match_the_router.
DTC_STATIC_SUBPATHS = frozenset({"search", "categories", "bulk"})

DTC_PLACEHOLDER = "{dtc_code}"
INVALID_DTC_PLACEHOLDER = "{invalid_dtc}"


def dtc_segment_label(segment: str, previous_segment: Optional[str]) -> Optional[str]:
    """Prometheus label for ``segment``, or ``None`` if no DTC rule applies.

    ``None`` means "not mine" - the caller should carry on with its other
    patterns (UUID, VIN, numeric id) or keep the raw segment.

    Args:
        segment: The raw path segment being normalized.
        previous_segment: The raw segment before it, used to detect the
            ``{code}`` position. ``None`` for the first segment of a path.

    Returns:
        ``"{dtc_code}"`` for a structurally valid code anywhere in the path,
        ``"{invalid_dtc}"`` for a non-code sitting in the ``/dtc/{code}`` slot,
        ``None`` otherwise.
    """
    if is_valid_dtc_code(segment):
        return DTC_PLACEHOLDER
    if (previous_segment or "").lower() != DTC_ROUTE_SEGMENT:
        return None
    if segment.lower() in DTC_STATIC_SUBPATHS:
        return None
    return INVALID_DTC_PLACEHOLDER
