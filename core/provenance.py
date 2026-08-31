"""Plot provenance sidecars (Tier 3): <plot>.png.prov.json next to each
generated figure — session/turn, the producing tool, its (compacted)
parameter values, and the compute tool behind an auto-chained plot.

Local reproducibility metadata: unlike trace events (names-not-values, may be
exported), sidecars deliberately carry parameter VALUES; they live next to
the artifact and are never exported anywhere.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

GENERATOR = "seismic-chatbot"


def write_plot_provenance(image_path: str, payload: Dict[str, Any]) -> Optional[str]:
    """Write <image_path>.prov.json; returns the sidecar path, or None on failure."""
    sidecar = f"{image_path}.prov.json"
    record: Dict[str, Any] = {
        "artifact": os.path.basename(image_path),
        "generator": GENERATOR,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    record.update(payload)
    try:
        with open(sidecar, "w") as f:
            json.dump(record, f, default=str, indent=2)
        return sidecar
    except Exception as e:
        logger.warning(f"provenance sidecar failed for {image_path}: {e}")
        return None
