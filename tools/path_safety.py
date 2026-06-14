"""Path confinement for user/LLM-supplied file paths.

Tool parameters such as ``export_path`` are fillable by the LLM (and therefore
indirectly by the end user / prompt injection). They must never be allowed to
write outside a dedicated sandbox directory.
"""
import os
from typing import Optional


def safe_export_path(export_path: Optional[str], base_dir: str) -> Optional[str]:
    """Resolve ``export_path`` to an absolute path confined within ``base_dir``.

    Returns ``None`` when no path is requested (empty/None). Raises ``ValueError``
    for absolute paths or any path that escapes ``base_dir`` via ``..``.
    """
    if not export_path:
        return None

    if os.path.isabs(export_path):
        raise ValueError(f"export_path must be relative, got absolute path: {export_path!r}")

    base_abs = os.path.abspath(base_dir)
    candidate = os.path.abspath(os.path.normpath(os.path.join(base_abs, export_path)))

    if candidate != base_abs and not candidate.startswith(base_abs + os.sep):
        raise ValueError(f"export_path escapes the allowed directory: {export_path!r}")

    return candidate
