import os

import pytest

from tools.path_safety import safe_export_path


def test_relative_filename_resolves_inside_base(tmp_path):
    base = str(tmp_path)
    result = safe_export_path("curves.csv", base)
    assert result == os.path.join(base, "curves.csv")


def test_nested_relative_path_allowed(tmp_path):
    base = str(tmp_path)
    result = safe_export_path("sub/curves.csv", base)
    assert result == os.path.join(base, "sub", "curves.csv")


def test_absolute_path_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_export_path("/etc/passwd", str(tmp_path))


def test_parent_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_export_path("../escape.csv", str(tmp_path))


def test_sneaky_nested_traversal_rejected(tmp_path):
    with pytest.raises(ValueError):
        safe_export_path("sub/../../escape.csv", str(tmp_path))


def test_empty_or_none_returns_none(tmp_path):
    assert safe_export_path("", str(tmp_path)) is None
    assert safe_export_path(None, str(tmp_path)) is None
