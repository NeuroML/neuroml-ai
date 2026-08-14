#!/usr/bin/env python3
"""
Tests for the shared path permission layer.

File: utils_pkg/tests/test_tools_permission.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import pytest
from klea_utils.mcp.errors import PermissionDeniedError
from klea_utils.mcp.tools.permission import check_path_access

logger = logging.getLogger(__name__)


def _root_and_outside(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    return root, outside


def test_allows_inside_project(tmp_path):
    root, _ = _root_and_outside(tmp_path)
    (root / "sub").mkdir()
    check_path_access(root, root)
    check_path_access(root / "sub", root)
    check_path_access(root / "file.txt", root)
    check_path_access(root / "sub" / "deep.txt", root)


def test_allows_root_itself(tmp_path):
    root, _ = _root_and_outside(tmp_path)
    check_path_access(root, root)


def test_denies_outside_project(tmp_path):
    root, outside = _root_and_outside(tmp_path)
    outside.write_text("secret")
    with pytest.raises(PermissionDeniedError):
        check_path_access(outside, root)


def test_denies_sibling(tmp_path):
    root, _ = _root_and_outside(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(PermissionDeniedError):
        check_path_access(other, root)


def test_denies_dotdot_smuggling(tmp_path):
    root, outside = _root_and_outside(tmp_path)
    outside.write_text("secret")
    with pytest.raises(PermissionDeniedError):
        check_path_access(root / ".." / "outside", root)


def test_denies_symlink_escape(tmp_path):
    root, outside = _root_and_outside(tmp_path)
    outside.write_text("secret")
    link = root / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(PermissionDeniedError):
        check_path_access(link, root)


def test_allows_symlink_inside(tmp_path):
    root, _ = _root_and_outside(tmp_path)
    (root / "real.txt").write_text("x")
    link = root / "link.txt"
    link.symlink_to(root / "real.txt")
    check_path_access(link, root)


def test_default_root_is_cwd(tmp_path, monkeypatch):
    root, _ = _root_and_outside(tmp_path)
    monkeypatch.chdir(root)
    check_path_access(root / "sub")
    with pytest.raises(PermissionDeniedError):
        check_path_access(tmp_path)
