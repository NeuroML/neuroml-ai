#!/usr/bin/env python3
"""
Shared user-level directory utilities for Klea packages.

Wraps ``platformdirs.PlatformDirs`` to provide OS-appropriate paths for
cache, data, and config directories (``~/.cache/klea/``, ``~/.local/share/klea/``,
``~/.config/klea/`` on Linux, with equivalents on macOS and Windows).

Consumers pass an *app_name* so that different packages (``klea``, ``nml_mcp``)
get isolated directories without repeating the boilerplate.

File: klea_utils/paths.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pathlib import Path

from platformdirs import PlatformDirs


def get_cache_dir(dirs: PlatformDirs) -> Path:
    """Return the OS-appropriate per-user cache directory for *dirs*.

    On Linux: ``~/.cache/{app_name}/``
    On macOS: ``~/Library/Caches/{app_name}/``
    On Windows: ``C:\\Users\\<user>\\AppData\\Local\\{app_name}\\cache\\``
    """
    return Path(dirs.user_cache_dir)


def get_data_dir(dirs: PlatformDirs) -> Path:
    """Return the OS-appropriate per-user data directory for *dirs*.

    On Linux: ``~/.local/share/{app_name}/``
    On macOS: ``~/Library/Application Support/{app_name}/``
    On Windows: ``C:\\Users\\<user>\\AppData\\Local\\{app_name}\\``
    """
    return Path(dirs.user_data_dir)


def get_config_dir(dirs: PlatformDirs) -> Path:
    """Return the OS-appropriate per-user config directory for *dirs*.

    On Linux: ``~/.config/{app_name}/``
    On macOS: ``~/Library/Preferences/{app_name}/``
    On Windows: ``C:\\Users\\<user>\\AppData\\Roaming\\{app_name}\\``
    """
    return Path(dirs.user_config_dir)


def init_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it doesn't exist.

    :returns: The same *path* as a ``Path`` for chaining.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cleanup_dir(path: str | Path) -> None:
    """Remove all *contents* of *path* but keep the directory itself."""
    p = Path(path)
    if not p.exists():
        return
    import shutil

    for item in p.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
