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


def resolve_app_config_path(
    config_file: str,
    conf_dir: str | Path,
    cwd: str | Path | None = None,
) -> Path:
    """Locate the application config file, checking the working directory first.

    ``config_file`` may be an absolute path or a relative name/path.  The
    working directory is searched before *conf_dir* (the per-app config
    directory), so a profile dropped in ``CWD`` overrides one installed in
    the config directory.

    :param config_file: App config file path or bare filename
    :param conf_dir: Config directory searched after the working directory
    :param cwd: Working directory to search first (defaults to ``Path.cwd()``)
    :returns: The first existing match
    :raises FileNotFoundError: If no existing file matches
    :raises ValueError: If *config_file* is empty
    """
    if not config_file:
        raise ValueError("Empty config file name")

    cwd_path = Path(cwd) if cwd is not None else Path.cwd()
    conf_dir_path = Path(conf_dir)

    candidate = Path(config_file)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find config file: {candidate}")

    cwd_candidate = cwd_path / candidate
    if cwd_candidate.exists():
        return cwd_candidate

    conf_candidate = conf_dir_path / candidate
    if conf_candidate.exists():
        return conf_candidate

    raise FileNotFoundError(
        f"Could not find config file '{config_file}' in:\n"
        f"  {cwd_path}\n"
        f"  {conf_dir_path}\n"
        "Create it there, or select another config with --profile <name>."
    )


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
