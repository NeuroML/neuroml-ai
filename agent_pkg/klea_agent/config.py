#!/usr/bin/env python3
"""
Configurations for the API server

File: klea_agent/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application configuration loaded from the JSON config file."""

    mcp_servers: dict[str, Any] = Field(default_factory=dict)
    providers: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)


def write_config_template(output_dir: str | Path) -> Path:
    """Write a scaffold ``klea_agent.json`` into *output_dir*.

    The template is built from the ``AppConfig`` schema defaults so every
    field is present and ready to fill in.  Refuses to overwrite an
    existing file so a real config is never clobbered.

    :param output_dir: Directory to write the template into
    :returns: Path to the written template
    :raises FileExistsError: If the target file already exists
    """
    target = Path(output_dir) / "klea_agent.json"
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing config: {target}. "
            "Use --profile <name> with a different name instead."
        )
    target.write_text(AppConfig().model_dump_json(exclude_none=True, indent=2) + "\n")
    return target
