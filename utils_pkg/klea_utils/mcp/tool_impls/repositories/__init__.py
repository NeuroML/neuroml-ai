#!/usr/bin/env python3
"""
Repository source implementations (GitHub, FigShare, DANDI, BioModels).

Framework-agnostic functions that list the versions and files of archival
repositories.  The returned file lists carry direct ``download_url`` values
that can be fed to the shared ``download_file`` / ``download_file_to_cache``
implementations.  Import the per-source functions from their modules, e.g.
``klea_utils.mcp.tool_impls.repositories.github``.
"""
