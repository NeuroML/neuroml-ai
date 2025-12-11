#!/usr/bin/env python3
"""
A sans-sandbox executor. Only to be used for development.
Be VERY careful when using this one.

File: neuroml_ai/mcp/tools/sandbox/nope.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
from tempfile import NamedTemporaryFile

from neuroml_ai.mcp.tools.sandbox.sandbox import AsyncSandbox, RunCommand, RunPythonCode


class LocalSandbox(AsyncSandbox):
    """A local execution sandbox"""

    def __init__(self, path: str):
        """Init

        :param path: path where all commands for this context manager will be run

        """
        self.wdir = path

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, request):
        """Dummy fallback"""
        raise NotImplementedError("Not implemented")

    @AsyncSandbox.run.register  # type: ignore
    async def _(self, request: RunPythonCode):
        with NamedTemporaryFile(prefix="nml_ai", mode="w") as f:
            print(request.code, file=f)

            process = await asyncio.create_subprocess_exec(
                "python",
                f.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            return [stdout, stderr]

    @AsyncSandbox.run.register  # type: ignore
    async def _(self, request: RunCommand):
        process = await asyncio.create_subprocess_shell(
            " ".join(request.cmd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        return [stdout, stderr]
