#!/usr/bin/env python3
"""
General code execution tools

File: neuroml_ai/mcp/tools/code_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import List

from neuroml_ai.mcp.tools.sandbox import (RunCommand, RunPythonCode, docker,
                                          local)

# set the implementation for development
sbox = local.LocalSandbox


async def dummy_code_tool(astring: str) -> str:
    """Dummy tool that returns the string given to it. Doesn't do anything
    else. Only here for unit testing. Ignore me.

    :param astring: a string
    :type astring: str
    :returns: the given string in a sentence
    :rtype: str

    """
    return f"I got {astring}"


async def run_command(command: List[str]):
    """Run a command in a shell"""
    request = RunCommand(cmd=command)
    async with sbox(".") as f:
        stdout, stderr = await f.run(request)


async def run_python_code(code: str):
    """Run given python code"""
    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        stdout, stderr = await f.run(request)
