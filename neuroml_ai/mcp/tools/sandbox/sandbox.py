#!/usr/bin/env python3
"""
Sandbox interface to be used by implementations

File: mcp/tools/sandbox/sandbox.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from functools import singledispatchmethod
from typing import List

from pydantic.dataclasses import dataclass


@dataclass
class RunPythonCode:
    """Run Python code"""

    code: str


@dataclass
class RunCommand:
    """Run a command"""

    cmd: list[str]


class AsyncSandbox(AbstractAsyncContextManager, ABC):
    """Abstract async context manager class"""

    @singledispatchmethod
    @abstractmethod
    async def run(self, request):
        """Runner method to be implemented"""
        raise NotImplementedError("Not implemented")
