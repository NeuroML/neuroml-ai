#!/usr/bin/env python3
"""
Tools for generating NeuroML code

File: codegen_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
from dataclasses import asdict
from textwrap import dedent
from typing import Any

import httpx
from cachetools import TTLCache
from fastmcp import Context
from fastmcp.tools import ToolResult
from klea_utils.mcp.registry import tool_meta
from klea_utils.mcp.schemas import ToolInfo
from klea_utils.mcp.tool_impls.download_file import download_file_to_cache
from klea_utils.mcp.tool_result import to_result
from klea_utils.paths import get_cache_dir
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..utils import NML_MCP_DIRS

# set the implementation for development
from .sandbox import nml_mcp_sandbox
from .sandbox.sandbox import RunCommand

sbox = nml_mcp_sandbox

logger = logging.getLogger(__name__)


MAX_RESULTS = 20
SEARCH_TIMEOUT = httpx.Timeout(30)
XML_DOWNLOAD_TIMEOUT = httpx.Timeout(60)

# Cache for search results (2 hour TTL, max 100 entries)
NEUROMLDB_SEARCH_CACHE: TTLCache[str, Any] = TTLCache(maxsize=100, ttl=7200)

# Cache for XML downloads (2 hour TTL, max 100 entries)
NEUROMLDB_XML_CACHE: TTLCache[str, Any] = TTLCache(maxsize=100, ttl=7200)

# OSBv2 cache
OSBv2_SEARCH_CACHE: TTLCache[str, Any] = TTLCache(maxsize=100, ttl=7200)

#: Dedicated httpx client for neuroml-db.org requests.
#:
#: neuroml-db.org serves an incomplete certificate chain that system CA
#: stores cannot verify ("unable to verify the first certificate"), so this
#: client disables TLS verification for that endpoint only.  This keeps the
#: shared lifespan session (used by web_fetch and the OSB tools) at full
#: verification.  TODO: re-test neuroml-db.org against the shared session
#: (verify=True) from time to time and drop this dedicated client once their
#: certificate chain is fixed.
NMLDB_CLIENT = httpx.AsyncClient(verify=False)


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
    reraise=True,
)
async def _search_neuromldb(session, url, query):
    """Search NeuroML-DB with retry logic."""
    response = await session.get(
        url,
        params={"q": query},
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response.json()


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((httpx.HTTPError, asyncio.TimeoutError)),
    reraise=True,
)
async def _search_osbv2_repos(session, url, query, content_types, user_id, max_num):
    """Search NeuroML-DB with retry logic."""
    response = await session.get(
        url,
        params={
            "q": query,
            "types": content_types,
            "user_id": user_id,
            "page": 1,
            "per_page": max_num,
        },
        timeout=SEARCH_TIMEOUT,
        follow_redirects=True,
    )
    logger.debug(f"{response.request = }")
    response.raise_for_status()
    return response.json()


@tool_meta(ToolInfo(title="Create a NeuroML model template", tags={"neuroml"}))
def create_new_NeuroML_model(model_name: str = "NeuroMLModel") -> ToolResult:
    """Create a new blank NeuroML model template.

    Use this tool to generate a starting template for NeuroML models, which
    you can then customize with cells, networks, and simulations.

    Use when:
    - Starting a new NeuroML project and need boilerplate to build on.
    - You need a minimal model structure to extend.

    Do not use for:
    - Reading existing NeuroML files (use the file reading tools instead).
    - Validating or simulating models (use the validation and simulation
      tools instead).

    Example: create_new_NeuroML_model("MyNeuralNetwork")

    Args:
        model_name: Name for the NeuroML model. Will be used as the network
            ID. Spaces are automatically removed for XML compatibility.
            Defaults to "NeuroMLModel" if not specified.

    Returns:
        Dictionary with the generated Python code template and an empty error field.
    """
    model_name = model_name.replace(" ", "")

    model_str = dedent(
        f"""
    from neuroml.utils import component_factory

    nml_document = component_factory(neuroml.NeuroMLDocument, id="{model_name}")
    network = nml_document.add(neuroml.Network, id="{model_name}", validate=False)

    """
    )

    return to_result({"template": model_str, "error": ""})


@tool_meta(
    ToolInfo(
        title="Run a LEMS simulation",
        tags={"neuroml", "local", "code"},
        destructive=True,
    )
)
async def run_lems_simulation(lems_file: str) -> ToolResult:
    """Execute a LEMS simulation using pynml and the jLEMS simulator.

    Use this tool to run NeuroML simulation files and generate results. This
    is the standard way to execute LEMS (Low Entropy Model Specification)
    simulations for NeuroML models.

    Use when:
    - You need to run a NeuroML/LEMS simulation and get its output.
    - You need to test model behaviour under different parameters.

    Do not use for:
    - Creating LEMS files (use the model generation tools instead).
    - Analysing results (use the data analysis tools instead).

    Example: run_lems_simulation("LEMS_NML2_Ex9_Dynamics.xml")

    Args:
        lems_file: Path to the LEMS simulation XML file. Can be relative or
            absolute, must be valid XML, and should reference existing
            NeuroML models. The extension should typically be .xml.

    Returns:
        Dict with simulation results: stdout, stderr, returncode (0 for
        success), and data (execution time, resource usage, etc.).
    """
    command_args = ["pynml", lems_file]
    request = RunCommand(command=command_args)
    async with sbox(".") as f:
        result = await f.run(request)
    data = asdict(result)
    # Map non-zero returncode to is_error for MCP compliance
    if data.get("returncode") not in (0, None):
        data = {
            **data,
            "error": data.get("stderr")
            or f"LEMS simulation failed with returncode {data.get('returncode')}",
        }
    else:
        data = {**data, "error": ""}
    return to_result(data)


@tool_meta(
    ToolInfo(
        title="Find models on NeuroML-db",
        tags={"neuroml", "web", "neuroml-db"},
        read_only=True,
        open_world=True,
    )
)
async def get_models_from_neuromldb(
    ctx: Context, search_query: str, num: int = 3, download: bool = False
) -> ToolResult:
    """Search and optionally download cell and ion channel models from NeuroML-DB.

    Use this tool when you need example cell or ion channel models, or want
    to download models for local use.

    Use when:
    - Finding example cell and ion channel models.
    - Downloading models for use in your project.

    Do not use for:
    - Creating or editing NeuroML models (use the model template tool instead).
    - Running simulations (use the simulation tools instead).

    Example: get_models_from_neuromldb(search_query="cerebellum", download=True)

    Args:
        search_query: search term for querying NeuroML-DB. Must be non-empty.
        num: number of search results to get (clamped to 1-20).
        download: set to true to also download the models.

    Returns:
        Dictionary of model information with metadata and model content.
    """
    if not search_query or not search_query.strip():
        return to_result({"Error": "search_query must be a non-empty string"})

    num = max(1, min(num, MAX_RESULTS))

    neuromldb_search_url = "http://neuroml-db.org/api/search"
    neuromldb_model_url = "http://neuroml-db.org/model_info?model_id="
    neuromldb_model_xml_url = "https://neuroml-db.org/render_xml_file"
    models: dict[str, Any] = {}

    logger.debug(f"Searching NeuroML-DB with query: {search_query}")

    if search_query in NEUROMLDB_SEARCH_CACHE:
        logger.debug(f"Cache hit for search: {search_query}")
        res = NEUROMLDB_SEARCH_CACHE[search_query]
    else:
        try:
            res = await _search_neuromldb(
                NMLDB_CLIENT, neuromldb_search_url, search_query
            )
            NEUROMLDB_SEARCH_CACHE[search_query] = res
        except Exception as e:  # noqa: BLE001
            error_text = f"Error searching NeuroML-DB: {e.__class__.__name__}: {e}"
            logger.error(error_text)
            return to_result({"Error": error_text})

    # Process up to num results
    for i, m in enumerate(res[:num]):
        mcopy = m.copy()
        model_id = m.get("Model_ID", f"unknown_{i}")
        mcopy["url"] = neuromldb_model_url + model_id

        if download:
            # Rate limit: sleep between requests (but not before the first)
            if i > 0:
                await asyncio.sleep(1)

            if model_id in NEUROMLDB_XML_CACHE:
                logger.debug(f"Cache hit for XML: {model_id}")
                mcopy["resource"] = NEUROMLDB_XML_CACHE[model_id]
            else:
                try:
                    xml_path = await download_file_to_cache(
                        NMLDB_CLIENT,
                        neuromldb_model_xml_url,
                        cache_dir=get_cache_dir(NML_MCP_DIRS),
                        file_name=f"{model_id}.xml",
                        params={"modelID": model_id},
                        timeout=XML_DOWNLOAD_TIMEOUT,
                    )
                    if xml_path is not None:
                        NEUROMLDB_XML_CACHE[model_id] = xml_path
                        mcopy["resource"] = xml_path
                    else:
                        logger.error(f"Could not get model xml for {model_id}")
                        mcopy["resource"] = ""
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Error downloading xml for {model_id}: {e}")
                    mcopy["resource"] = None
        else:
            mcopy["resource"] = None
        models[model_id] = mcopy

    # Success case has no top-level error field; helper treats missing/empty as is_error=False
    return to_result(models)


@tool_meta(
    ToolInfo(
        title="Find repositories on Open Source Brain",
        tags={"neuroml", "web", "osb"},
        read_only=True,
        open_world=True,
    )
)
async def get_repositories_from_open_source_brain(
    ctx: Context,
    search_query: str,
    search_data: bool = True,
    search_models: bool = True,
    num: int = 5,
) -> ToolResult:
    """Search the Open Source Brain (v2) platform for model and data repositories.

    Use this tool to find neuroscience projects and repositories indexed from
    archival platforms (GitHub, DANDI Archive, FigShare), containing
    computational models (NeuroML, NEURON, NetPyNE, Brian, etc.) and
    experimental data (often NWB). Results include the URLs to the projects'
    file storage locations, which can be passed to other tools to download
    files.

    Use when:
    - Finding neuroscience models or data by topic.
    - Locating repository URLs to download files from.

    Do not use for:
    - Directly downloading files (use the download tools instead).
    - Searching NeuroML-DB for cell models (use the NeuroML-DB search tool).

    Example: get_repositories_from_open_source_brain(search_query="cerebellum")

    Args:
        search_query: search term for querying Open Source Brain. Must be
            non-empty.
        search_data: true if data related repositories should be searched.
        search_models: true if modelling related repositories should be searched.
        num: number of search results to get (clamped to 1-20).

    Returns:
        Dictionary of repository information.
    """
    search_query = search_query.strip()
    if not search_query:
        return to_result({"Error": "search_query must be a non-empty string"})
    logger.debug(f"{search_query = }")

    num = max(1, min(num, MAX_RESULTS))

    session: httpx.AsyncClient = ctx.lifespan_context["http_session"]
    if session is None:
        return to_result({"Error": "OSB session not initialized"})

    # swagger-dev: https://workspaces.v2dev.opensourcebrain.org/api/ui/
    # swagger: https://workspaces.v2.opensourcebrain.org/api/ui/
    osb_repo_search_url = (
        "https://v2.opensourcebrain.org/proxy/workspaces/api/osbrepository"
    )

    # OSB Admin user id: we limit the search to repositories added by the Admin only
    user_id = "7aafb661-2f39-4683-8f35-528de0752dd7"
    query = f"name={search_query}+summary__like=%{search_query}%"

    if search_data and search_models:
        content_types = "experimental+modeling"
    elif search_data:
        content_types = "experimental"
    else:
        content_types = "modeling"

    repositories: dict[str, Any] = {}

    logger.debug(f"Searching OSBv2 repositories with query: {query}")

    cache_key = f"{search_query}+{content_types}"

    if search_query in OSBv2_SEARCH_CACHE:
        logger.debug(f"Cache hit for search: {cache_key}")
        res = OSBv2_SEARCH_CACHE[cache_key]
    else:
        try:
            res = await _search_osbv2_repos(
                session,
                osb_repo_search_url,
                query,
                content_types,
                user_id,
                max_num=num,
            )
            OSBv2_SEARCH_CACHE[cache_key] = res
        except Exception as e:  # noqa: BLE001
            error_text = f"Error searching OSBv2: {e.__class__.__name__}: {e}"
            logger.error(error_text)
            return to_result({"Error": error_text})

    # Process up to num results
    logger.debug(f"{res =}")
    results = res["osbrepositories"]

    for i, m in enumerate(results[:num]):
        mcopy = m.copy()

        # remove user information
        del mcopy["user"]

        repositories[m.get("id", f"unknown_{i}")] = mcopy

    return to_result(repositories)
