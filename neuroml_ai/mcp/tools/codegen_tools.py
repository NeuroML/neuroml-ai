#!/usr/bin/env python3
"""
Tools for generating NeuroML code

File: codegen_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from textwrap import dedent


async def dummy_tool(astring: str) -> str:
    """Dummy tool that returns the string given to it. Doesn't do anything
    else. Only here for unit testing.

    :param astring: a string
    :type astring: str
    :returns: the given string in a sentence
    :rtype: str

    """
    return f"I got {astring}"


def create_new_NeuroML_model_tool(model_name: str = "NeuroMLModel") -> str:
    """Create a new blank NeuroML model.

    This function will create a new blank NeuroML model that includes some
    basic scaffolding that can be filled in to complete the model.

    :param model_name: name of the model
    :type model_name: str
    :returns: a model template
    :rtype: str

    """

    model_name = model_name.replace(" ", "")

    model_str = dedent(
    f"""
    from neuroml.utils import component_factory

    nml_document = component_factory(neuroml.NeuroMLDocument, id=network_name)
    network = nml_document.add(neuroml.Network, id="{model_name}", validate=False)

    """
    )

    return model_str
