# NeuroML-AI

AI assistant for helping with NeuroML queries and model generation.

[![GitHub CI](https://github.com/NeuroML/neuroml-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuroML/neuroml-ai/actions/workflows/ci.yml)
[![GitHub](https://img.shields.io/github/license/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/blob/master/LICENSE)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/pulls)
[![GitHub issues](https://img.shields.io/github/issues/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/issues)
[![GitHub Org's stars](https://img.shields.io/github/stars/NeuroML?style=social)](https://github.com/NeuroML)
[![Twitter Follow](https://img.shields.io/twitter/follow/NeuroML?style=social)](https://twitter.com/NeuroML)
[![Gitter](https://badges.gitter.im/NeuroML/community.svg)](https://gitter.im/NeuroML/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


![Langchain schematic](neuroml_ai/nml-ai-lang-graph.png "Langchain schematic")

Please note that this project is under active development and does not currently provide a stable release/API/ABI.

## Set up

The package consists of three services:

- the AI assistant backend: this includes the RAG and LLM/agentic logic. This is exposed to users via a FastAPI based REST API.
- the MCP server: this is where we define various "tools" that the LLM can use in its agents.
- the frontend: currently, there's a streamlit UI, but one can use anything to communicate with the REST API---even just the swagger interface.

So, before using the frontend, one must start the MCP server and the FastAPI server.
The `devstack.sh` script in the `scripts` folder does this for now.
Please take a look at it and feel free to modify it for your own deployments.

Please note that the ports are currently hard-coded in the code.

### Models

The following models are currently being used for testing:

#### Ollama for local deployments

Currently, Ollama is used for local deployments.
Please see the code to see what models are currently used.
You can modify these to use different models to suit your hardware.
However, do note that picking smaller models will most certainly affect the correctness/performance of the RAG.
To install Ollama and pull the models, please follow the official documentation: https://ollama.com/download

#### HuggingFace

Models via inference providers on [HuggingFace](https://huggingface.co/models) are also supported.
Before using these, please ensure that you have a token on HuggingFace and are logged in:

```
hf auth login
```

#### Other services

Since this project uses LangChain, you can use any model that is supported by LangChain.
In most cases, you will need to declare some environment variables that will contain your API keys.
Please see the LangChain documentation for more information:

https://docs.langchain.com/oss/python/integrations/providers/overview


## License

This code is licensed under the MIT license.
Please refer to the licenses of the various LLM models for information on their licensing and usage terms.
