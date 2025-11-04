# NeuroML-AI

AI assistant for helping with NeuroML queries and model generation.

[![GitHub CI](https://github.com/NeuroML/neuroml-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/NeuroML/neuroml-ai/actions/workflows/ci.yml)
[![GitHub](https://img.shields.io/github/license/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/blob/master/LICENSE)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/pulls)
[![GitHub issues](https://img.shields.io/github/issues/NeuroML/neuroml-ai)](https://github.com/NeuroML/neuroml-ai/issues)
[![GitHub Org's stars](https://img.shields.io/github/stars/NeuroML?style=social)](https://github.com/NeuroML)
[![Twitter Follow](https://img.shields.io/twitter/follow/NeuroML?style=social)](https://twitter.com/NeuroML)
[![Gitter](https://badges.gitter.im/NeuroML/community.svg)](https://gitter.im/NeuroML/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


Please note that this project is under active development and does not currently provide a stable release/API/ABI.

## Set up

The package is `pip` installable.

### Models

The following models are currently being used for testing:

#### Ollama for local deployments

Currently, Ollama is used for local deployments.
Please see the code to see what models are currently used.
You can modify these to use different models to suit your hardware.
However, do note that picking smaller models will most certainly affect the correctness/performance of the RAG.
To install Ollama and pull the models, please follow the official documentation: https://ollama.com/download

#### Gemini

You can use the Gemini chat model and embeddings.
For this, please export the `GOOGLE_API_KEY` environment variable.

## License

This code is licensed under the MIT license.
Please refer to the licenses of the various LLM models for information on their licensing and usage terms.
