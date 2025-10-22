# NeuroML-AI

AI assistant for helping with NeuroML queries and model generation.

## Set up

The Python packages can be installed using the provided `requirements.txt` file.

### Models

The following models are currently being used for testing:

#### Ollama for local deployments

Currently, Ollama is used for local deployments with the qwen3:0.6b chat model and qwen3-embeddings:0.6b embedding model.
To install Ollama and pull the models, please follow the official documentation: https://ollama.com/download

#### Gemini

You can use the Gemini chat model and embeddings.
For this, please export the `GOOGLE_API_KEY` environment variable.

## License

This code is licensed under the MIT license.
Please refer to the licenses of the various LLM models for information on their licensing and usage terms.
