#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
# File : scripts/run-docker.sh
#
# build and run docker

docker build -f docker/Dockerfile -t nml-ai . && docker run --env HF_TOKEN_NML_AI=$HF_TOKEN_NML_AI -it -p 7860:7860 nml-ai
