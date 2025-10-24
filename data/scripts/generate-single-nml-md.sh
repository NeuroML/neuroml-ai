#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
# File : data/scripts/generate-single-nml-md.sh
#


git clone --depth 1 https://github.com/NeuroML/Documentation nml-docs
pushd nml-docs
    bash ./build-helper.sh -m
popd
cp -v nml-docs/source/_build/single-markdown.md ../

echo "Please delete the nml-docs folder"
