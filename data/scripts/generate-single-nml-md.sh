#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com> 
# File : data/scripts/generate-single-nml-md.sh
#


git clone --depth 1 https://github.com/NeuroML/Documentation nml-docs
python3 jupyterbook2singlemd.py "nml-docs/source"
cp -v single-page-markdown.md nml-docs-single-page.md

echo "Please delete the nml-docs folder"
