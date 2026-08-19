#!/bin/bash

# Copyright 2025 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File : data/scripts/generate-single-nml-md.sh
#


git clone --depth 1 https://github.com/NeuroML/Documentation nml-docs

echo "Generating bibs"
find . -name "*.bib" -type f -execdir bash -c '
    for file; do
        pandoc "$file" --citeproc -t markdown_strict -o "${file%.bib}.md"
    done
' bash {} +

python3 jupyterbook2singlemd.py "nml-docs/source"
mv -v single-page-markdown.md nml-docs-single-page.md
mv -v single-page-markdown-schema.md nml-docs-single-page-schema.md
mv -v url-map.json nml-docs-single-page-url-map.json

echo "Please delete the nml-docs folder and copy the generated file over to the data/files folder"
