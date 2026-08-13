#!/bin/bash

# Copyright 2026 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File : vector_stores/create-stores.sh
#


# commands to generate nml-elife vector store

DOCLING_DEVICE=cpu DOCLING_NUM_THREADS=16 klea-stores-create chunk --force nml-elife/
DOCLING_DEVICE=cpu DOCLING_NUM_THREADS=16 klea-stores-create store --collection "nml-elife" --store chroma:./nml-elife/ --metadata-map ./metadata-map.json --bm25-store nml-elife/bm25.pkl nml-elife/
