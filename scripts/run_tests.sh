#!/bin/bash

# Copyright 2026 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File : test.sh
#
# Scripts to run tests in all packages.


set -e

export STORES_TEST_CONFIG="stores-tests.json"

# Deterministic CLI output in CI and local runs.  rich/typer colorise help
# output based on the environment (e.g. FORCE_COLOR on CI runners); the
# ANSI escapes split tokens such as "--debug", which breaks substring
# assertions on CLI output.  NO_COLOR is the standard opt-out; TERM=dumb
# additionally overrides FORCE_COLOR (rich gives FORCE_COLOR precedence
# over NO_COLOR).
export NO_COLOR=1
export TERM=dumb
unset FORCE_COLOR 2>/dev/null || true

echo ">> Unit tests"
for d in *_pkg
do
    if [ -d "${d}/tests" ]
    then
        pushd "$d" || exit 1
        pytest -v -n auto
        popd || exit 1
    fi
done
