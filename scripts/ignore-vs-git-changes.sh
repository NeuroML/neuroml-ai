#!/bin/bash

# Copyright 2026 Ankur Sinha
# Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
# File : scripts/ignore-vs-git-changes.sh
#
# Toggle whether vector stores changes should be tracked by git.
# They change each time they are accessed, but we don't need to store all these
# changes and it's a pain to keep stashing them, each time pre-commit runs, for
# example
#
#
#

mapfiles -d $'\0' VECTOR_STORES_DIR < <(find . -name "vector*" -type d -print0)

ignore () {
    for folder in "${VECTOR_STORES_DIR[@]}"
    do
        pushd "$folder" || exit 1
            find . -type f -print -execdir git update-index --assume-unchanged '{}' \;
        popd || exit 1
    done
    echo > "VECTOR_STORES_IGNORED"
}

unignore () {
    for folder in "${VECTOR_STORES_DIR[@]}"
    do
        pushd "$folder" || exit 1
            find . -type f -print -execdir git update-index --no-assume-unchanged '{}' \;
        popd || exit 1
    done
    rm -f "VECTOR_STORES_IGNORED"
}

if [ "$#" -ne 1 ]
then
    echo "Only one argument allowed: i/u"
fi

if  [ "$1" == "i" ]
then
    ignore
    git status -s
elif [ "$1" == "u" ]
then
    unignore
    git status -s
else
    echo "Only one argument allowed: i/u"
fi
