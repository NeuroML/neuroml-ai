#!/usr/bin/env python3
"""
Generate a single page markdown from Jupyter book sources

File: data/scripts/jupyterbook2singlemd.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""


import re
import sys
from pathlib import Path



def runner(source: str):
    """Main runner

    :param source: TODO
    :returns: TODO

    """
    toc = f"{source}/_toc.yml"
    filelist = []
    with open(toc, 'r') as toc_f:
        for line in toc_f.readlines():
            if "file:" in line:
                source_file = Path(line.split("file:")[1].strip())
                if source_file.suffix == "":
                    source_file_full = f"{source_file}.md"
                else:
                    source_file_full = str(source_file)

                # ignore notebooks
                if source_file_full.endswith("ipynb"):
                    continue
                print(f"Found: {source_file_full}")
                filelist.append(source_file_full)

    text = ""
    schema_text = ""
    # ignore lines starting with these
    start_ignores = (
        "----",
        "language: ",
        "lines: ",
        ":class: ",
        ":align: ",
        ":alt: ",
        ":scale: ",
        ":gutter: ",
        ":columns: ",
        ":widths: ",
        ":width: ",
        ":delim: ",
        "%"
    )
    refs  = {}

    replacements = {
        r"{ref}`(.+?)>`": r"\1>",
        r"{ref}`(.+?)`": r"\1",
        r"{doc}`(.+?)>`": r"\1>",
        r"{eq}`(.+?)>`": r" (see equation \1)",
        r"{cite}`(.+?)`": r"[citation: \1]",
        r"{superscript}`(.+?)`": r"^\1",
        r"{csv-table}": r"Table:",
        r"{(image|figure)} (.+)": r"\nFigure: \2",
        r"{(admonition|tip|warning|note|important)}": r"\nNOTE: ",
        r"{(code|code-block|download|tab-item|grid-item-card|grid|tab-set)}": r"",
        r"(schema:|units:|<i>|</i>|&emsp;)": r""
    }

    for srcfile in filelist:
        srcfilepath = Path(f"{source}/{srcfile}")
        print(f"Processing {srcfilepath}")
        adding_text_to = ""

        with open(srcfilepath, 'r') as srcfile_f:
            in_block = False
            section_ref = ""
            for line in srcfile_f.readlines():
                # handle code includes
                if line.startswith(start_ignores):
                    continue

                # section heading
                if line.startswith("(") and line.strip().endswith(")="):
                    section_ref = f"<{line[1:-3]}>"
                    continue

                if len(section_ref) > 0:
                    refs[section_ref] = "(see section: " + line.replace("#", "").strip() + ")"
                    section_ref = ""

                if "{literalinclude}" in line:
                    in_block = True
                    file_to_include = line.split("{literalinclude}")[1].strip()
                    with open(f"{srcfilepath.parent}/{file_to_include}", 'r') as incfile_f:
                        included_cont = incfile_f.read()
                        adding_text_to += f"```\n\n{included_cont}\n\n```\n"
                # exit the block
                elif "```" in line and in_block:
                    adding_text_to += "\n"
                    in_block = False
                else:
                    adding_text_to += line

        if "Schemas/" in str(srcfilepath):
            schema_text += adding_text_to
        else:
            text += adding_text_to



    for pat, rep in replacements.items():
        text = re.sub(pat, rep, text, count=0)

    for pat, rep in refs.items():
        text = re.sub(pat, rep, text, count=0)

    for pat, rep in replacements.items():
        schema_text = re.sub(pat, rep, schema_text, count=0)

    for pat, rep in refs.items():
        schema_text = re.sub(pat, rep, schema_text, count=0)

    with open("single-page-markdown.md", 'w') as out:
        print(text, file=out)

    with open("single-page-markdown-schema.md", 'w') as out:
        print(schema_text, file=out)

    # print(refs)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Only one argument permitted: location of source folder")
    runner(sys.argv[1])
