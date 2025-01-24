import nbformat
from nbconvert import MarkdownExporter
import os
import re
from datetime import datetime
from datetime import timezone
import json
import argparse
import shutil

meta_data = """---
date: 2025-01-22T10:36:50+02:00
lastmod: 2025-01-22
showTableOfContents: false
tags: ["physics", "neural-networks"]
title: "Solving ODE's with Physics Informed Neural Networks"
type: "post"
---
"""


def build_meta_post_section(date, meta_last_mod, tags, title):
    return f'---\ndate: {date}\nlastmod: {meta_last_mod}\nshowTableOfContents: true\ntags: {tags}\ntitle: "{title}"\ntype: "post"\n---\n'
    # return meta_data


def replace_math_delimiters(text):
    # Replace all $$...$$ first to avoid conflicts with single $
    # text = re.sub(r"\$\$(.*?)\$\$", r"\\[\1\\]", text, flags=re.DOTALL)

    # Replace all $...$ afterwards
    # text = re.sub(r"\$(.*?)\$", r"\\(\1\\)", text, flags=re.DOTALL)

    text = re.sub(
        r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", r"\\(\1\\)", text, flags=re.DOTALL
    )

    return text


def extract_title(text):
    start = text.index("#") + 1  # Find first # and move past it
    end = text.index("\n", start)  # Find next newline after #
    return text[start:end].strip()  # Return trimmed text between them


def extract_existing_date(text):
    start = text.index("date:") + 5
    end = text.index("\n", start)
    return text[start:end].strip()


def extract_existing_tags(text):
    start = text.index("tags:") + 5
    end = text.index("\n", start)
    return text[start:end].strip()


def format_tags(*args):
    if len(args) == 0:
        return None
    return json.dumps(list(args))


def create_post_folder(post_directory, post_markdown_file):
    os.mkdir(post_directory)
    with open(post_markdown_file, "w"):
        pass


def main(note_book_name, meta_tags):
    post_directory = f"./content/posts/{note_book_name}"
    post_markdown_file = f"{post_directory}/index.md"

    with open(f"notebooks/{note_book_name}.ipynb") as f:
        notebook = nbformat.read(f, as_version=4)

    markdown_exporter = MarkdownExporter()
    (body, resources) = markdown_exporter.from_notebook_node(notebook)

    now_utc = datetime.now(timezone.utc)

    if not os.path.exists(post_directory):
        create_post_folder(post_directory, post_markdown_file)

        meta_date = now_utc.strftime("%Y-%m-%d")

        if meta_tags is None:
            raise Exception("Need to specify tags on the first export")
    else:
        with open(post_markdown_file, "r") as f:
            content = f.read()

        meta_date = extract_existing_date(content)
        meta_tags = extract_existing_tags(content) if meta_tags is None else meta_tags

        shutil.rmtree(post_directory)

        create_post_folder(post_directory, post_markdown_file)

    meta_title = extract_title(body)
    meta_last_mod = now_utc.strftime("%Y-%m-%dT%H:%M:%S")

    meta_data = build_meta_post_section(meta_date, meta_last_mod, meta_tags, meta_title)

    with open(post_markdown_file, "w") as f:
        f.write(replace_math_delimiters(meta_data + body))

    # Save images and resources
    for filename, content in resources["outputs"].items():
        content_filename = os.path.join(post_directory, filename)
        with open(content_filename, "wb") as f:
            f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument(
        "--notebook",
        type=str,
        help="Specify the notebook name",
        required=True,
    )
    parser.add_argument("--tags", nargs="+", help="List of tags", required=False)

    args = parser.parse_args()

    main(args.notebook, args.tags)
