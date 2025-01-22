import nbformat
from nbconvert import MarkdownExporter
import os

meta_data = """---
date: 2025-01-22T10:36:50+02:00
# description: ""
# image: ""
lastmod: 2025-01-22
showTableOfContents: false
tags: ["physics", "neural-networks"]
title: "Solving ODE's with Physics Informed Neural Networks v2"
type: "post"
---
"""


def main(note_book_name="solving-odes-with-neural-networks"):
    post_directory = f"./content/posts/{note_book_name}"
    post_markdown_file = f"{post_directory}/index.md"

    with open(f"notebooks/{note_book_name}.ipynb") as f:
        notebook = nbformat.read(f, as_version=4)

    markdown_exporter = MarkdownExporter()
    (body, resources) = markdown_exporter.from_notebook_node(notebook)

    if not os.path.exists(post_directory):
        os.mkdir(post_directory)
        with open(post_markdown_file, "w"):
            pass

    with open(post_markdown_file, "w") as f:
        f.write(meta_data + body)

    # Save images and resources
    for filename, content in resources["outputs"].items():
        content_filename = os.path.join(post_directory, filename)
        with open(content_filename, "wb") as f:
            f.write(content)


if __name__ == "__main__":
    main()


# ---
# date: 2025-01-22T18:50:35+02:00
# # description: ""
# # image: ""
# lastmod: 2025-01-22
# showTableOfContents: false
# # tags: ["",]
# title: "Something"
# type: "post"
# ---
