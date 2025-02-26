import nbformat
from nbconvert import MarkdownExporter
import os
import re
from datetime import datetime
from datetime import timezone
import json
import argparse
import shutil


def build_meta_post_section(date, meta_last_mod, tags, title):
    return f'---\ndate: {date}\nlastmod: {meta_last_mod}\nshowTableOfContents: true\ntags: {tags}\ntitle: "{title}"\ntype: "post"\n---\n'


def replace_math_delimiters(text):
    return re.sub(
        r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", r"\\(\1\\)", text, flags=re.DOTALL
    )


# def wrap_code_blocks(text):
#     # Regex pattern to find code blocks enclosed by ```
#     pattern = re.compile(r"```(.*?)```", re.DOTALL)

#     # Replace code blocks with wrapped Hugo details shortcode
#     def replacer(match):
#         code_content = match.group(1).strip()
#         return f'{{{{< details title="Code" >}}}}\n```{code_content}\n```\n{{{{< /details >}}}}'

#     return pattern.sub(replacer, text)


def remove_ignored_code_blocks(text):
    # Regex pattern to match code blocks starting with ```\n#IGNORE and remove them
    pattern = re.compile(r"```python\n#IGNORE.*?```", re.DOTALL)

    # Replace matched code blocks with an empty string
    return pattern.sub("", text)


def remove_pre_blocks(text):
    # Remove content inside <pre> tags
    return re.sub(r"<pre.*?</pre>", "", text, flags=re.DOTALL)


def extract_and_copy_video(raw_markdown, destination_directory):
    # Regex pattern to extract the video file path from the src attribute
    video_path_pattern = r'<video\s+.*?src="([^"]+)"'
    match = re.search(video_path_pattern, raw_markdown)

    if not match:
        print("No video source found in the provided HTML snippet.")
        return

    video_path = match.group(1)

    # Check if the video file exists
    if not os.path.exists(f"./notebooks/{video_path}"):
        raise FileNotFoundError(
            f"The video file ./notebooks/{video_path} does not exist."
        )

    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # Get the video file name and construct the destination path
    video_filename = os.path.basename(video_path)
    destination_path = os.path.join(destination_directory, video_filename)

    # Copy the video file to the destination directory
    shutil.copy(video_path, destination_path)

    print(
        f"Video file '{video_filename}' successfully copied to '{destination_directory}'."
    )
    return destination_path


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
        # Process and clean up the markdown content
        text = replace_math_delimiters(meta_data + body)
        text = remove_ignored_code_blocks(text)
        text = remove_pre_blocks(text)  # Remove <pre> blocks
        f.write(text)

    # Save images and resources
    for filename, content in resources["outputs"].items():
        content_filename = os.path.join(post_directory, filename)
        with open(content_filename, "wb") as f:
            f.write(content)

    # Copy manim generated video and paste it in the page bundle
    # extract_and_copy_video(text, post_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some inputs.")

    parser.add_argument(
        "--notebook",
        type=str,
        help="Specify the notebook name",
        required=True,
        default="chaos-lorenz-system",
    )
    parser.add_argument("--tags", nargs="+", help="List of tags", required=False)

    args = parser.parse_args()

    main(args.notebook, args.tags)
