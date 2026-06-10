# jackbmontgomery.com

Personal website and blog. Built with [Marmite](https://marmite.blog), deployed to
[Cloudflare Pages](https://pages.cloudflare.com).

## Stack

| Tool                            | Role                                          |
| ------------------------------- | --------------------------------------------- |
| [Marmite](https://marmite.blog) | Static site generator — Markdown in, HTML out |
| Cloudflare Pages                | Hosting                                       |
| GitHub Actions                  | CI/CD — builds and deploys on push to `main`  |

## Project Structure

```
content/        # Markdown source files (posts, pages, hero text)
site/           # Generated output (do not edit directly)
custom.css      # Style overrides
custom.js       # JS overrides
marmite.yaml    # Site config (title, nav, colorscheme, etc.)
favicon.ico     # Site icon
```

## Local Development

Install Marmite:

```bash
curl -sS https://marmite.blog/install.sh | sh
```

Build and serve:

```bash
marmite . --serve
```

## Writing

Add a new post as a Markdown file by running:

```
marmite . --new TITLE -t TAGS
```

This will make a new Markdown file in the content directory, with the default date time format in
the name of the file.

Frontmatter tags and metadata are optional — Marmite infers date and slug from the filename.

## Deployment

Push to `main`. GitHub Actions installs Marmite, runs `marmite . --debug` to build into `site/`,
then deploys via `wrangler pages deploy`.

Required secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`.
