# Publishing to GitHub and PyPI

This page is for maintainers who publish Cisformer releases.

## Prerequisites

- Work from a clean git checkout with permission to push to GitHub.
- Install packaging tools in the active Python environment:

```bash
python -m pip install build twine
```

- Configure a PyPI token. The simplest non-interactive setup is:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-token-here
```

- Update release notes before publishing if the release has user-visible changes.

## Dry Run

The script is safe by default. Without `--yes`, it only prints the actions it
would run:

```bash
scripts/publish.sh 1.1.1
```

## Publish

Run this from the project root:

```bash
scripts/publish.sh 1.1.1 --yes
```

The script will:

1. Update `pyproject.toml` and `docs/source/conf.py` to the requested version.
2. Build the source distribution and wheel.
3. Run `twine check` on the generated files.
4. Commit the version bump if needed.
5. Create and push the `v<version>` git tag to GitHub.
6. Upload the built package files to PyPI.

By default, package files are built in a temporary directory so old files under
`dist/` are not uploaded by accident. To keep the generated package files in the
project directory, set `DIST_DIR`:

```bash
DIST_DIR=dist scripts/publish.sh 1.1.1 --yes
```

PyPI does not allow reusing a published version number. If upload fails because
the version already exists, bump the version and rerun the release.

## TestPyPI

To test the package upload without touching the main PyPI project:

```bash
scripts/publish.sh 1.1.1 --yes --repository-url https://test.pypi.org/legacy/
```

Use a TestPyPI token in `TWINE_PASSWORD` for this command.

## Useful Options

- `--remote upstream`: push to another git remote instead of `origin`.
- `--branch main`: push a specific branch instead of the current branch.
- `--skip-git`: build and upload to PyPI without committing, tagging, or pushing.
- `--skip-pypi`: update GitHub only, without building or uploading the package.
