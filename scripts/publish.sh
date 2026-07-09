#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/publish.sh VERSION [--yes] [options]

Options:
  --yes                 Run the release. Without this, only print the plan.
  --remote NAME         Git remote to push to. Default: origin.
  --branch NAME         Branch to push. Default: current branch.
  --skip-git            Do not commit, tag, or push to GitHub.
  --skip-pypi           Do not build or upload to PyPI.
  --repository-url URL  Upload to a custom PyPI endpoint, such as TestPyPI.
  DIST_DIR=PATH         Optional env var for package output. Default: temp dir.
  -h, --help            Show this help.

Examples:
  scripts/publish.sh 1.1.1
  scripts/publish.sh 1.1.1 --yes
  scripts/publish.sh 1.1.1 --yes --repository-url https://test.pypi.org/legacy/
EOF
}

VERSION=""
YES=0
REMOTE="origin"
BRANCH=""
SKIP_GIT=0
SKIP_PYPI=0
REPOSITORY_URL=""
BUILD_DIST_DIR="${DIST_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes)
      YES=1
      shift
      ;;
    --remote)
      REMOTE="${2:?missing remote name}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:?missing branch name}"
      shift 2
      ;;
    --skip-git)
      SKIP_GIT=1
      shift
      ;;
    --skip-pypi)
      SKIP_PYPI=1
      shift
      ;;
    --repository-url)
      REPOSITORY_URL="${2:?missing repository URL}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
    *)
      if [[ -n "$VERSION" ]]; then
        echo "Unexpected argument: $1" >&2
        usage
        exit 2
      fi
      VERSION="$1"
      shift
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  usage
  exit 2
fi

if [[ ! "$VERSION" =~ ^[0-9]+(\.[0-9]+){2}([a-zA-Z0-9._+-]+)?$ ]]; then
  echo "Version should look like 1.1.1 or 1.1.1rc1: $VERSION" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
TAG="v$VERSION"

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

need_module() {
  "$PYTHON" - "$@" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    print("Missing Python module(s): " + ", ".join(missing), file=sys.stderr)
    print("Install them with: python -m pip install build twine", file=sys.stderr)
    raise SystemExit(1)
PY
}

sync_version() {
  "$PYTHON" - "$VERSION" <<'PY'
from pathlib import Path
import re
import sys

version = sys.argv[1]
updates = {
    Path("pyproject.toml"): (r'(?m)^version = "[^"]+"', f'version = "{version}"'),
    Path("docs/source/conf.py"): (r"(?m)^release = '[^']+'", f"release = '{version}'"),
}

for path, (pattern, replacement) in updates.items():
    text = path.read_text()
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"Could not update version in {path}")
    path.write_text(new_text)
PY
}

build_package() {
  need_module build twine
  if [[ -z "$BUILD_DIST_DIR" ]]; then
    BUILD_DIST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cisformer-dist.XXXXXX")"
  else
    run rm -rf "$BUILD_DIST_DIR"
    run mkdir -p "$BUILD_DIST_DIR"
  fi
  run "$PYTHON" -m build --outdir "$BUILD_DIST_DIR"
  run "$PYTHON" -m twine check "$BUILD_DIST_DIR"/*
}

upload_package() {
  if [[ -n "$REPOSITORY_URL" ]]; then
    run "$PYTHON" -m twine upload --repository-url "$REPOSITORY_URL" "$BUILD_DIST_DIR"/*
  else
    run "$PYTHON" -m twine upload "$BUILD_DIST_DIR"/*
  fi
}

if [[ "$YES" -ne 1 ]]; then
  git_step="commit the version bump, create tag $TAG, and push to GitHub"
  pypi_step="build sdist and wheel, run twine check, and upload package files to PyPI"
  [[ "$SKIP_GIT" -eq 1 ]] && git_step="skip GitHub commit, tag, and push"
  [[ "$SKIP_PYPI" -eq 1 ]] && pypi_step="skip build and PyPI upload"
  cat <<EOF
Dry run. Re-run with --yes to publish.

Will release $TAG from:
  $ROOT_DIR

Will do:
  - update pyproject.toml and docs/source/conf.py to $VERSION
  - $pypi_step
  - $git_step

Options:
  remote: $REMOTE
  branch: ${BRANCH:-current branch}
  skip git: $SKIP_GIT
  skip PyPI: $SKIP_PYPI
  repository URL: ${REPOSITORY_URL:-default PyPI}
  package output: ${DIST_DIR:-temporary directory}
EOF
  exit 0
fi

if [[ "$SKIP_GIT" -ne 1 ]]; then
  command -v git >/dev/null || { echo "Missing git" >&2; exit 1; }
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Not inside a git repository. Use --skip-git to publish only to PyPI." >&2
    exit 1
  }
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Working tree is not clean. Commit or stash changes before publishing." >&2
    exit 1
  fi
  if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "Tag already exists: $TAG" >&2
    exit 1
  fi
  if [[ -z "$BRANCH" ]]; then
    BRANCH="$(git branch --show-current)"
  fi
  if [[ -z "$BRANCH" ]]; then
    echo "Could not detect current branch. Pass --branch NAME." >&2
    exit 1
  fi
fi

sync_version

if [[ "$SKIP_PYPI" -ne 1 ]]; then
  build_package
fi

if [[ "$SKIP_GIT" -ne 1 ]]; then
  if ! git diff --quiet -- pyproject.toml docs/source/conf.py; then
    run git add pyproject.toml docs/source/conf.py
    run git commit -m "Release $TAG"
  else
    echo "Version files already at $VERSION; no release commit created."
  fi
  run git tag -a "$TAG" -m "Release $TAG"
  run git push "$REMOTE" "$BRANCH"
  run git push "$REMOTE" "$TAG"
fi

if [[ "$SKIP_PYPI" -ne 1 ]]; then
  upload_package
fi
