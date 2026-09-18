# How to contribute to Optimum RBLN

Optimum RBLN is an open source project, so all contributions and suggestions are welcome.

You can contribute in many different ways: giving ideas, answering questions, reporting bugs, proposing enhancements, improving the documentation, fixing bugs, and adding support for new models.

Many thanks in advance to every contributor.

## Asking questions and reporting issues

> If you want to ask a question, we assume that you have read the available [documentation](https://docs.rbln.ai/software/optimum/optimum_rbln.html).

Before opening a new issue, search the existing [issues](https://github.com/rbln-sw/optimum-rbln/issues) first. If none of them covers your case, open a [new issue](https://github.com/rbln-sw/optimum-rbln/issues/new/choose) using one of the templates and provide as much context as you can (model id, `rbln_config`, versions of `optimum-rbln` and `rebel-compiler`, full traceback).

## Development setup

### Prerequisites

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/), version 0.11.25 or newer
- `rebel-compiler`, which is available to approved users only. Follow the [installation guide](https://docs.rbln.ai/getting_started/installation_guide.html) to install it into the virtual environment created below. Everything except compiling and running on an NPU works without it.

### Clone and install

Fork the [repository](https://github.com/rbln-sw/optimum-rbln), clone your fork and add the base repository as a remote:

```bash
git clone git@github.com:<your GitHub handle>/optimum-rbln.git
cd optimum-rbln
git remote add upstream https://github.com/rbln-sw/optimum-rbln.git
```

Create the virtual environment and install the project in editable mode together with every dependency group (`tests`, `quality`, `deploy`):

```bash
uv sync --all-groups
```

`uv sync` creates `.venv`, installs the locked dependencies from `uv.lock` and installs `optimum-rbln` itself in editable mode. Run tools through `uv run <command>` or activate the environment with `source .venv/bin/activate`.

The package version is derived from git tags by `hatch-vcs` and written to `src/optimum/rbln/__version__.py` when the project is installed. After pulling new commits or tags, refresh it with `uv sync --reinstall-package optimum-rbln`.

### Pre-commit hooks

Install the git hooks once after cloning:

```bash
uv run pre-commit install
```

The hooks run `ruff check --fix`, `ruff format`, a `uv.lock` consistency check and a few file hygiene checks on every commit. To run them over the whole repository:

```bash
uv run pre-commit run --all-files
```

## Making changes

1. Create a branch from `dev`. Do not work on `dev` or `main` directly.

   ```bash
   git checkout -b a-descriptive-name-for-my-changes upstream/dev
   ```

2. Develop your change. Keep upstream `dev` merged in regularly so the pull request stays easy to review:

   ```bash
   git fetch upstream
   git rebase upstream/dev
   ```

3. Format and lint your code. If you installed the pre-commit hooks this happens automatically on commit; otherwise run:

   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   ```

4. Run the tests relevant to your change (see below), commit and push to your fork:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

5. Open a pull request against the `dev` branch. Only critical hotfixes target `main`, and those must be merged into `dev` as well.

### Pull request titles

Pull request titles follow the conventional commit format and are checked by CI:

```
type(optional scope): description
```

| Type          | Use it for                                                        |
| ------------- | ----------------------------------------------------------------- |
| `model`       | Adding a new model or fixing an existing one                      |
| `performance` | Making a model or the library itself faster or lighter            |
| `refactor`    | Re-arranging code without changing behavior                       |
| `doc`         | Docstring or documentation changes                                |
| `dependency`  | Dependency and lockfile updates                                   |
| `release`     | Merging `dev` into `main` for a release                            |
| `other`       | Anything else, such as CI or tooling changes                      |

## Tests

Tests live in `tests/` and are run with `pytest`. Most of them compile a model and execute it on an RBLN NPU, so they need `rebel-compiler` and a device. Run a single suite with:

```bash
uv run pytest tests/test_config.py -v
```

The CI splits the suites into `test_config.py`, `test_transformers.py`, `test_diffusers.py` and `test_llm.py`; the helper scripts it uses are in `scripts/`.

Public classes must have docstrings that `mkdocstrings` can render, because the reference documentation is generated from them. Check the files you changed with:

```bash
bash scripts/check-docstrings.sh
```

## Dependencies

Runtime dependencies are declared in `pyproject.toml` and locked in `uv.lock`, which is committed. Change them with `uv add` / `uv remove` (use `--group tests` or `--group quality` for development-only tools) and commit the updated lockfile together with `pyproject.toml`. Dependency updates for `transformers` and `diffusers` are opened automatically by Renovate.

## Code of conduct

This project adheres to the Rebellions [code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.
