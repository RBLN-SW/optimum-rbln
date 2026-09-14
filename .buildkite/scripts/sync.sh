#!/usr/bin/env bash
# uv sync + rebel-compiler, shared by the pytest suites and the BC steps.
#
# Pods share UV_CACHE_DIR, so a step often reuses an editable build another pod
# made. That is what we want -- rebuilding it costs ~18s per step and the install
# is editable, so the code always comes from this checkout -- but only the pod
# that built it has the gitignored src/optimum/rbln/__version__.py the hatch-vcs
# hook writes. The others write it here from the installed metadata; the code
# reads nothing but __version__ from it.
set -euo pipefail

echo '--- :package: uv sync'
uv sync --locked --python 3.12 --group tests
if [ ! -f src/optimum/rbln/__version__.py ]; then
  uv run --no-sync python -c "import importlib.metadata as m, pathlib; pathlib.Path('src/optimum/rbln/__version__.py').write_text('__version__ = version = %r\n' % m.version('optimum-rbln'))"
fi
bash .buildkite/scripts/install-rebel-compiler.sh
