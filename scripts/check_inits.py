#!/usr/bin/env python
# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Check that every package ``__init__.py`` under ``src/optimum/rbln`` re-exports exactly the
children that declare ``__all__``.

Public objects are registered by listing them in the ``__all__`` of the module that defines
them. ``define_import_structure`` discovers them at runtime, and the ``if TYPE_CHECKING:``
block of each package ``__init__.py`` mirrors the same set for type checkers and IDEs::

    if TYPE_CHECKING:
        from .configuration_llama import *
        from .modeling_llama import *

Run ``python scripts/check_inits.py`` to verify the blocks and ``--fix`` to rewrite them.
"""

import argparse
import ast
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "src" / "optimum" / "rbln"
IGNORED_MODULES = {"__version__"}


def rel(path: Path) -> Path:
    return path.relative_to(REPO)


def module_all(path: Path) -> list[str] | None:
    """Return the names listed in the module's ``__all__``, or None when it declares none."""
    source = path.read_text()
    tree = ast.parse(source)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    ]
    if not assignments:
        return None
    node = assignments[-1]
    if not isinstance(node.value, ast.List) or not all(
        isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.value.elts
    ):
        raise SystemExit(f"{rel(path)}: __all__ must be a list literal of string constants")
    lines = source.splitlines()[node.lineno - 1 : node.end_lineno]
    if len(lines) > 1 and (len(lines) != len(node.value.elts) + 2 or lines[-1].strip() != "]"):
        raise SystemExit(f"{rel(path)}: a multi-line __all__ must hold one name per line and close with a bare ']'")
    return [e.value for e in node.value.elts]


def defined_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name for a in node.names)
    return names


def exporting_children(package: Path, errors: list[str]) -> list[str]:
    """Names of the direct child modules and subpackages of ``package`` that export something."""
    children = []
    for entry in sorted(package.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists():
            if exporting_children(entry, errors):
                children.append(entry.name)
        elif entry.suffix == ".py" and entry.name != "__init__.py" and entry.stem not in IGNORED_MODULES:
            names = module_all(entry)
            if names is None:
                continue
            missing = set(names) - defined_names(entry)
            if missing:
                errors.append(f"{rel(entry)}: __all__ lists undefined names {sorted(missing)}")
            if names:
                children.append(entry.stem)
    return children


def type_checking_block(source: str) -> tuple[int, int] | None:
    lines = source.splitlines()
    try:
        start = lines.index("if TYPE_CHECKING:")
    except ValueError:
        return None
    end = start + 1
    while end < len(lines) and lines[end].startswith("    "):
        end += 1
    return start + 1, end


def check_package(package: Path, fix: bool, errors: list[str]) -> None:
    for entry in sorted(package.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists():
            check_package(entry, fix, errors)

    children = exporting_children(package, errors)
    if not children:
        return
    init = package / "__init__.py"
    source = init.read_text()
    block = type_checking_block(source)
    if block is None or "define_import_structure(" not in source:
        errors.append(
            f"{rel(init)}: packages that export objects must use the lazy `define_import_structure` template"
        )
        return
    expected = [f"    from .{child} import *" for child in children]
    lines = source.splitlines()
    actual = lines[block[0] : block[1]]
    if actual == expected:
        return
    if fix:
        init.write_text("\n".join(lines[: block[0]] + expected + lines[block[1] :]) + "\n")
        print(f"fixed {rel(init)}")
    else:
        errors.append(f"{rel(init)}: TYPE_CHECKING imports are out of date, run `python scripts/check_inits.py --fix`")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="rewrite the TYPE_CHECKING blocks in place")
    args = parser.parse_args()

    errors: list[str] = []
    check_package(ROOT, args.fix, errors)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
