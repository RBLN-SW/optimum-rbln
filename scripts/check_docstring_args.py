#!/usr/bin/env python
# Copyright 2026 Rebellions Inc. All rights reserved.

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
Check that the ``Args:`` section of every docstring agrees with the signature it documents.

A function docstring is compared with the parameters of the function and a class docstring with the
parameters of its ``__init__``. The reference documentation reads an entry as ``name: description`` or
``name (type): description``, and these are reported:

- an entry written in another way, which the reference documentation cannot read,
- an entry that names no parameter,
- entries that are written in a different order than the signature declares the parameters,
- an ``Args:`` header the reference documentation skips, because there is no blank line above it or a blank
  line between it and its entries.

Parameters that are not documented are left alone, and so is a class that defines no ``__init__``.

Run ``python scripts/check_docstring_args.py`` to check ``src/optimum/rbln``, or pass files and
directories to check only those.
"""

import argparse
import ast
import inspect
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "src" / "optimum" / "rbln"
ARGS_HEADER = re.compile(r"(?:Args|Arguments|Params|Parameters):\s*", re.IGNORECASE)
ENTRY_NAME = re.compile(r"\*{0,2}(\w+)")
Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class Section(NamedTuple):
    header: str
    names: list[str]
    unreadable: list[str]
    defect: str | None


class Violation(NamedTuple):
    lineno: int
    name: str
    message: str


def rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO)
    except ValueError:
        return path


def indentation(line: str) -> int:
    return len(line) - len(line.lstrip())


def indented(line: str) -> bool:
    return line.startswith(" ") and bool(line.strip())


def header_defect(lines: list[str], index: int) -> str | None:
    """Return why the reference docs skip the header at ``lines[index]``, or None when they read it."""
    below = [*lines[index + 1 : index + 3], "", ""]
    entries_follow = indented(below[0])
    entries_follow_blank_line = not below[0].strip() and indented(below[1])
    if not (entries_follow or entries_follow_blank_line):
        return None
    if index > 0 and lines[index - 1].strip():
        return "needs a blank line above it"
    if entries_follow_blank_line:
        return "must be followed by its entries without a blank line"
    return None


def entry_name(entry: str) -> str | None:
    """Return the parameter name the reference docs read from ``entry``, or None when they cannot read one."""
    head, colon, _ = entry.partition(":")
    if "(" in head and head.endswith(")"):
        head = head.split("(", 1)[0].strip()
    name = ENTRY_NAME.fullmatch(head)
    return name.group(1) if colon and name else None


def args_sections(docstring: str) -> list[Section]:
    """Return the ``Args`` sections of ``docstring`` with their entries in the order they are written."""
    lines = inspect.cleandoc(docstring).splitlines()
    sections: list[Section] = []
    index = 0
    while index < len(lines):
        if not ARGS_HEADER.fullmatch(lines[index]):
            index += 1
            continue
        header = lines[index].strip()
        defect = header_defect(lines, index)
        names: list[str] = []
        unreadable: list[str] = []
        entry_indent = None
        index += 1
        while index < len(lines) and (not lines[index].strip() or indentation(lines[index]) > 0):
            line = lines[index]
            if line.strip():
                if entry_indent is None:
                    entry_indent = indentation(line)
                if indentation(line) == entry_indent:
                    name = entry_name(line.strip())
                    if name is None:
                        unreadable.append(line.strip())
                    else:
                        names.append(name)
            index += 1
        sections.append(Section(header, names, unreadable, defect))
    return sections


def parameter_names(arguments: ast.arguments) -> list[str]:
    names = [argument.arg for argument in (*arguments.posonlyargs, *arguments.args)]
    if names and names[0] in ("self", "cls"):
        del names[0]
    if arguments.vararg:
        names.append(arguments.vararg.arg)
    names.extend(argument.arg for argument in arguments.kwonlyargs)
    if arguments.kwarg:
        names.append(arguments.kwarg.arg)
    return names


def signature_of(definition: Definition) -> list[str] | None:
    """Return the parameters the docstring of ``definition`` describes, or None for a class without ``__init__``."""
    if not isinstance(definition, ast.ClassDef):
        return parameter_names(definition.args)
    for statement in reversed(definition.body):
        if isinstance(statement, ast.FunctionDef) and statement.name == "__init__":
            return parameter_names(statement.args)
    return None


def definitions(tree: ast.Module) -> Iterator[tuple[str, Definition]]:
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
    while stack:
        node, prefix = stack.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, Definition):
                qualname = prefix + child.name
                yield qualname, child
                stack.append((child, f"{qualname}."))
            else:
                stack.append((child, prefix))


def argument_problems(names: list[str], parameters: list[str]) -> list[str]:
    problems = []
    unknown = [name for name in dict.fromkeys(names) if name not in parameters]
    if unknown:
        problems.append(
            f"documents arguments the signature does not have: {', '.join(f'`{name}`' for name in unknown)}\n"
            f"    parameters: {', '.join(parameters) or 'none'}"
        )
    documented = list(dict.fromkeys(name for name in names if name in parameters))
    expected = [name for name in parameters if name in documented]
    if documented != expected:
        late, early = next((late, early) for late, early in zip(documented, expected, strict=True) if late != early)
        problems.append(
            f"`{late}` is documented before `{early}`, but the signature declares `{early}` before `{late}`\n"
            f"    expected order: {', '.join(expected)}"
        )
    return problems


def find_violations(source: str | bytes) -> list[Violation]:
    violations: list[Violation] = []
    for qualname, definition in definitions(ast.parse(source)):
        docstring = ast.get_docstring(definition, clean=False)
        if docstring is None:
            continue
        parameters = signature_of(definition)
        for section in args_sections(docstring):
            problems = []
            if section.defect:
                problems.append(f"`{section.header}` {section.defect}, or the reference docs skip the section")
            problems.extend(
                f"the reference docs cannot read the entry `{entry}`, write it as `name: description` "
                "or `name (type): description`"
                for entry in section.unreadable
            )
            if parameters is not None:
                problems.extend(argument_problems(section.names, parameters))
            violations.extend(Violation(definition.lineno, qualname, problem) for problem in problems)
    return sorted(violations, key=lambda violation: (violation.lineno, violation.name))


def check_file(path: Path) -> list[str]:
    try:
        violations = find_violations(path.read_bytes())
    except SyntaxError as error:
        return [f"{rel(path)}:{error.lineno}: cannot parse: {error.msg}"]
    return [f"{rel(path)}:{violation.lineno}: {violation.name}: {violation.message}" for violation in violations]


def python_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        files.update(path.rglob("*.py") if path.is_dir() else [path])
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths", nargs="*", type=Path, default=[ROOT], help="files or directories to check (default: src/optimum/rbln)"
    )
    args = parser.parse_args()
    missing = [path for path in args.paths if not path.exists()]
    if missing:
        parser.error(f"no such file or directory: {', '.join(map(str, missing))}")

    errors = [error for path in python_files(args.paths) for error in check_file(path)]
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
