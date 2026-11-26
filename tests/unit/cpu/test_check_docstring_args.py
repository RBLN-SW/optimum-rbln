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

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_docstring_args.py"

spec = importlib.util.spec_from_file_location("check_docstring_args", SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

BLANK_LINE_ABOVE = "`Args:` needs a blank line above it, or the reference docs skip the section"
BLANK_LINE_BELOW = (
    "`Args:` must be followed by its entries without a blank line, or the reference docs skip the section"
)


def function_source(signature, entries, header="Args"):
    body = "\n".join(f"        {entry}: described" for entry in entries)
    return f'def f({signature}):\n    """\n    {header}:\n{body}\n    """\n'


def messages(source):
    return [violation.message for violation in checker.find_violations(source)]


def run_cli(*paths):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, paths)], capture_output=True, text=True, check=False)


@pytest.mark.parametrize(
    ("signature", "entries"),
    [
        ("model, config, device", ["model", "config", "device"]),
        ("model, config, device", ["model", "device"]),
        ("model, config, device", ["config"]),
        ("model, *args, config, **kwargs", ["model", "*args", "config", "**kwargs"]),
        ("model, *args, config, **kwargs", ["model", "args", "config", "kwargs"]),
        ("model, /, config, *, device", ["model", "config", "device"]),
        ("self, model, config", ["model", "config"]),
        ("cls, model", ["model"]),
        ("model, config", ["model", "config", "model"]),
        ("model, config", []),
    ],
)
def test_entries_that_follow_the_signature_pass(signature, entries):
    assert messages(function_source(signature, entries)) == []


@pytest.mark.parametrize(
    ("signature", "entries", "message"),
    [
        (
            "model, config, device",
            ["config", "model", "device"],
            "`config` is documented before `model`, but the signature declares `model` before `config`\n"
            "    expected order: model, config, device",
        ),
        (
            "model, config, device",
            ["device", "model"],
            "`device` is documented before `model`, but the signature declares `model` before `device`\n"
            "    expected order: model, device",
        ),
        (
            "model, *args, config, **kwargs",
            ["**kwargs", "model"],
            "`kwargs` is documented before `model`, but the signature declares `model` before `kwargs`\n"
            "    expected order: model, kwargs",
        ),
        (
            "model, *args, config, **kwargs",
            ["config", "*args"],
            "`config` is documented before `args`, but the signature declares `args` before `config`\n"
            "    expected order: args, config",
        ),
        (
            "model, /, config, *, device",
            ["device", "config"],
            "`device` is documented before `config`, but the signature declares `config` before `device`\n"
            "    expected order: config, device",
        ),
        (
            "model, config",
            ["config", "model", "config"],
            "`config` is documented before `model`, but the signature declares `model` before `config`\n"
            "    expected order: model, config",
        ),
    ],
)
def test_entries_out_of_signature_order_are_reported(signature, entries, message):
    assert messages(function_source(signature, entries)) == [message]


@pytest.mark.parametrize(
    ("signature", "entries", "message"),
    [
        (
            "model, config",
            ["model", "stale", "config"],
            "documents arguments the signature does not have: `stale`\n    parameters: model, config",
        ),
        (
            "model, config",
            ["old_a", "model", "old_b"],
            "documents arguments the signature does not have: `old_a`, `old_b`\n    parameters: model, config",
        ),
        (
            "model, config",
            ["stale", "stale"],
            "documents arguments the signature does not have: `stale`\n    parameters: model, config",
        ),
        (
            "model",
            ["model", "**kwargs"],
            "documents arguments the signature does not have: `kwargs`\n    parameters: model",
        ),
        ("", ["model"], "documents arguments the signature does not have: `model`\n    parameters: none"),
        (
            "self, model",
            ["self", "model"],
            "documents arguments the signature does not have: `self`\n    parameters: model",
        ),
        ("cls, model", ["cls"], "documents arguments the signature does not have: `cls`\n    parameters: model"),
    ],
)
def test_entries_naming_no_parameter_are_reported(signature, entries, message):
    assert messages(function_source(signature, entries)) == [message]


def test_unknown_names_do_not_hide_an_order_violation():
    assert messages(function_source("model, config", ["config", "stale", "model"])) == [
        "documents arguments the signature does not have: `stale`\n    parameters: model, config",
        "`config` is documented before `model`, but the signature declares `model` before `config`\n"
        "    expected order: model, config",
    ]


@pytest.mark.parametrize("header", ["Args", "Arguments", "Params", "Parameters", "args"])
def test_every_arguments_section_title_is_checked(header):
    assert len(messages(function_source("model, config", ["config", "model"], header=header))) == 1


def test_violation_carries_the_definition_line_not_the_decorator_line():
    source = '@decorator\ndef f(model, config):\n    """\n    Args:\n        config: x\n        model: x\n    """\n'
    assert [(violation.lineno, violation.name) for violation in checker.find_violations(source)] == [(2, "f")]


def test_class_docstring_is_compared_with_init():
    source = textwrap.dedent(
        '''
        class Wrapper:
            """
            Args:
                config: x
                model: x
            """

            def __init__(self, model, config):
                pass
        '''
    )
    assert [(v.lineno, v.name, v.message) for v in checker.find_violations(source)] == [
        (
            2,
            "Wrapper",
            "`config` is documented before `model`, but the signature declares `model` before `config`\n"
            "    expected order: model, config",
        )
    ]


def test_class_without_init_is_skipped():
    source = textwrap.dedent(
        '''
        class Wrapper:
            """
            Args:
                stale: x
                config: x
                model: x
            """

            model: int
            config: int
        '''
    )
    assert messages(source) == []


def test_the_last_init_in_a_class_body_is_the_signature():
    source = textwrap.dedent(
        '''
        class Wrapper:
            """
            Args:
                config: x
                model: x
            """

            def __init__(self, config, model): ...

            def __init__(self, model, config): ...
        '''
    )
    assert [violation.name for violation in checker.find_violations(source)] == ["Wrapper"]


def test_methods_and_nested_definitions_are_reported_by_qualified_name_in_line_order():
    source = textwrap.dedent(
        '''
        class Outer:
            class Inner:
                async def method(self, model, config):
                    """
                    Args:
                        config: x
                        model: x
                    """


        def outer(model, config):
            if True:

                def inner(model, config):
                    """
                    Args:
                        config: x
                        model: x
                    """
        '''
    )
    assert [violation.name for violation in checker.find_violations(source)] == ["Outer.Inner.method", "outer.inner"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('def f(model):\n    """\n    Summary.\n    Args:\n        model: x\n    """\n', BLANK_LINE_ABOVE),
        ('def f(model):\n    """\n    Args:\n\n        model: x\n    """\n', BLANK_LINE_BELOW),
        ('class C:\n    """\n    Summary.\n    Args:\n        model: x\n    """\n', BLANK_LINE_ABOVE),
    ],
)
def test_a_header_the_reference_docs_skip_is_reported(source, message):
    assert messages(source) == [message]


@pytest.mark.parametrize(
    "source",
    [
        'def f(model):\n    """\n    Args:\n        model: x\n    """\n',
        'def f(model):\n    """\n    Summary.\n\n    Args:\n        model: x\n    """\n',
        'def f(model):\n    """\n    Summary.\n    Args:\n    """\n',
        'def f(model):\n    """\n    Summary.\n    Args:\n    More prose.\n    """\n',
    ],
)
def test_a_header_the_reference_docs_read_is_not_reported(source):
    assert messages(source) == []


def test_a_whitespace_only_line_below_the_header_is_not_an_entry():
    source = 'def f(model):\n    """\n    Summary.\n    Args:\n        \n    More prose.\n    """\n'
    assert messages(source) == []


def test_the_header_is_reported_with_the_title_as_written():
    source = 'def f(model):\n    """\n    Summary.\n    Parameters:\n        model: x\n    """\n'
    assert messages(source) == ["`Parameters:` needs a blank line above it, or the reference docs skip the section"]


@pytest.mark.parametrize(
    ("entry", "shown"),
    [
        ("model", "model"),
        ("model : int", "model : int"),
        ("model (int) : described", "model (int) : described"),
    ],
)
def test_an_entry_the_reference_docs_cannot_read_is_reported_as_written(entry, shown):
    source = f'def f(model):\n    """\n    Args:\n        {entry}\n    """\n'
    assert messages(source) == [
        f"the reference docs cannot read the entry `{shown}`, write it as `name: description` "
        "or `name (type): description`"
    ]


def test_an_unreadable_entry_is_reported_even_when_the_signature_is_unknown():
    source = 'class Wrapper:\n    """\n    Args:\n        model\n    """\n'
    assert messages(source) == [
        "the reference docs cannot read the entry `model`, write it as `name: description` "
        "or `name (type): description`"
    ]


def test_an_unreadable_entry_does_not_hide_the_readable_ones_from_the_checks():
    source = 'def f(model, config):\n    """\n    Args:\n        config: x\n        stale\n        model: x\n    """\n'
    assert messages(source) == [
        "the reference docs cannot read the entry `stale`, write it as `name: description` "
        "or `name (type): description`",
        "`config` is documented before `model`, but the signature declares `model` before `config`\n"
        "    expected order: model, config",
    ]


def test_a_skipped_header_does_not_stop_the_entries_from_being_checked():
    source = 'def f(model):\n    """\n    Summary.\n    Args:\n        stale: x\n    """\n'
    assert messages(source) == [
        BLANK_LINE_ABOVE,
        "documents arguments the signature does not have: `stale`\n    parameters: model",
    ]


def test_args_sections_read_every_entry_style_the_reference_docs_read():
    docstring = textwrap.dedent(
        """
        Args:
            plain: described
            typed (int): described
            tight(int): described
            backticked (`str | None`, *optional*, defaults to `"x"`):
                described on the next line
            *args: described
            **kwargs (dict): described
        """
    )
    [section] = checker.args_sections(docstring)
    assert (section.names, section.unreadable) == (["plain", "typed", "tight", "backticked", "args", "kwargs"], [])


def test_args_sections_set_aside_the_entries_the_reference_docs_cannot_read():
    docstring = textwrap.dedent(
        """
        Args:
            readable: described
            no_colon
            numpy_style : int
                described
            spaced_type (int) : described
            See the docs: prose that looks like an entry
            (int): a type without a name
        """
    )
    [section] = checker.args_sections(docstring)
    assert section.names == ["readable"]
    assert section.unreadable == [
        "no_colon",
        "numpy_style : int",
        "spaced_type (int) : described",
        "See the docs: prose that looks like an entry",
        "(int): a type without a name",
    ]


def test_args_sections_skip_continuation_lines_and_other_sections():
    docstring = textwrap.dedent(
        """
        Summary.

        Args:
            model (int): first
                continued: with a colon
                - "static": a bullet
            **kwargs: rest

        Returns:
            value: not an argument
        """
    )
    assert [section.names for section in checker.args_sections(docstring)] == [["model", "kwargs"]]


def test_args_sections_are_returned_separately_with_their_headers():
    docstring = textwrap.dedent(
        """
        Args:
            model: x

        Parameters:
            config: x
        """
    )
    assert [(section.header, section.names, section.defect) for section in checker.args_sections(docstring)] == [
        ("Args:", ["model"], None),
        ("Parameters:", ["config"], None),
    ]


def test_default_root_is_the_package_source():
    assert (checker.ROOT / "__init__.py").is_file()


def test_python_files_are_listed_in_sorted_order(tmp_path):
    names = ["pkg/z.py", "a.py", "pkg/m.py", "pkg/sub/b.py", "c.py", "pkg/a.py", "y.py", "pkg/sub/a.py"]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert checker.python_files([tmp_path]) == sorted(tmp_path / name for name in names)


def test_python_files_lists_each_file_once(tmp_path):
    (tmp_path / "a.py").write_text("")
    assert checker.python_files([tmp_path, tmp_path / "a.py"]) == [tmp_path / "a.py"]


def test_cli_passes_a_clean_tree_silently(tmp_path):
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "clean.py").write_text(function_source("model, config", ["model", "config"]))
    result = run_cli(tmp_path)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_cli_fails_with_the_location_the_offending_pair_and_the_expected_order(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(function_source("model, config, device", ["config", "model", "device"]))
    result = run_cli(module)
    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        f"{module}:1: f: `config` is documented before `model`, but the signature declares `model` before `config`",
        "    expected order: model, config, device",
    ]


def test_cli_fails_with_the_unknown_name_and_the_parameters_of_the_signature(tmp_path):
    module = tmp_path / "module.py"
    module.write_text(function_source("model, config", ["model", "stale"]))
    result = run_cli(module)
    assert result.returncode == 1
    assert result.stderr.splitlines() == [
        f"{module}:1: f: documents arguments the signature does not have: `stale`",
        "    parameters: model, config",
    ]


def test_cli_reports_every_file_in_path_order(tmp_path):
    names = ["e.py", "b.py", "pkg/d.py", "a.py", "pkg/c.py", "f.py"]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(function_source("model, config", ["config", "model"]))
    result = run_cli(tmp_path)
    locations = [line.split(":")[0] for line in result.stderr.splitlines() if line.startswith(str(tmp_path))]
    assert (result.returncode, locations) == (1, sorted(str(tmp_path / name) for name in names))


def test_cli_reports_a_file_that_does_not_parse(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def f(:\n")
    result = run_cli(broken)
    assert result.returncode == 1
    assert result.stderr.startswith(f"{broken}:1: cannot parse")


def test_cli_rejects_a_path_that_does_not_exist(tmp_path):
    result = run_cli(tmp_path / "missing.py")
    assert result.returncode == 2
    assert "no such file or directory" in result.stderr
