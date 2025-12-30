#!/usr/bin/env python3
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

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from .__version__ import __version__
from .configuration_utils import RBLNModelConfig
from .utils.model_utils import get_rbln_model_cls
from .utils.runtime_utils import ContextRblnConfig


def set_nested_dict(dictionary, key_path, value):
    """
    Set a value in a nested dictionary using dot notation.

    Args:
        dictionary: The dictionary to modify
        key_path: Dot-separated key path (e.g., "unet.batch_size")
        value: The value to set
    """
    keys = key_path.split(".")
    current = dictionary

    # Navigate to the parent of the final key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    current[keys[-1]] = value


def parse_value(value_str):
    """
    Parse a string value to appropriate Python type.

    Args:
        value_str: String value to parse

    Returns:
        Parsed value (bool, int, float, list, dict, or str)
    """
    # First try to parse as JSON (handles dicts, lists, etc.)
    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        pass

    # Handle boolean values
    if value_str.lower() in ["true", "false"]:
        return value_str.lower() == "true"

    # Handle comma-separated values as lists
    if "," in value_str:
        parts = [part.strip() for part in value_str.split(",")]
        # Recursively parse each part
        return [parse_single_value(part) for part in parts]

    # Handle single values
    return parse_single_value(value_str)


def parse_single_value(value_str):
    """
    Parse a single string value to appropriate Python type.

    Args:
        value_str: String value to parse (no commas)

    Returns:
        Parsed value (bool, int, float, or str)
    """
    # Handle boolean values
    if value_str.lower() in ["true", "false"]:
        return value_str.lower() == "true"

    # Handle integer values
    if value_str.isdigit() or (value_str.startswith("-") and value_str[1:].isdigit()):
        return int(value_str)

    # Handle float values
    try:
        return float(value_str)
    except ValueError:
        pass

    # Return as string if all else fails
    return value_str


# ---- Simple ANSI styling helpers for richer CLI output ----
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_UNDERLINE = "\033[4m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_BRIGHT_RED = "\033[91m"
ANSI_BRIGHT_GREEN = "\033[92m"
ANSI_BRIGHT_YELLOW = "\033[93m"
ANSI_BRIGHT_BLUE = "\033[94m"
ANSI_BRIGHT_MAGENTA = "\033[95m"
ANSI_BRIGHT_CYAN = "\033[96m"

STYLES_ENABLED = True


def _color(text: str, color: str) -> str:
    if not STYLES_ENABLED:
        return text
    return f"{color}{text}{ANSI_RESET}"


def _underline(text: str) -> str:
    if not STYLES_ENABLED:
        return text
    return f"{ANSI_UNDERLINE}{text}{ANSI_RESET}"


def _section(title: str, color: str = ANSI_BRIGHT_CYAN, icon: str = "✦") -> str:
    line = f"{icon} {title}"
    return _underline(_color(line, color))


def _label(text: str) -> str:
    # Inline label for key names
    return _color(text, ANSI_BRIGHT_CYAN)


EXAMPLES_TEXT = r"""
Quick start examples
  1) Compile a Llama chat model for causal LM
     optimum-rbln-cli --output-dir ./compiled_llama \
       --model-id meta-llama/Llama-2-7b-chat-hf \
       --batch-size 2 --tensor-parallel-size 4

  2) Compile with explicit class (Auto sequence classification)
     optimum-rbln-cli --output-dir ./compiled_bert \
       --class RBLNAutoModelForSequenceClassification \
       --model-id bert-base-uncased \
       --batch-size 8 --max-seq-len 512

  3) Pass nested rbln_config with dot-notation (e.g., for diffusion)
     optimum-rbln-cli --output-dir ./compiled_sd \
       --model-id runwayml/stable-diffusion-v1-5 \
       --unet.batch_size 2 --vae.batch_size 1

Notes
  - Any extra --key value pairs not defined above are collected into rbln_config
    and forwarded to from_pretrained(..., rbln_config=...).
  - Use --list-classes to see available RBLN classes.
  - Use --show-rbln-config to see accepted rbln_config keys for the resolved class
    (via --class or inferred from --model-id).
  - Show this examples list anytime with:  optimum-rbln-cli --examples
"""


def _list_available_rbln_classes():
    """Return a sorted list of (name, kind) for available RBLN classes; kind in {"Model","Pipeline","Auto"}."""
    try:
        # Import lazily exposed module and enumerate public names
        import optimum.rbln as rbln  # noqa: WPS433 (third-party import within function)

        # Import bases for filtering
        RBLNBaseModel = getattr(rbln, "RBLNBaseModel", None)
        RBLNDiffusionMixin = getattr(rbln, "RBLNDiffusionMixin", None)

        class_names = []
        for name in dir(rbln):
            if not name.startswith("RBLN"):
                continue
            try:
                obj = getattr(rbln, name)
                if not inspect.isclass(obj):
                    continue
                # Exclude config classes and obvious non-user-facing bases
                if name.endswith("Config") or name in {"RBLNModel", "RBLNBaseModel", "RBLNDiffusionMixin"}:
                    continue

                # Keep only concrete models/pipelines/auto
                is_model = RBLNBaseModel is not None and isinstance(obj, type) and issubclass(obj, RBLNBaseModel)
                is_pipeline = (
                    RBLNDiffusionMixin is not None and isinstance(obj, type) and issubclass(obj, RBLNDiffusionMixin)
                )
                is_auto = name.startswith("RBLNAuto")
                if is_model:
                    class_names.append((name, "Model"))
                elif is_pipeline:
                    class_names.append((name, "Pipeline"))
                elif is_auto:
                    class_names.append((name, "Auto"))
            except Exception:
                # Skip anything that errors on attribute access
                continue
        # Deduplicate and sort by kind then name
        unique = {(n, k) for (n, k) in class_names}
        return sorted(unique, key=lambda x: (x[1], x[0]))
    except Exception:
        return []


def _print_rbln_config_options(class_name: str):
    """Inspect the RBLN config class for a given model/pipeline and print accepted rbln_config keys."""
    try:
        model_cls = get_rbln_model_cls(class_name)
    except Exception as e:
        print(f"Unknown RBLN class: {class_name}. Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Obtain the associated config class
    try:
        config_cls = model_cls.get_rbln_config_class()
    except Exception:
        print(
            f"The class '{class_name}' does not provide an associated RBLN config class.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Description from both class docstring and __init__ docstring
    class_doc = None
    init_doc = None
    try:
        class_doc = inspect.getdoc(config_cls)
    except Exception:
        class_doc = None
    try:
        init_doc = inspect.getdoc(getattr(config_cls, "__init__", None))
    except Exception:
        init_doc = None

    # Base and specific parameter sets via signature introspection
    base_params = set()
    try:
        base_sig = inspect.signature(RBLNModelConfig.__init__)
        base_params = {p.name for p in base_sig.parameters.values() if p.name not in {"self"}}
    except Exception:
        pass

    try:
        cfg_sig = inspect.signature(config_cls.__init__)
        cfg_params = [p for p in cfg_sig.parameters.values() if p.name not in {"self"}]
    except Exception:
        cfg_params = []

    # Identify submodule keys if present
    submodules = []
    try:
        submodules = list(getattr(config_cls, "submodules", []) or [])
    except Exception:
        submodules = []

    # Categorize parameters
    common_keys = []
    specific_keys = []
    for p in cfg_params:
        if p.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        if p.name in submodules:
            continue
        if p.name in base_params:
            common_keys.append(p)
        else:
            specific_keys.append(p)

    print(_section(f"RBLN class: {class_name}", ANSI_BRIGHT_BLUE, icon="🧩"))
    print(_underline(_color(f"Config class: {config_cls.__name__}", ANSI_BRIGHT_CYAN)))
    if class_doc:
        print(_underline("\nDescription (class):"))
        for line in class_doc.splitlines():
            print(f"  {line}")
    if init_doc:
        print(_underline("\nDescription (__init__):"))
        for line in init_doc.splitlines():
            print(f"  {line}")
    if submodules:
        print(_underline("\nSubmodules:"))
        for s in submodules:
            print(f"  • {s}  {_color('(use nested keys like --' + s + '.batch_size 2)', ANSI_DIM)}")

    # Curated: common compile-time options that live in RBLNModelConfig (non-runtime)
    print(_underline("\nCommon compile-time options (in rbln_config):"))
    print("  • npu: Target NPU for compilation (e.g., 'RBLN-CA25').")
    print("  • tensor_parallel_size: Number of NPUs to shard the model at compile time.")

    print(_underline("\nTips:"))
    print("  - Pass config keys as CLI flags, e.g., --batch_size 2 --max_seq_len 4096")
    print("  - Compile-time examples: --npu RBLN-CA25 --tensor_parallel_size 4")
    print("  - Use dot-notation for submodules, e.g., --vision_tower.image_size 336 --language_model.batch_size 1")
    print("  - To see examples: optimum-rbln-cli --examples")


def _read_json_from_model_id(
    model_id: str,
    filename: str,
    *,
    hf_token: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    hf_force_download: bool = False,
    hf_local_files_only: bool = False,
) -> Optional[dict]:
    """Read a JSON file (e.g., config.json or model_index.json) from a local path or the HuggingFace Hub.

    Args:
        model_id: Local directory path or HuggingFace Hub repo id
        filename: Name of the JSON file to read

    Returns:
        Parsed JSON dictionary if found, else None
    """
    # Local directory
    local_dir = Path(model_id)
    if local_dir.exists() and local_dir.is_dir():
        local_file = local_dir / filename
        if local_file.exists():
            try:
                with local_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

    # HuggingFace Hub
    try:
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=hf_revision,
            token=hf_token,
            cache_dir=hf_cache_dir,
            force_download=hf_force_download,
            local_files_only=hf_local_files_only,
        )
        p = Path(downloaded_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None

    return None


def _infer_rbln_class_from_model_id(
    model_id: str,
    *,
    hf_token: Optional[str] = None,
    hf_revision: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    hf_force_download: bool = False,
    hf_local_files_only: bool = False,
) -> Optional[str]:
    """Infer RBLN class name from model files by prefixing discovered class with 'RBLN'.

    Order of precedence:
      1) model_index.json['pipeline'] -> e.g., 'StableDiffusionPipeline' -> 'RBLNStableDiffusionPipeline'
      2) config.json['architectures'][0] -> e.g., 'LlamaForCausalLM' -> 'RBLNLlamaForCausalLM'
    """
    # 1) Diffusers-style pipeline
    model_index = _read_json_from_model_id(
        model_id,
        "model_index.json",
        hf_token=hf_token,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        hf_force_download=hf_force_download,
        hf_local_files_only=hf_local_files_only,
    )
    if isinstance(model_index, dict):
        pipeline_cls = model_index.get("_class_name")
        if isinstance(pipeline_cls, str) and pipeline_cls:
            return f"RBLN{pipeline_cls}"

    # 2) Transformers config architectures
    cfg = _read_json_from_model_id(
        model_id,
        "config.json",
        hf_token=hf_token,
        hf_revision=hf_revision,
        hf_cache_dir=hf_cache_dir,
        hf_force_download=hf_force_download,
        hf_local_files_only=hf_local_files_only,
    )
    if isinstance(cfg, dict):
        architectures = cfg.get("architectures")
        if isinstance(architectures, list) and architectures:
            arch0 = architectures[0]
            if isinstance(arch0, str) and arch0:
                return f"RBLN{arch0}"

    return None


def main():
    """
    Main CLI function for optimum-rbln model compilation.
    """
    # Pre-parse lightweight flags that should work without other required args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--list-classes", action="store_true", help="List available RBLN classes and exit")
    pre_parser.add_argument("--examples", action="store_true", help="Show quick start examples and exit")
    pre_parser.add_argument("--version", action="store_true", help="Show version and exit")
    pre_parser.add_argument("--no-style", action="store_true", help="Disable ANSI styling in output")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.version:
        print(f"optimum-rbln-cli {__version__}")
        return

    # Apply style preference as early as possible
    global STYLES_ENABLED
    if pre_args.no_style:
        STYLES_ENABLED = False

    if pre_args.list_classes:
        classes = _list_available_rbln_classes()
        if not classes:
            print(_section("No RBLN classes found", ANSI_RED, icon="✖"))
            print("Please ensure the package is installed correctly.")
        else:
            autos = [n for n, k in classes if k == "Auto"]
            models = [n for n, k in classes if k == "Model"]
            pipes = [n for n, k in classes if k == "Pipeline"]
            print(_section("Available RBLN classes (use with --class)", ANSI_BRIGHT_BLUE, icon="📚"))
            if autos:
                print(_underline(_color("\nAuto classes:", ANSI_BRIGHT_YELLOW)))
                for name in autos:
                    print(f"  • {name}")
            if models:
                print(_underline(_color("\nModels:", ANSI_BRIGHT_GREEN)))
                for name in models:
                    print(f"  • {name}")
            if pipes:
                print(_underline(_color("\nPipelines:", ANSI_BRIGHT_MAGENTA)))
                for name in pipes:
                    print(f"  • {name}")
            print(f"\nTotal: {_underline(str(len(classes)))}")
        return

    if pre_args.examples:
        print(EXAMPLES_TEXT)
        return

    parser = argparse.ArgumentParser(
        description=(
            "Compile and export HuggingFace models/pipelines for RBLN devices.\n\n"
            "Required: --model-id.\n"
            "Additional --key value pairs are forwarded to rbln_config.\n"
            "Use dot-notation for nested fields (e.g., --unet.batch_size 2)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLES_TEXT,
    )

    parser.add_argument(
        "--model-id",
        dest="model_id",
        type=str,
        required=True,
        help="Model ID from HuggingFace Hub or local directory path",
    )

    # Optional output directory argument (defaults to ./rbln_out)
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="./rbln_out",
        help="Directory where the compiled model will be saved (default: ./rbln_out)",
    )

    # Optional class argument (can be inferred)
    parser.add_argument(
        "--class",
        dest="model_class",
        type=str,
        required=False,
        help=(
            "RBLN model class to use for compilation (e.g., RBLNLlamaForCausalLM, RBLNAutoModelForCausalLM). "
            "If omitted, it will be inferred from model_id by reading model_index.json or config.json."
        ),
    )

    # Optional flag to show rbln_config for the resolved class (no compilation)
    parser.add_argument(
        "--show-rbln-config",
        dest="show_rbln_config",
        action="store_true",
        help="Show rbln_config keys for the resolved RBLN class (via --class or inferred from --model-id) and exit",
    )

    # Standard --version that integrates with argparse (works after full parse)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )
    parser.add_argument("--no-style", action="store_true", help="Disable ANSI styling in output")

    # HuggingFace Hub access options
    parser.add_argument(
        "--hf-token",
        dest="hf_token",
        type=str,
        default=None,
        help="HuggingFace token to access private repositories",
    )
    parser.add_argument(
        "--hf-revision",
        dest="hf_revision",
        type=str,
        default=None,
        help="Specific model revision to download (branch, tag, or commit)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        dest="hf_cache_dir",
        type=str,
        default=None,
        help="Directory to use as HuggingFace download cache",
    )
    parser.add_argument(
        "--hf-force-download",
        dest="hf_force_download",
        action="store_true",
        help="Force redownload of files from the HuggingFace Hub",
    )
    parser.add_argument(
        "--hf-local-files-only",
        dest="hf_local_files_only",
        action="store_true",
        help="Only use local files and do not attempt to download from the network",
    )
    # All other arguments will be parsed dynamically and passed to from_pretrained

    # Print help with examples when no args were provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(2)

    # Parse known args to allow for additional rbln_* arguments
    args, unknown_args = parser.parse_known_args()

    try:
        # Resolve or infer model class for compilation
        resolved_class_name: Optional[str] = args.model_class
        if not resolved_class_name:
            resolved_class_name = _infer_rbln_class_from_model_id(
                args.model_id,
                hf_token=args.hf_token,
                hf_revision=args.hf_revision,
                hf_cache_dir=args.hf_cache_dir,
                hf_force_download=args.hf_force_download,
                hf_local_files_only=args.hf_local_files_only,
            )
            if not resolved_class_name:
                print(
                    "Could not infer RBLN class from model files. Please specify --class explicitly.",
                    file=sys.stderr,
                )
                sys.exit(2)

        if args.show_rbln_config:
            _print_rbln_config_options(resolved_class_name)
            return

        # Get the model class using the utility function (with helpful error)
        try:
            model_class = get_rbln_model_cls(resolved_class_name)
        except AttributeError:
            print(
                f"Unknown RBLN class: {resolved_class_name}.\n"
                "Run 'optimum-rbln-cli --list-classes' to see available classes.",
                file=sys.stderr,
            )
            sys.exit(2)

        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare rbln_config by parsing all unknown arguments
        rbln_config = {}

        # Parse all unknown arguments
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if arg.startswith("--"):
                arg_name = arg[2:].replace("-", "_")
                if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    # Has a value
                    arg_value = unknown_args[i + 1]
                    parsed_value = parse_value(arg_value)

                    # Check if this is a nested config argument (contains dots)
                    if "." in arg_name:
                        set_nested_dict(rbln_config, arg_name, parsed_value)
                    else:
                        rbln_config[arg_name] = parsed_value
                    i += 2
                else:
                    # Boolean flag
                    if "." in arg_name:
                        set_nested_dict(rbln_config, arg_name, True)
                    else:
                        rbln_config[arg_name] = True
                    i += 1
            else:
                i += 1

        # Set create_runtimes to False by default for CLI compilation if not specified
        create_runtimes = rbln_config.pop("create_runtimes", False)

        print(_section("Starting compilation", ANSI_BRIGHT_BLUE, icon="🚀"))
        print(f"{_label('Model:')} {args.model_id}")
        print(f"{_label('Class:')} {resolved_class_name}")
        print(f"{_label('Output:')} {output_path.absolute()}")
        print(f"{_label('rbln_config:')} {json.dumps(rbln_config, indent=2, ensure_ascii=False)}")

        with ContextRblnConfig(create_runtimes=create_runtimes):
            _ = model_class.from_pretrained(
                args.model_id, export=True, model_save_dir=str(output_path), rbln_config=rbln_config
            )

        print(_section("Model compilation completed successfully", ANSI_BRIGHT_GREEN, icon="✅"))
        print(f"Saved to: {output_path.absolute()}")

    except Exception as e:
        print(f"❌ Error during model compilation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
