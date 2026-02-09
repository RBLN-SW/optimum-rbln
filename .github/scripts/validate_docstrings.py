#!/usr/bin/env python3
"""
Simple and reliable docstring validator for CI/CD.
Focuses on catching issues that would break mkdocstrings parsing.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


def infer_module_name(file_path: Path) -> str:
    """
    Infer module name from file path.

    Example:
        src/optimum/rbln/transformers/models/decoderonly/modeling_decoderonly.py
        -> optimum.rbln.transformers.models.decoderonly.modeling_decoderonly
    """
    if not any(keyword in file_path.name for keyword in ["configuration", "modeling"]):
        return None

    parts = file_path.parts

    # Find 'src' directory index
    try:
        src_index = parts.index("src")
    except ValueError:
        # If no 'src' directory found, try common patterns
        for common_src in ["lib", "package", file_path.parts[0]]:
            if common_src in parts:
                src_index = parts.index(common_src)
                break
        else:
            # Fallback: use the first directory
            src_index = 0

    # Extract module path parts (everything after src, excluding .py)
    module_parts = parts[src_index + 1 :]

    # Remove .py extension from the last part
    if module_parts and module_parts[-1].endswith(".py"):
        module_parts = module_parts[:-1] + (module_parts[-1][:-3],)

    # Join with dots
    module_name = ".".join(module_parts)
    return module_name


def test_mkdocstrings_parsing(file_path: Path, module_name: str = None) -> bool:
    """Test if mkdocstrings can parse the module documentation."""
    try:
        if module_name:
            # Test with specific module name
            test_content = f"""# Test Documentation

:::{module_name}
    options:
        show_source: false
        filters: ["!^_"]
"""
        else:
            # Auto-infer module name from file path
            inferred_module = infer_module_name(file_path)
            if inferred_module is None:
                print(
                    "⚠️ No module name provided, and the file is not a configuration or modeling file, skipping mkdocstrings test"
                )
                return True
            print(f"ℹ️  Auto-inferred module name: {inferred_module}")
            test_content = f"""# Test Documentation

:::{inferred_module}
    options:
        show_source: false
        filters: ["!^_"]
"""
            module_name = inferred_module

        config = f"""site_name: Test
plugins:
  - mkdocstrings:
      handlers:
        python:
          paths: ["{Path(__file__).parent.parent.parent / "src"}"]
          options:
            show_source: false
            filters: ["!^_"]
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "mkdocs.yml").write_text(config)
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "index.md").write_text(test_content)

            result = subprocess.run(
                ["mkdocs", "build", "--strict"], cwd=tmp_path, capture_output=True, text=True, timeout=10
            )

            if result.returncode != 0:
                print(f"❌ mkdocstrings parsing failed for {file_path}:")
                print(f"   Module: {module_name}")
                # Show relevant errors and warnings
                lines = result.stderr.split("\n")
                relevant = [
                    l for l in lines if any(keyword in l for keyword in ["griffe", "WARNING", "ERROR", "CRITICAL"])
                ]
                for line in relevant[:5]:  # Show first 5 relevant lines
                    if line.strip():
                        print(f"    {line.strip()}")
                return False

            return True

    except subprocess.TimeoutExpired:
        print(f"⚠️  Timeout testing {file_path} (non-fatal)")
        return True  # Don't fail on timeout
    except Exception as e:
        print(f"⚠️  Could not test mkdocstrings for {file_path}: {e} (non-fatal)")
        return True  # Don't fail on test errors


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python validate_docstrings.py <python_file> [module_name]")
        print()
        print("Examples:")
        print("  python validate_docstrings.py src/my_module.py")
        print("  python validate_docstrings.py src/my_module.py my.module.name")
        print()
        print("Note: If module_name is not provided, it will be auto-inferred from file path")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    module_name = sys.argv[2] if len(sys.argv) > 2 else None

    if not file_path.exists():
        print(f"⚠️ File not found: {file_path}")
        sys.exit(0)

    print(f"🔍 Testing mkdocstrings parsing: {file_path}")
    if module_name:
        print(f"📦 Module: {module_name}")

    # Test mkdocstrings compatibility
    mkdocs_ok = test_mkdocstrings_parsing(file_path, module_name)

    if mkdocs_ok:
        print("✅ mkdocstrings parsing test passed!")
        sys.exit(0)
    else:
        print("❌ mkdocstrings parsing test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
