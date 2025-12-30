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
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run command and return success status."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✅ Command succeeded")
    return True


def test_cli_help():
    """Test CLI help command."""
    print("\n🔍 Testing CLI help command...")
    return run_command(["uv", "run", "python", "-m", "optimum.rbln.cli", "--help"])


def test_resnet_compilation():
    """Test ResNet model compilation."""
    print("\n🔍 Testing ResNet model compilation...")

    test_output_dir = "/tmp/test_cli_resnet"

    # Clean up if exists
    if Path(test_output_dir).exists():
        shutil.rmtree(test_output_dir)

    # Run CLI compilation
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "optimum.rbln.cli",
        "--output-dir",
        test_output_dir,
        "--model-id",
        "hf-internal-testing/tiny-random-ResNetForImageClassification",
    ]

    if not run_command(cmd):
        return False

    # Check if required files are generated
    output_path = Path(test_output_dir)

    required_files = ["rbln_config.json", "compiled_model.rbln"]
    for file_name in required_files:
        file_path = output_path / file_name
        if not file_path.exists():
            print(f"❌ Required file not found: {file_path}")
            return False
        print(f"✅ Found required file: {file_path}")

    # Clean up
    shutil.rmtree(test_output_dir)
    print("✅ ResNet compilation test passed")
    return True


def test_stable_diffusion_compilation():
    """Test Stable Diffusion model compilation."""
    print("\n🔍 Testing Stable Diffusion model compilation...")

    test_output_dir = "/tmp/test_cli_stable_diffusion"

    # Clean up if exists
    if Path(test_output_dir).exists():
        shutil.rmtree(test_output_dir)

    # Run CLI compilation with diffusion-specific config
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "optimum.rbln.cli",
        "--output-dir",
        test_output_dir,
        "--model-id",
        "hf-internal-testing/tiny-sd-pipe",
        "--vae.sample_size",
        "64,64",  # Fix incorrect tiny-sd-pipe's vae config.json sample_size
        "--unet.batch_size",
        "2",
    ]

    if not run_command(cmd):
        return False

    # Check if submodel directories and files are generated
    output_path = Path(test_output_dir)

    submodels = ["vae", "text_encoder", "unet"]
    for submodel in submodels:
        submodel_path = output_path / submodel
        if not submodel_path.exists():
            print(f"❌ Submodel directory not found: {submodel_path}")
            return False
        print(f"✅ Found submodel directory: {submodel_path}")

        # Check for rbln_config.json in each submodel
        config_file = submodel_path / "rbln_config.json"
        if not config_file.exists():
            print(f"❌ rbln_config.json not found in: {submodel_path}")
            return False
        print(f"✅ Found rbln_config.json in: {submodel_path}")

        # Check for .rbln files in each submodel
        rbln_files = list(submodel_path.glob("*.rbln"))
        if not rbln_files:
            print(f"❌ No .rbln files found in: {submodel_path}")
            return False
        print(f"✅ Found .rbln files in {submodel_path}: {[f.name for f in rbln_files]}")

    # Clean up
    shutil.rmtree(test_output_dir)
    print("✅ Stable Diffusion compilation test passed")
    return True


def test_argument_parsing():
    """Test CLI argument parsing and error handling."""
    print("\n🔍 Testing CLI argument parsing...")

    # Test missing required arguments
    print("Testing missing required arguments...")
    result = subprocess.run(["uv", "run", "python", "-m", "optimum.rbln.cli"], capture_output=True, text=True)
    if result.returncode == 0:
        print("❌ Expected non-zero exit for missing arguments")
        return False
    print("✅ Correctly failed with missing arguments (help shown)")

    # Test invalid model class
    print("Testing invalid model class...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "optimum.rbln.cli",
            "--output-dir",
            "/tmp/test",
            "--class",
            "InvalidClass",
            "--model-id",
            "test",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("❌ Expected error for invalid model class")
        return False
    print("✅ Correctly failed with invalid model class error")

    print("✅ Argument parsing tests passed")
    return True


def test_error_handling():
    """Test CLI error handling."""
    print("\n🔍 Testing CLI error handling...")

    # Test with non-existent model
    print("Testing non-existent model...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "optimum.rbln.cli",
            "/tmp/test_error",
            "--class",
            "RBLNAutoModelForCausalLM",
            "--model-id",
            "non-existent-model",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("❌ Expected error for non-existent model")
        return False
    print("✅ Correctly failed with non-existent model error")

    print("✅ Error handling tests passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test CLI functionality")
    parser.add_argument(
        "test_type", choices=["basic", "argument-parsing", "error-handling"], help="Type of test to run"
    )

    args = parser.parse_args()

    success = True

    if args.test_type == "basic":
        success &= test_cli_help()
        success &= test_resnet_compilation()
        success &= test_stable_diffusion_compilation()
    elif args.test_type == "argument-parsing":
        success &= test_argument_parsing()
    elif args.test_type == "error-handling":
        success &= test_error_handling()

    if success:
        print(f"\n🎉 All {args.test_type} tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 Some {args.test_type} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
