# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
Refer to huggingface/optimum/blob/4fdeea77d71e79451ba53e0c1f9d8f37e9704268/optimum/utils/save_utils.py
"""

from pathlib import Path

from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

from .logging import get_logger


logger = get_logger(__name__)


def maybe_load_preprocessors(
    src_name_or_path: str | Path, subfolder: str = "", trust_remote_code: bool = False
) -> list:
    preprocessors = []
    try:
        preprocessors.append(
            AutoTokenizer.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoProcessor.from_pretrained(src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code)
        )
    except Exception:
        pass

    try:
        preprocessors.append(
            AutoFeatureExtractor.from_pretrained(
                src_name_or_path, subfolder=subfolder, trust_remote_code=trust_remote_code
            )
        )
    except Exception:
        pass
    return preprocessors


def maybe_save_preprocessors(
    src_name_or_path: str | Path,
    dest_dir: str | Path,
    src_subfolder: str = "",
    trust_remote_code: bool = False,
):
    """
    Saves the tokenizer, the processor and the feature extractor when found in `src_dir` in `dest_dir`.

    Args:
        src_dir (`Union[str, Path]`):
            The source directory from which to copy the files.
        dest_dir (`Union[str, Path]`):
            The destination directory to copy the files to.
        src_subfolder (`str`, defaults to `""`):
            In case the preprocessor files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        trust_remote_code (`bool`, defaults to `False`):
            Whether to allow to save preprocessors that is allowed to run arbitrary code. Use this option at your own risk.
    """
    if not isinstance(dest_dir, Path):
        dest_dir = Path(dest_dir)

    dest_dir.mkdir(exist_ok=True)
    preprocessors = maybe_load_preprocessors(
        src_name_or_path, subfolder=src_subfolder, trust_remote_code=trust_remote_code
    )
    for preprocessor in preprocessors:
        preprocessor.save_pretrained(dest_dir)
    return preprocessors
