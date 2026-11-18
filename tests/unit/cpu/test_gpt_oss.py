# Copyright 2026 Rebellions Inc. All rights reserved.
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

from unittest.mock import Mock

import pytest
import torch
from transformers import GptOssConfig

from optimum.rbln.transformers.models.gpt_oss import modeling_gpt_oss


@pytest.mark.parametrize("provide_config", [False, True])
def test_gpt_oss_loads_weights_at_requested_revision(monkeypatch, provide_config):
    repo_id = "org/gpt-oss"
    revision = "a" * 40
    config = GptOssConfig()
    load_weights = Mock(return_value=["model.safetensors"])
    load_config = Mock(return_value=(config, {}))
    create_model = Mock(return_value=torch.nn.Module())
    monkeypatch.setattr(modeling_gpt_oss, "load_weight_files", load_weights)
    monkeypatch.setattr(modeling_gpt_oss, "load_file", Mock(return_value={}))
    monkeypatch.setattr(modeling_gpt_oss, "_replace_with_mxfp4_linear", Mock())
    monkeypatch.setattr(modeling_gpt_oss.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(modeling_gpt_oss.AutoModelForCausalLM, "from_config", create_model)

    modeling_gpt_oss.RBLNGptOssForCausalLM.get_pytorch_model(
        repo_id, revision=revision, token=True, config=config if provide_config else None
    )

    load_weights.assert_called_once_with(repo_id, revision=revision, token=True, exception_keywords=["original"])
    if not provide_config:
        load_config.assert_called_once_with(repo_id, revision=revision, token=True, return_unused_kwargs=True)
    else:
        load_config.assert_not_called()
    create_model.assert_called_once_with(config, dtype=torch.bfloat16)
