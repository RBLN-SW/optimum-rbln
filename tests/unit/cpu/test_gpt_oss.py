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


@pytest.fixture
def mock_checkpoint_loading(monkeypatch):
    load_weights = Mock(return_value=["model.safetensors"])
    create_model = Mock(return_value=torch.nn.Module())
    monkeypatch.setattr(modeling_gpt_oss, "load_weight_files", load_weights)
    monkeypatch.setattr(modeling_gpt_oss, "load_file", Mock(return_value={}))
    monkeypatch.setattr(modeling_gpt_oss, "_replace_with_mxfp4_linear", Mock())
    monkeypatch.setattr(modeling_gpt_oss.AutoModelForCausalLM, "from_config", create_model)
    return load_weights, create_model


@pytest.mark.parametrize("provide_config", [False, True])
def test_gpt_oss_loads_weights_at_requested_revision(monkeypatch, mock_checkpoint_loading, provide_config):
    repo_id = "org/gpt-oss"
    revision = "a" * 40
    config = GptOssConfig()
    load_weights, create_model = mock_checkpoint_loading
    load_config = Mock(return_value=(config, {}))
    monkeypatch.setattr(modeling_gpt_oss.AutoConfig, "from_pretrained", load_config)

    modeling_gpt_oss.RBLNGptOssForCausalLM.get_pytorch_model(
        repo_id, revision=revision, token=True, config=config if provide_config else None
    )

    load_weights.assert_called_once_with(repo_id, revision=revision, token=True, exception_keywords=["original"])
    if not provide_config:
        load_config.assert_called_once_with(repo_id, revision=revision, token=True, return_unused_kwargs=True)
    else:
        load_config.assert_not_called()
    create_model.assert_called_once_with(config, dtype=torch.bfloat16)


def test_gpt_oss_applies_overrides_to_loaded_config(tmp_path, mock_checkpoint_loading):
    GptOssConfig(num_hidden_layers=4).save_pretrained(tmp_path)
    _, create_model = mock_checkpoint_loading
    layer_types = ["full_attention", "sliding_attention"]

    modeling_gpt_oss.RBLNGptOssForCausalLM.get_pytorch_model(
        str(tmp_path), num_hidden_layers=2, layer_types=layer_types
    )

    config = create_model.call_args.args[0]
    assert config.num_hidden_layers == 2
    assert config.layer_types == layer_types
    assert create_model.call_args.kwargs == {"dtype": torch.bfloat16}
