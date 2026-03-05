# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Union

from ....configuration_utils import RBLNModelConfig
from ..decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelConfig, RBLNDecoderOnlyModelForCausalLMConfig


class RBLNExaone4_5_ForConditionalGenerationConfig(RBLNDecoderOnlyModelForCausalLMConfig):
    submodules = ["visual"]

    def __init__(
        self,
        use_inputs_embeds: bool = True,
        visual: Optional[RBLNModelConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(use_inputs_embeds=use_inputs_embeds, **kwargs)
        if not self.use_inputs_embeds:
            raise ValueError(
                "RBLNExaone4_5_ForConditionalGenerationConfig does not allow `use_inputs_embeds=False` "
                "because multimodal prefill requires embedding replacement."
            )
        self.visual = visual


class RBLNExaone4_5_ModelConfig(RBLNDecoderOnlyModelConfig):
    submodules = ["visual"]

    def __init__(self, visual: Optional[RBLNModelConfig] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.visual = self.initialize_submodule_config(submodule_config=visual)


class RBLNExaone4_5_VisionPreTrainedModelConfig(RBLNModelConfig):
    def __init__(self, max_seq_lens: Union[int, List[int]] = None, **kwargs: Any):
        super().__init__(**kwargs)

        if max_seq_lens is not None:
            if isinstance(max_seq_lens, int):
                max_seq_lens = [max_seq_lens]
            elif isinstance(max_seq_lens, list):
                max_seq_lens.sort(reverse=True)
        else:
            raise ValueError("'max_seq_lens' must be specified.")

        self.max_seq_lens = max_seq_lens
