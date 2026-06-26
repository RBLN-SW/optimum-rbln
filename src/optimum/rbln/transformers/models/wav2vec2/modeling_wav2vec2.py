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


from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import AutoModelForCTC, Wav2Vec2Config, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from .configuration_wav2vec2 import RBLNWav2Vec2ForCTCConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


class _Wav2Vec2(torch.nn.Module):
    def __init__(self, model: "Wav2Vec2ForCTC"):
        super().__init__()
        self.model = model

    def forward(self, input_values):
        output = self.model.wav2vec2(input_values=input_values)
        return self.model.lm_head(output[0])


class RBLNWav2Vec2ForCTC(RBLNModel):
    """
    Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).

    It implements the methods to convert a pre-trained Wav2Vec2 model into a RBLN Wav2Vec2 model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.
    """

    main_input_name = "input_values"
    auto_model_class = AutoModelForCTC
    rbln_dtype = "float32"

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNWav2Vec2ForCTCConfig) -> torch.nn.Module:
        return _Wav2Vec2(model).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "Wav2Vec2Config" = None,
        rbln_config: Optional[RBLNWav2Vec2ForCTCConfig] = None,
    ) -> RBLNWav2Vec2ForCTCConfig:
        if rbln_config.max_seq_len is None:
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_max_length"):
                    rbln_config.max_seq_len = tokenizer.model_max_length
                    break
            if rbln_config.max_seq_len is None:
                raise ValueError("`rbln_max_seq_len` should be specified!")

        rbln_compile_config = RBLNCompileConfig(
            input_info=[
                (
                    "input_values",
                    [
                        rbln_config.batch_size,
                        rbln_config.max_seq_len,
                    ],
                    "float32",
                )
            ]
        )

        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self, input_values: torch.Tensor, return_dict: Optional[bool] = None, **kwargs
    ) -> Union[CausalLMOutput, tuple]:
        """
        Forward pass for the RBLN-optimized Wav2Vec2 model for Connectionist Temporal Classification (CTC).

        Args:
            input_values (torch.FloatTensor of shape (batch_size, sequence_length)): Float values of input raw speech waveform. Values can be obtained by loading a .flac or .wav audio file into an array of type List[float] or a numpy.ndarray, e.g. via the soundfile library (pip install soundfile). To prepare the array into input_values, the AutoProcessor should be used for padding and conversion into a tensor of type torch.FloatTensor.
            return_dict (bool, optional): Whether or not to return a ModelOutput instead of a plain tuple.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a CausalLMOutput object.
        """
        return super().forward(input_values=input_values, return_dict=return_dict, **kwargs)
