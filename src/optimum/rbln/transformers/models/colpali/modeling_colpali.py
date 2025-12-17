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

import bisect
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import no_init_weights
from transformers.models.colpali.modeling_colpali import ColPaliForRetrieval, ColPaliForRetrievalOutput

from ....configuration_utils import RBLNModelConfig
from ....modeling import RBLNModel
from ...utils.rbln_runtime_wrapper import LoopProcessor


class LoopVisionTower(LoopProcessor):
    def __init__(self, vision_tower: "RBLNModel"):
        super().__init__(model=vision_tower.model[0])

    def _get_batch_size(self, pixel_values, **kwargs):
        return pixel_values.shape[0]

    def _prepare_inputs_for_iteration(self, index, common_inputs, pixel_values, **kwargs):
        pixel_values_item = pixel_values[index : index + 1]
        out_buffer = kwargs["out"][index : index + 1]
        return ([pixel_values_item], {"out": out_buffer})

    def _process_outputs(self, outputs: list, **kwargs) -> "BaseModelOutputWithPooling":
        return BaseModelOutputWithPooling(
            last_hidden_state=kwargs["out"],
        )


class LoopLanguageModel(LoopProcessor):
    def __init__(self, language_model: RBLNModel, rbln_config: RBLNModelConfig):
        super().__init__(model=language_model)
        self.rbln_config = rbln_config

    def _get_batch_size(self, inputs_embeds, **kwargs):
        return inputs_embeds.shape[0]

    def _prepare_inputs_before_loop(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        input_len = inputs_embeds.shape[1]
        idx = bisect.bisect_left(self.rbln_config.max_seq_lens, input_len)
        if idx == len(self.rbln_config.max_seq_lens):
            raise ValueError(
                f"Required seq_len({input_len}) is larger than available max_seq_lens({self.rbln_config.max_seq_lens})."
            )
        max_seq_len = self.rbln_config.max_seq_lens[idx]
        padded_inputs_embed = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, max_seq_len - input_len))
        padded_attn_mask = torch.nn.functional.pad(attention_mask, (0, max_seq_len - input_len)).to(torch.float32)
        padded_position_ids = torch.arange(max_seq_len, dtype=torch.int32).view(1, -1)

        return {
            "padded_inputs_embed": padded_inputs_embed,
            "padded_attn_mask": padded_attn_mask,
            "padded_position_ids": padded_position_ids,
        }

    def _prepare_inputs_for_iteration(self, index: int, common_inputs, *args, **kwargs):
        item_kwargs = {
            "inputs_embeds": common_inputs["padded_inputs_embed"][index : index + 1],
            "attention_mask": common_inputs["padded_attn_mask"][index : index + 1],
            "position_ids": common_inputs["padded_position_ids"],
            "out": [tensor[index : index + 1] for tensor in kwargs["out"]],
        }
        return ([], item_kwargs)

    def _process_outputs(self, outputs: list, **kwargs):
        if self.rbln_config.output_hidden_states:
            return kwargs["out"][0], tuple(kwargs["out"][1:])
        else:
            return kwargs["out"]


class RBLNColPaliForRetrieval(RBLNModel):
    """
    The ColPali Model transformer for document retrieval using vision-language models.
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based `ColPaliForRetrieval` model on RBLN devices.
    It implements the methods to convert a pre-trained transformers `ColPaliForRetrieval` model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    **Configuration:**
    This model uses [`RBLNColPaliForRetrievalConfig`] for configuration. When calling methods like `from_pretrained` or `from_model`,
    the `rbln_config` parameter should be an instance of [`RBLNColPaliForRetrievalConfig`] or a dictionary conforming to its structure.

    See the [`RBLNColPaliForRetrievalConfig`] class for all available configuration options.

    Examples:
        ```python
        from optimum.rbln import RBLNColPaliForRetrieval

        # Simple usage using rbln_* arguments
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config={
                "vlm": {
                    "language_model": {
                        "prefill_chunk_size": 8192, # same as model's max_position_embeddings (max_seq_len)
                    }
                }
            }
        )

        # Using a config dictionary
        rbln_config = {
            "vlm": {
                "language_model": {
                    "prefill_chunk_size": 8192, # same as model's max_position_embeddings (max_seq_len)
                }
            }
        }
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config=rbln_config
        )

        # Using a RBLNColPaliForRetrievalConfig instance (recommended for type checking)
        from optimum.rbln import RBLNColPaliForRetrievalConfig

        config = RBLNColPaliForRetrievalConfig(
            vlm={
                "language_model": {"prefill_chunk_size": 8192},
            },
            output_hidden_states=False,
            tensor_parallel_size=4
        )
        model = RBLNColPaliForRetrieval.from_pretrained(
            "vidore/colpali-v1.3-hf",
            export=True,
            rbln_config=config
        )
        ```
    """

    auto_model_class = None
    _rbln_submodule_postfix = "model"
    _rbln_submodules = [
        {"name": "vlm"},
    ]

    def __post_init__(self, **kwargs):
        self.vlm_model = self.rbln_submodules[0]
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self.embedding_proj_layer = self._create_embedding_proj_layer()
        self.embedding_proj_layer.load_state_dict(artifacts["embedding_proj_layer"])
        return super().__post_init__(**kwargs)

    def _create_embedding_proj_layer(self):
        with no_init_weights():
            embedding_proj_layer = torch.nn.Linear(
                self.config.vlm_config.text_config.hidden_size, self.config.embedding_dim
            )
        return embedding_proj_layer

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "ColPaliForRetrieval",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["embedding_proj_layer"] = model.embedding_proj_layer.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ColPaliForRetrievalOutput]:
        """
        Forward pass for the RBLN-optimized ColPaliForRetrieval model.

        Args:
            input_ids (torch.LongTensor of shape (batch_size, sequence_length)): Indices of input sequence tokens in the vocabulary.
            pixel_values (torch.Tensor of shape (batch_size, num_channels, image_size, image_size)): The tensors corresponding to the input images.
            attention_mask (torch.Tensor of shape (batch_size, sequence_length)): Mask to avoid performing attention on padding token indices.
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
            return_dict (bool, optional): Whether or not to return a ModelOutput instead of a plain tuple.

        Returns:
            ColPaliForRetrievalOutput or tuple(torch.FloatTensor)
        """
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.rbln_config.output_hidden_states
        )
        if output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        vlm_output = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        vlm_hidden_states = vlm_output.hidden_states if output_hidden_states else None
        vlm_image_hidden_states = vlm_output.image_hidden_states if pixel_values is not None else None

        last_hidden_states = vlm_output[0]
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(last_hidden_states.to(proj_dtype))
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        return ColPaliForRetrievalOutput(
            embeddings=embeddings,
            hidden_states=vlm_hidden_states,
            image_hidden_states=vlm_image_hidden_states,
        )
