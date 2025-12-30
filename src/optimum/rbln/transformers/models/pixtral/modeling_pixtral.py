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

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import rebel
import torch
import torch.nn as nn
from transformers import PixtralVisionConfig, PixtralVisionModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import no_init_weights
from transformers.models.pixtral.modeling_pixtral import PixtralRMSNorm, PixtralRotaryEmbedding

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from .configuration_pixtral import RBLNPixtralVisionModelConfig
from .pixtral_architecture import PixtralAttention


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel


class RBLNRuntimePixtralVisionModel(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        config: PixtralVisionConfig,
        rbln_config: RBLNPixtralVisionModelConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.patch_positional_embedding = PixtralRotaryEmbedding(config)
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.hidden_size = config.hidden_size
        self.max_image_size = rbln_config.max_image_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if pixel_values.shape[2] > self.max_image_size[0] or pixel_values.shape[3] > self.max_image_size[1]:
            raise ValueError("The height() and width of pixel_values can't be larger than max_image_size.")

        if pixel_values.shape[2] != self.max_image_size[0] or pixel_values.shape[3] != self.max_image_size[1]:
            padded_pixel_values = [
                torch.nn.functional.pad(
                    image,
                    pad=(
                        0,
                        self.max_image_size[1] - pixel_values.shape[3],
                        0,
                        self.max_image_size[0] - pixel_values.shape[2],
                    ),
                )
                for image in pixel_values
            ]
            pixel_values = torch.stack(padded_pixel_values)

        batch_size, _, H_max, W_max = pixel_values.shape
        H_max_p = H_max // self.patch_size
        W_max_p = W_max // self.patch_size

        final_hidden_states = None

        last_hidden_state_list = []
        if output_hidden_states:
            batch_hidden_states_list = []

        for i in range(batch_size):
            h_patched_original = image_sizes[i, 0] // self.patch_size
            w_patched_original = image_sizes[i, 1] // self.patch_size

            single_pixel_values = pixel_values[i : i + 1]
            patch_embed = self.patch_conv(single_pixel_values)
            patch_embed_seq = patch_embed[:, :, :h_patched_original, :w_patched_original].flatten(2).transpose(1, 2)
            patch_embed_seq = self.ln_pre(patch_embed_seq)
            patch_embed_seq = nn.functional.pad(
                patch_embed_seq, (0, 0, 0, H_max_p * W_max_p - patch_embed_seq.shape[1]), "constant", value=0
            )

            max_w_from_config = self.image_size // self.patch_size
            mesh = torch.meshgrid(torch.arange(h_patched_original), torch.arange(w_patched_original), indexing="ij")
            h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
            ids = h_grid * max_w_from_config + v_grid
            position_ids = ids[:, 0]

            position_embeddings = self.patch_positional_embedding(patch_embed_seq, position_ids)
            cos = nn.functional.pad(
                position_embeddings[0],
                (0, 0, 0, H_max_p * W_max_p - position_embeddings[0].shape[0]),
                "constant",
                value=0,
            )
            sin = nn.functional.pad(
                position_embeddings[1],
                (0, 0, 0, H_max_p * W_max_p - position_embeddings[1].shape[0]),
                "constant",
                value=0,
            )

            attention_mask = torch.full(
                (1, patch_embed_seq.shape[-2]), fill_value=torch.finfo(patch_embed_seq.dtype).min
            )
            attention_mask[:, : h_patched_original * w_patched_original] = 0
            if "out" in kwargs:
                super().forward(patch_embed_seq, attention_mask, cos, sin, **kwargs)
                transformer_output = kwargs["out"]
            else:
                transformer_output = super().forward(patch_embed_seq, attention_mask, cos, sin, **kwargs)

            last_hidden_state_list.append(transformer_output[0][:, : h_patched_original * w_patched_original, :])
            hidden_states = transformer_output[1:]

            if output_hidden_states:
                batch_hidden_states_list.append(
                    [hidden_state[:, : h_patched_original * w_patched_original, :] for hidden_state in hidden_states]
                )

        final_last_hidden_state = torch.cat(last_hidden_state_list, dim=1)

        if output_hidden_states:
            hidden_states = [
                torch.cat(
                    [batch_hidden_states[layer_idx] for batch_hidden_states in batch_hidden_states_list],
                    dim=1,
                )
                for layer_idx in range(len(batch_hidden_states_list[0]))
            ]

            final_hidden_states = tuple(hidden_states)

        if not return_dict:
            return tuple(v for v in (final_last_hidden_state, final_hidden_states) if v is not None)

        # TODO: output_attentions
        return BaseModelOutput(
            last_hidden_state=final_last_hidden_state,
            hidden_states=final_hidden_states,
        )


class _PixtralVisionModel(torch.nn.Module):
    def __init__(self, model: PixtralVisionModel, output_hidden_states: bool):
        super().__init__()
        self.transformer = self.convert_to_rbln_pixtral_vision_model(model)
        self.output_hidden_states = output_hidden_states

    def convert_to_rbln_pixtral_vision_model(self, model: nn.Module):
        for layer in model.transformer.layers:
            layer.attention = PixtralAttention(layer.attention)
        return model.transformer

    def forward(self, patch_embeds, attention_mask, position_embeddings_1, position_embeddings_2):
        output = self.transformer(
            inputs_embeds=patch_embeds,
            attention_mask=attention_mask,
            position_embeddings=(position_embeddings_1, position_embeddings_2),
            output_hidden_states=self.output_hidden_states,
            return_dict=False,
        )
        return output


class RBLNPixtralVisionModel(RBLNModel):
    """
    RBLN optimized Pixtral vision encoder model.

    This class provides hardware-accelerated inference for Pixtral vision encoders
    on RBLN devices, supporting image encoding for multimodal tasks.
    """

    def __post_init__(self, **kwargs):
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        with no_init_weights():
            self.patch_conv = nn.Conv2d(
                in_channels=self.config.num_channels,
                out_channels=self.config.hidden_size,
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
                bias=False,
            )
            self.ln_pre = PixtralRMSNorm(self.config.hidden_size, eps=1e-5)
        self.patch_conv.load_state_dict(artifacts["patch_conv"])
        self.ln_pre.load_state_dict(artifacts["ln_pre"])
        self.model = RBLNRuntimePixtralVisionModel(
            self.model[0],
            main_input_name="pixel_values",
            config=self.config,
            rbln_config=self.rbln_config,
            patch_conv=self.patch_conv,
            ln_pre=self.ln_pre,
        )

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNModelConfig,
    ):
        save_dict = {}
        save_dict["patch_conv"] = model.get_input_embeddings().state_dict()
        save_dict["ln_pre"] = model.ln_pre.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(
        cls, model: torch.nn.Module, rbln_config: RBLNPixtralVisionModelConfig
    ) -> torch.nn.Module:
        wrapper_cfg = {
            "output_hidden_states": rbln_config.output_hidden_states,
        }
        return _PixtralVisionModel(model, **wrapper_cfg).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "PixtralVisionConfig" = None,
        rbln_config: Optional[RBLNPixtralVisionModelConfig] = None,
    ) -> RBLNPixtralVisionModelConfig:
        if rbln_config.max_image_size is None:
            rbln_config.max_image_size = (model_config.image_size, model_config.image_size)

        if rbln_config.output_hidden_states is None:
            rbln_config.output_hidden_states = getattr(model_config, "output_hidden_states", False)

        num_total_patches = (rbln_config.max_image_size[0] // model_config.patch_size) * (
            rbln_config.max_image_size[1] // model_config.patch_size
        )

        rbln_compile_config = RBLNCompileConfig(
            input_info=[
                (
                    "patch_embeds",
                    [1, num_total_patches, model_config.hidden_size],
                    "float32",
                ),
                ("attention_mask", [1, num_total_patches], "float32"),
                (
                    "position_embeddings_1",
                    [
                        num_total_patches,
                        model_config.head_dim,
                    ],
                    "float32",
                ),
                (
                    "position_embeddings_2",
                    [
                        num_total_patches,
                        model_config.head_dim,
                    ],
                    "float32",
                ),
            ]
        )

        rbln_config.set_compile_cfgs([rbln_compile_config])
        return rbln_config

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass for the RBLN-optimized Pixtral vision model.

        Args:
            pixel_values (torch.Tensor of shape (batch_size, num_channels, image_size, image_size)) — The tensors corresponding to the input images. Pixel values can be obtained using PixtralImageProcessor. See PixtralImageProcessor.call() for details (PixtralProcessor uses PixtralImageProcessor for processing images).
            image_sizes (torch.Tensor of shape (batch_size, 2), optional) — The sizes of the images in the batch, being (height, width) for each image.
            output_hidden_states (bool, optional) — Whether or not to return the hidden states of all layers. See hidden_states under returned tensors for more detail.
            return_dict (bool, optional) — Whether or not to return a ModelOutput instead of a plain tuple.

        Returns:
            BaseModelOutput or tuple(torch.FloatTensor)
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.rbln_config.output_hidden_states
        )

        if output_hidden_states != self.rbln_config.output_hidden_states:
            raise ValueError(
                f"Variable output_hidden_states {output_hidden_states} is not equal to rbln_config.output_hidden_states {self.rbln_config.output_hidden_states} "
                f"Please compile again with the correct argument."
            )

        output = self.model(
            pixel_values, image_sizes, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs
        )

        return output
