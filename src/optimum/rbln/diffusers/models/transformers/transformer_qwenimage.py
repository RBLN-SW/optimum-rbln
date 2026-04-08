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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from transformers import PretrainedConfig

from ....configuration_utils import RBLNCompileConfig, RBLNModelConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ...configurations import RBLNQwenImageTransformer2DModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PreTrainedModel

    from ...modeling_diffusers import RBLNDiffusionMixin, RBLNDiffusionMixinConfig

logger = get_logger(__name__)


class QwenImageTransformer2DModelWrapper(torch.nn.Module):
    """Wrapper that simplifies QwenImageTransformer2DModel forward for RBLN compilation."""

    def __init__(self, model: "QwenImageTransformer2DModel") -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        timestep: torch.FloatTensor = None,
        guidance: torch.FloatTensor = None,
        img_shapes_tensor: torch.LongTensor = None,
    ):
        # Reconstruct img_shapes from flattened tensor:
        # img_shapes_tensor has shape [batch_size, num_img_groups, 3]
        # Convert back to list[list[tuple[int, int, int]]] for the original model
        batch_size = img_shapes_tensor.shape[0]
        num_groups = img_shapes_tensor.shape[1]
        img_shapes = []
        for b in range(batch_size):
            shapes = []
            for g in range(num_groups):
                t, h, w = img_shapes_tensor[b, g].tolist()
                if t == 0 and h == 0 and w == 0:
                    break
                shapes.append((t, h, w))
            img_shapes.append(shapes)

        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            guidance=guidance,
            img_shapes=img_shapes,
            return_dict=False,
        )


class RBLNQwenImageTransformer2DModel(RBLNModel):
    """
    RBLN implementation of QwenImageTransformer2DModel for the Qwen-Image-Edit pipeline.

    This transformer takes text+image embeddings from Qwen2.5-VL and packed image latents,
    and predicts the noise to be removed during the diffusion denoising process.

    This class inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    hf_library_name = "diffusers"
    auto_model_class = QwenImageTransformer2DModel
    _output_class = Transformer2DModelOutput

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self._img_shapes_tensor = None

    @classmethod
    def _wrap_model_if_needed(cls, model: torch.nn.Module, rbln_config: RBLNModelConfig) -> torch.nn.Module:
        return QwenImageTransformer2DModelWrapper(model).eval()

    @classmethod
    def update_rbln_config_using_pipe(
        cls, pipe: "RBLNDiffusionMixin", rbln_config: "RBLNDiffusionMixinConfig", submodule_name: str
    ) -> "RBLNDiffusionMixinConfig":
        if rbln_config.transformer.sample_size is None:
            if rbln_config.image_size is not None:
                vae_scale_factor = pipe.vae_scale_factor
                rbln_config.transformer.sample_size = (
                    rbln_config.image_size[0] // vae_scale_factor,
                    rbln_config.image_size[1] // vae_scale_factor,
                )
            else:
                rbln_config.transformer.sample_size = pipe.default_sample_size

        return rbln_config

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNQwenImageTransformer2DModelConfig,
    ) -> RBLNQwenImageTransformer2DModelConfig:
        if rbln_config.sample_size is None:
            rbln_config.sample_size = model_config.sample_size

        if isinstance(rbln_config.sample_size, int):
            rbln_config.sample_size = (rbln_config.sample_size, rbln_config.sample_size)

        # QwenImage packs latents into 2x2 patches, so sequence length = (H/2) * (W/2)
        latent_height = rbln_config.sample_size[0]
        latent_width = rbln_config.sample_size[1]
        # hidden_states is packed: [batch, (H/2)*(W/2), in_channels * 4]
        # But for the edit pipeline, image_latents are concatenated:
        # latent_model_input = torch.cat([latents, image_latents], dim=1)
        # So the seq_len is doubled
        image_seq_len = (latent_height // 2) * (latent_width // 2)
        total_seq_len = image_seq_len * 2  # latents + image_latents concatenated

        input_info = [
            (
                "hidden_states",
                [
                    rbln_config.batch_size,
                    total_seq_len,
                    model_config.in_channels,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states",
                [
                    rbln_config.batch_size,
                    rbln_config.prompt_embed_length,
                    model_config.joint_attention_dim,
                ],
                "float32",
            ),
            (
                "encoder_hidden_states_mask",
                [
                    rbln_config.batch_size,
                    rbln_config.prompt_embed_length,
                ],
                "float32",
            ),
            ("timestep", [rbln_config.batch_size], "float32"),
            ("guidance", [rbln_config.batch_size], "float32"),
            (
                "img_shapes_tensor",
                [
                    rbln_config.batch_size,
                    rbln_config.num_img_groups,
                    3,
                ],
                "int64",
            ),
        ]

        compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([compile_config])
        return rbln_config

    @property
    def compiled_batch_size(self):
        return self.rbln_config.compile_cfgs[0].input_info[0][1][0]

    def _build_img_shapes_tensor(
        self,
        img_shapes: list,
        batch_size: int,
        device: torch.device,
    ) -> torch.LongTensor:
        """Convert img_shapes list to a fixed-size tensor for compiled model input."""
        num_img_groups = self.rbln_config.num_img_groups
        tensor = torch.zeros((batch_size, num_img_groups, 3), dtype=torch.long, device=device)
        for b in range(min(batch_size, len(img_shapes))):
            for g in range(min(num_img_groups, len(img_shapes[b]))):
                t, h, w = img_shapes[b][g]
                tensor[b, g, 0] = t
                tensor[b, g, 1] = h
                tensor[b, g, 2] = w
        return tensor

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: list = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Transformer2DModelOutput, Tuple]:
        """
        Forward pass for the RBLN-optimized QwenImageTransformer2DModel.

        Args:
            hidden_states (torch.FloatTensor): Packed latent image embeddings.
            encoder_hidden_states (torch.FloatTensor): Conditional embeddings from Qwen2.5-VL.
            encoder_hidden_states_mask (torch.FloatTensor): Attention mask for encoder hidden states.
            timestep (torch.LongTensor): Current denoising step.
            img_shapes (list): Image shapes for RoPE computation.
            guidance (torch.Tensor): Guidance tensor for guidance-distilled models.
            return_dict (bool): Whether to return Transformer2DModelOutput or tuple.

        Returns:
            Union[Transformer2DModelOutput, Tuple]
        """
        batch_size = hidden_states.shape[0]

        # Convert img_shapes to tensor format
        img_shapes_tensor = self._build_img_shapes_tensor(
            img_shapes, batch_size, hidden_states.device
        )

        # Pad encoder_hidden_states_mask if needed
        if encoder_hidden_states_mask is None:
            prompt_embed_length = self.rbln_config.prompt_embed_length
            encoder_hidden_states_mask = torch.ones(
                (batch_size, prompt_embed_length),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Pad guidance if needed
        if guidance is None:
            guidance = torch.zeros(batch_size, dtype=torch.float32, device=hidden_states.device)

        return super().forward(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            timestep,
            guidance,
            img_shapes_tensor,
            return_dict=return_dict,
        )
