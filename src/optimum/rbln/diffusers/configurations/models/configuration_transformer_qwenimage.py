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

from typing import Any

from ....configuration_utils import RBLNModelConfig


class RBLNQwenImageTransformer2DModelConfig(RBLNModelConfig):
    """
    Configuration class for RBLN QwenImageTransformer2DModel.

    This class provides configuration options for the transformer model used in
    the Qwen-Image-Edit diffusion pipeline.
    """

    subclass_non_save_attributes = ["_batch_size_is_specified", "_sample_size"]
    _sample_size = None

    def __init__(
        self,
        batch_size: int | None = None,
        prompt_embed_length: int | None = None,
        sample_size: int | tuple[int, int] | None = None,
        num_img_groups: int = 2,
        tensor_parallel_size: int = 1,
        hoist_modulation: str | bool | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (Optional[int]): The batch size for inference. Defaults to 1.
            sample_size (Optional[Union[int, Tuple[int, int]]]): The spatial dimensions (height, width)
                of the latent samples. If an integer is provided, it's used for both height and width.
            prompt_embed_length (Optional[int]): The length of the embedded prompt vectors.
            num_img_groups (int): The number of image shape groups passed to the transformer
                for RoPE computation. In QwenImageEdit, this is typically 2 (one for the
                target latent shape and one for the source image latent shape). Defaults to 2.
            hoist_modulation (str | bool | None): Which of each block's modulation layers (``SiLU + Linear``, a
                function of the timestep only) to compute on the host and feed to the compiled graph as one small
                input instead of compiling their weights into it. The hoisted weights are saved next to the artifact
                (``modulation.pth``) and evaluated on the host per timestep (cached per timestep).
                ``"txt"`` (default when compiling): the text-stream ``txt_mod`` — 6.4 GiB less weight on the chip's
                first memory node (RBLN-CR13, rebel 0.11.2), which lets the text encoder and VAE share the NPU with the
                transformer; numerically indistinguishable from the in-graph version.
                ``"all"``: also the image-stream ``img_mod`` — a ~17% faster forward and fully balanced memory nodes,
                but the compiler then repartitions the image stream and its result drifts further from an fp32
                reference (4% -> 11% after 60 blocks on 0.11.2); use only after checking quality.
                ``"none"`` / ``False``: keep everything in the graph. ``True`` means ``"txt"``. ``None`` lets the
                artifact decide when loading (it has a ``mod_params`` input iff it was compiled with hoisting).
            kwargs: Additional arguments passed to the parent RBLNModelConfig.
        """
        super().__init__(**kwargs)
        self._batch_size_is_specified = batch_size is not None

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.prompt_embed_length = prompt_embed_length
        self.num_img_groups = num_img_groups
        self.tensor_parallel_size = tensor_parallel_size
        if hoist_modulation is True:
            hoist_modulation = "txt"
        elif hoist_modulation is False:
            hoist_modulation = "none"
        if hoist_modulation not in (None, "none", "txt", "all"):
            raise ValueError(
                f"hoist_modulation must be one of None, 'none', 'txt', 'all' (or a bool), got {hoist_modulation!r}"
            )
        self.hoist_modulation = hoist_modulation
        self._sample_size = sample_size

    @property
    def batch_size_is_specified(self):
        return self._batch_size_is_specified


__all__ = [
    "RBLNQwenImageTransformer2DModelConfig",
]
