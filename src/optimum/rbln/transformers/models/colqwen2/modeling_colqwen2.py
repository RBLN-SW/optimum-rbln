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


from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from transformers import ColQwen2Config, ColQwen2ForRetrieval
from transformers.modeling_utils import no_init_weights
from transformers.models.colqwen2.modeling_colqwen2 import ColQwen2ForRetrievalOutput

from ....modeling import RBLNModel
from ....transformers.modeling_outputs import _validate_output_hidden_states


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class RBLNColQwen2ForRetrieval(RBLNModel):
    """
    RBLNColQwen2ForRetrieval is a model for document retrieval using vision-language models.
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based `ColQwen2ForRetrieval` model on RBLN devices.
    It implements the methods to convert a pre-trained transformers `ColQwen2ForRetrieval` model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Examples:
        ```python
        import torch
        from PIL import Image
        from transformers import ColQwen2Processor

        from optimum.rbln import RBLNColQwen2ForRetrieval, RBLNColQwen2ForRetrievalConfig

        rbln_config = {
            "vlm": {
                "visual": {
                    "max_seq_lens": 6400,
                },
                "tensor_parallel_size": 4,
                "kvcache_partition_len": 16384,
                "max_seq_len": 16384 * 7,
            },
        }
        model = RBLNColQwen2ForRetrieval.from_pretrained("vidore/colqwen2-v1.0-hf", rbln_config=config)
        model.save_pretrained("compiled-colqwen2-v1.0-hf")

        # The document page screenshots from your corpus. Below are dummy images.
        images = [
            Image.new("RGB", (128, 128), color="white"),
            Image.new("RGB", (64, 32), color="black"),
        ]
        processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0-hf")

        queries = [
            "When was the United States Declaration of Independence proclaimed?",
            "Who printed the edition of Romeo and Juliet?",
        ]
        inputs_images = processor(images=images)
        inputs_text = processor(text=queries)

        # Forward pass
        with torch.no_grad():
            image_embeddings = model(**inputs_images).embeddings
            query_embeddings = model(**inputs_text).embeddings

        scores = processor.score_retrieval(query_embeddings, image_embeddings)
        print("Retrieval scores (query x image):")
        print(scores)
        ```
    """

    _rbln_submodule_postfix = "model"
    _rbln_submodules = [
        {"name": "vlm"},
    ]
    _supports_non_fp32 = True

    def __post_init__(self, **kwargs):
        self.vlm_model = self.rbln_submodules[0]
        return super().__post_init__(**kwargs)

    @classmethod
    def _reconstruct_model_if_needed(cls, model: "PreTrainedModel"):
        # if model is from Colpali-engine, convert it to a ColQwen2ForRetrieval model
        if hasattr(model, "custom_text_proj"):
            with no_init_weights():
                model_config = ColQwen2Config(
                    vlm_config=model.config, embedding_dim=model.custom_text_proj.out_features
                )
                new_model = ColQwen2ForRetrieval._from_config(model_config)
            new_model.embedding_proj_layer = model.custom_text_proj
            new_model.vlm.model.visual.load_state_dict(model.visual.state_dict())
            new_model.vlm.model.language_model.load_state_dict(model.language_model.state_dict())
            model = new_model

        # replace the lm_head with the custom text projection layer for optimization
        model.vlm.model.lm_head = model.embedding_proj_layer
        model.vlm.model.config.embedding_dim = model.config.embedding_dim

        # Some of the model weights are different from the model.dtype(vidore/colqwen2-v1.0-hf)
        return model.to(model.dtype)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, ColQwen2ForRetrievalOutput]:
        """
        Runs a ColQwen2 retrieval forward pass on text tokens and optional visual inputs.

        Args:
            input_ids (torch.LongTensor, optional): Indices of the textual tokens. Mutually exclusive with `inputs_embeds`.
            inputs_embeds (torch.FloatTensor, optional): Pre-computed embeddings fed directly into the language model.
            attention_mask (torch.Tensor, optional): Mask that selects which token positions contribute to the loss/embeddings.
            pixel_values (torch.Tensor, optional): Flattened image patches produced by `ColQwen2Processor` for document pages.
            image_grid_thw (torch.LongTensor, optional): Per-image `(t, h, w)` grid metadata that allows unpadding of `pixel_values`.
            output_hidden_states (bool, optional): If `True`, expose intermediate decoder hidden states.
            return_dict (bool, optional): If `True`, return a `ColQwen2ForRetrievalOutput`; otherwise return a tuple.
            **kwargs (dict[str, Any], optional): Extra multimodal args forwarded to the wrapped VLM (e.g. `pixel_values_videos`,
                `video_grid_thw`, `second_per_grid_ts`).

        Returns:
            Dataclass containing the embeddings and hidden states of the VLM model.
        """
        output_hidden_states = _validate_output_hidden_states(output_hidden_states, self.rbln_config)

        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            pixel_values = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(pixel_values, offsets)],
                dim=0,
            )

        vlm_output = self.vlm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = vlm_output.hidden_states if output_hidden_states else None

        embeddings = vlm_output[0]
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        return ColQwen2ForRetrievalOutput(
            embeddings=embeddings,
            hidden_states=hidden_states,
        )
