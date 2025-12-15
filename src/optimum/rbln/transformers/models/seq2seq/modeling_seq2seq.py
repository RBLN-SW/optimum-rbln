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

import inspect
from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoModelForSeq2SeqLM, PretrainedConfig, PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, ModelOutput, Seq2SeqLMOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from .configuration_seq2seq import RBLNModelForSeq2SeqLMConfig


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig


class RBLNRuntimeEncoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def forward(self, *args: List[torch.Tensor], **kwargs: torch.Tensor):
        output = super().forward(*args, **kwargs)
        return BaseModelOutput(last_hidden_state=output)


class RBLNRuntimeDecoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        batch_size: int,
        dec_max_seq_len: int,
        use_attention_mask: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self.batch_size = batch_size
        self.dec_max_seq_len = dec_max_seq_len
        self.use_attention_mask = use_attention_mask
        self.default_block_tables = torch.arange(0, self.batch_size, dtype=torch.int16).view(self.batch_size, 1)

    def forward(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        batch_size = decoder_input_ids.shape[0]
        if batch_size != self.batch_size:
            raise RuntimeError(
                f"Batch size mismatch: got {batch_size}, expected {self.batch_size} (compiled batch size)."
            )

        if batch_size != cache_position.shape[0]:
            raise RuntimeError(f"Cache position size mismatch: got {cache_position.shape[0]}, expected {batch_size}.")

        if self.use_attention_mask:
            for b_idx in range(self.batch_size):
                decoding_step = cache_position[b_idx].item()
                if not (0 <= decoding_step < self.dec_max_seq_len):
                    raise ValueError(
                        f"Decoding step {decoding_step} out of bounds for decoder_max_seq_len ({self.dec_max_seq_len})."
                    )
                decoder_attention_mask[b_idx, : decoding_step + 1] = 1

        if block_tables is None:
            block_tables = self.default_block_tables

        lm_logits = super().forward(
            decoder_input_ids,
            decoder_attention_mask if self.use_attention_mask else None,
            attention_mask,
            cache_position,
            block_tables=block_tables,
        )

        return Seq2SeqLMOutput(logits=lm_logits)


class RBLNModelForSeq2SeqLM(RBLNModel, GenerationMixin, ABC):
    """
    This is a generic model class that will be instantiated as one of the model classes of the library (with a sequence-to-sequence language modeling head) when created with the from_pretrained() class method.
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformers based Seq2SeqLM models on RBLN devices.
    It implements the methods to convert a pre-trained transformers Seq2SeqLM model into a RBLN transformer model by:
    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Currently, this model class only supports the 'bart' and 't5' models from the transformers library. Future updates may include support for additional model types.
    """

    main_input_name = "input_ids"
    auto_model_class = AutoModelForSeq2SeqLM
    support_causal_attn = None
    _is_stateful = False

    def __post_init__(self, **kwargs):
        batch_size = self.rbln_config.batch_size
        dec_max_seq_len = self.rbln_config.dec_max_seq_len
        self.use_attention_mask = self.rbln_config.use_attention_mask

        self.encoder = RBLNRuntimeEncoder(
            runtime=self.model[0],
            main_input_name="input_ids",
        )
        self.decoder = RBLNRuntimeDecoder(
            runtime=self.model[1],
            main_input_name="input_ids",
            batch_size=batch_size,
            dec_max_seq_len=dec_max_seq_len,
            use_attention_mask=self.use_attention_mask,
        )

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: PreTrainedModel, rbln_config: RBLNModelForSeq2SeqLMConfig):
        wrapped_model = cls._wrap_model_if_needed(model, rbln_config)

        enc_compile_config = rbln_config.compile_cfgs[0]
        dec_compile_config = rbln_config.compile_cfgs[1]

        context = CompileContext(use_weight_sharing=False)

        enc_example_inputs = enc_compile_config.get_dummy_inputs(fill=0)

        # Mark encoder's static tensors (cross kv states)
        static_tensors = {}
        for (name, _, _), tensor in zip(enc_compile_config.input_info, enc_example_inputs):
            if "key_value_states" in name:
                static_tensors[name] = tensor
                context.mark_static_address(tensor)

        dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)

        # Mark decoder's static tensors (self kv states)
        for (name, _, _), tensor in zip(dec_compile_config.input_info, dec_example_inputs):
            if "key_value_states" in name:
                context.mark_static_address(tensor)

        compiled_encoder = cls.compile(
            wrapped_model.encoder,
            enc_compile_config,
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
            example_inputs=enc_example_inputs,
            compile_context=context,
        )

        compiled_decoder = cls.compile(
            wrapped_model.decoder,
            dec_compile_config,
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
            example_inputs=dec_example_inputs,
            compile_context=context,
        )

        return {"encoder": compiled_encoder, "decoder": compiled_decoder}

    @classmethod
    def _update_paged_attention_config(cls, model_config: PretrainedConfig, rbln_config: RBLNModelForSeq2SeqLMConfig):
        rbln_config.kvcache_num_blocks = rbln_config.kvcache_num_blocks or rbln_config.batch_size
        rbln_config.kvcache_block_size = rbln_config.kvcache_block_size or rbln_config.dec_max_seq_len

        if rbln_config.kvcache_num_blocks != rbln_config.batch_size:
            raise NotImplementedError(
                f"kvcache_num_blocks ({rbln_config.kvcache_num_blocks}) must be equal to batch_size ({rbln_config.batch_size}) as flash attention is not supported yet."
            )

        if rbln_config.kvcache_block_size != rbln_config.dec_max_seq_len:
            raise NotImplementedError(
                f"kvcache_block_size ({rbln_config.kvcache_block_size}) must be equal to dec_max_seq_len ({rbln_config.dec_max_seq_len}) as flash attention is not supported yet."
            )

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNModelForSeq2SeqLMConfig] = None,
    ) -> RBLNModelForSeq2SeqLMConfig:
        if not cls.support_causal_attn:
            rbln_config.use_attention_mask = True

        n_layer = getattr(model_config, "decoder_layers", None) or model_config.num_layers
        n_head = getattr(model_config, "decoder_attention_heads", None) or model_config.num_heads
        d_kv = (
            model_config.d_kv
            if hasattr(model_config, "d_kv")
            else model_config.d_model // model_config.encoder_attention_heads
        )

        max_position_embeddings = getattr(model_config, "n_positions", None) or getattr(
            model_config, "max_position_embeddings", None
        )

        if rbln_config.enc_max_seq_len is None:
            enc_max_seq_len = max_position_embeddings
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_max_length"):
                    enc_max_seq_len = enc_max_seq_len or tokenizer.model_max_length
                    break

            if enc_max_seq_len is None:
                raise ValueError("`enc_max_seq_len` should be specified!")
            rbln_config.enc_max_seq_len = enc_max_seq_len

        if max_position_embeddings is not None and rbln_config.enc_max_seq_len > max_position_embeddings:
            raise ValueError("`enc_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_config.dec_max_seq_len is None:
            dec_max_seq_len = max_position_embeddings
            for tokenizer in preprocessors:
                if hasattr(tokenizer, "model_max_length"):
                    dec_max_seq_len = dec_max_seq_len or tokenizer.model_max_length
                    break

            if dec_max_seq_len is None:
                raise ValueError("`dec_max_seq_len` should be specified!")
            rbln_config.dec_max_seq_len = dec_max_seq_len

        if max_position_embeddings is not None and rbln_config.dec_max_seq_len > max_position_embeddings:
            raise ValueError("`dec_max_seq_len` should be less or equal than max_position_embeddings!")

        if rbln_config.support_paged_attention:
            cls._update_paged_attention_config(model_config, rbln_config)

        # model input info
        enc_input_info = [
            ("input_ids", [1, rbln_config.enc_max_seq_len], "int64"),
            ("attention_mask", [1, rbln_config.enc_max_seq_len], "float32"),
            ("block_tables", [1], "int16"),
        ]
        enc_input_info.extend(
            [
                (
                    f"cross_key_value_states_{i}",
                    [
                        rbln_config.batch_size,
                        n_head,
                        rbln_config.enc_max_seq_len,
                        d_kv,
                    ],
                    "float32",
                )
                for i in range(n_layer * 2)
            ]
        )

        dec_input_info = [
            ("input_ids", [rbln_config.batch_size, 1], "int64"),
            ("encoder_attention_mask", [rbln_config.batch_size, rbln_config.enc_max_seq_len], "float32"),
            (
                "cache_position",
                [rbln_config.batch_size, 1],
                "int32",
            ),
            ("block_tables", [rbln_config.batch_size, 1], "int16"),
        ]
        dec_input_info.extend(
            [
                (
                    f"cross_key_value_states_{i}",
                    [
                        rbln_config.batch_size,
                        n_head,
                        rbln_config.enc_max_seq_len,
                        d_kv,
                    ],
                    "float32",
                )
                for i in range(n_layer * 2)
            ]
        )
        dec_input_info.extend(
            [
                (
                    f"self_key_value_states_{i}",
                    [
                        rbln_config.batch_size,
                        n_head,
                        rbln_config.dec_max_seq_len,
                        d_kv,
                    ],
                    "float32",
                )
                for i in range(n_layer * 2)
            ]
        )

        if rbln_config.use_attention_mask:
            dec_input_info.insert(
                1, ("attention_mask", [rbln_config.batch_size, rbln_config.dec_max_seq_len], "float32")
            )

        enc_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config.set_compile_cfgs([enc_compile_config, dec_compile_config])

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNModelForSeq2SeqLMConfig,
    ) -> List[rebel.Runtime]:
        if any(model_name not in rbln_config.device_map for model_name in ["encoder", "decoder"]):
            cls._raise_missing_compiled_file_error(["encoder", "decoder"])

        return [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["encoder"],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            ),
            rebel.Runtime(
                compiled_models[1],
                tensor_type="pt",
                device=rbln_config.device_map["decoder"],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            ),
        ]

    def can_generate(self):
        return True

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        cur_seq_len = input_ids.shape[-1]
        cache_position = cur_seq_len - 1
        max_seq_len = self.rbln_config.dec_max_seq_len
        decoder_batch_size = input_ids.shape[0]
        input_ids = input_ids[:, cur_seq_len - 1 : cur_seq_len].contiguous()
        decoder_attention_mask = torch.zeros(decoder_batch_size, max_seq_len, dtype=torch.float32)
        decoder_attention_mask[:, :cur_seq_len] = 1

        return {
            "decoder_input_ids": input_ids,
            "attention_mask": attention_mask.to(torch.float32),
            "decoder_attention_mask": decoder_attention_mask,
            "cache_position": cache_position,
        }

    def forward(
        self,
        decoder_input_ids: torch.LongTensor = None,
        cache_position: Union[List[torch.Tensor], torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        # common decoder
        cache_position = torch.full((self.rbln_config.batch_size, 1), cache_position, dtype=torch.int32)
        logits = self.decoder(decoder_input_ids=decoder_input_ids, cache_position=cache_position, **kwargs).logits

        return Seq2SeqLMOutput(
            logits=logits,
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
        generation_config: Optional["GenerationConfig"] = None,
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        batch_size, input_len = inputs_tensor.shape
        inputs_tensor = torch.nn.functional.pad(
            inputs_tensor,
            (0, self.rbln_config.enc_max_seq_len - input_len),
            value=self.config.pad_token_id,
        )
        model_kwargs["attention_mask"] = torch.nn.functional.pad(
            model_kwargs["attention_mask"], (0, self.rbln_config.enc_max_seq_len - input_len)
        )

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs["output_hidden_states"] = False
        encoder_kwargs["output_attentions"] = False

        for b in range(batch_size):
            block_tables = torch.tensor([b], dtype=torch.int16)
            encoder_kwargs["input_ids"] = inputs_tensor[b].unsqueeze(0)
            encoder_kwargs["attention_mask"] = model_kwargs["attention_mask"][b].unsqueeze(0).to(torch.float32)
            model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs, block_tables=block_tables)

        return model_kwargs

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> Union[ModelOutput, torch.LongTensor]:
        """
        The generate function is utilized in its standard form as in the HuggingFace transformers library. User can use this function to generate text from the model.
        Check the [HuggingFace transformers documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationMixin.generate) for more details.

        Args:
            input_ids (torch.LongTensor): The input ids to the model.
            attention_mask (torch.LongTensor, optional): The attention mask to the model.
            generation_config (GenerationConfig, optional): The generation configuration to be used as base parametrization for the generation call. **kwargs passed to generate matching the attributes of generation_config will override them.
                If generation_config is not provided, the default will be used, which had the following loading priority: 1) from the generation_config.json model file, if it exists; 2) from the model configuration.
                Please note that unspecified parameters will inherit [GenerationConfig](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation#transformers.GenerationConfig)’s default values.
            kwargs (dict[str, Any], optional): Additional arguments passed to the generate function. See the HuggingFace transformers documentation for more details.

        Returns:
            Generates sequences of token ids for models with a language modeling head.
        """
        if generation_config is not None:
            kwargs["generation_config"] = generation_config
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask

        return super().generate(input_ids, **kwargs)
