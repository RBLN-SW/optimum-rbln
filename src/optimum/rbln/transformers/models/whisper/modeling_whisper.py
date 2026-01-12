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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoModelForSpeechSeq2Seq, WhisperForConditionalGeneration, WhisperModel
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from .configuration_whisper import RBLNWhisperForConditionalGenerationConfig
from .generation_whisper import RBLNWhisperGenerationMixin
from .whisper_architecture import WhisperWrapper


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
        GenerationConfig,
        PretrainedConfig,
        PreTrainedModel,
    )


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
        decoder_input_ids: torch.Tensor = None,
        decoder_attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        block_tables: torch.Tensor = None,
    ):
        inputs_bsz = decoder_input_ids.shape[0]
        padded_bsz = self.batch_size - inputs_bsz

        if padded_bsz > 0:
            decoder_input_ids = torch.nn.functional.pad(decoder_input_ids, (0, 0, 0, padded_bsz))

        if self.use_attention_mask:
            for b_idx in range(self.batch_size):
                decoding_step = cache_position[b_idx].item()
                if not (0 <= decoding_step < self.dec_max_seq_len):
                    raise ValueError(
                        f"Decoding step {decoding_step} out of bounds for attention mask with shape {self.dec_attn_mask.shape}."
                    )
                decoder_attention_mask[b_idx, : decoding_step + 1] = 1

        if block_tables is None:
            block_tables = self.default_block_tables

        outputs = super().forward(
            decoder_input_ids,
            decoder_attention_mask if self.use_attention_mask else None,
            cache_position,
            block_tables=block_tables,
        )

        if isinstance(outputs, torch.Tensor):
            return Seq2SeqLMOutput(logits=outputs[:inputs_bsz], cross_attentions=None)
        else:
            return Seq2SeqLMOutput(logits=outputs[0][:inputs_bsz], cross_attentions=outputs[1][:, :inputs_bsz])


class RBLNWhisperForConditionalGeneration(RBLNModel, RBLNWhisperGenerationMixin):
    """
    Whisper model for speech recognition and transcription optimized for RBLN NPU.

    This model inherits from [`RBLNModel`]. It implements the methods to convert and run
    pre-trained transformers based Whisper model on RBLN devices by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN compiler.

    Example (Short form):
    ```python
    import torch
    from transformers import AutoProcessor
    from datasets import load_dataset
    from optimum.rbln import RBLNWhisperForConditionalGeneration

    # Load processor and dataset
    model_id = "openai/whisper-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # Prepare input features
    input_features = processor(
        ds[0]["audio"]["array"],
        sampling_rate=ds[0]["audio"]["sampling_rate"],
        return_tensors="pt"
    ).input_features

    # Load and compile model (or load pre-compiled model)
    model = RBLNWhisperForConditionalGeneration.from_pretrained(
        model_id=model_id,
        export=True,
        rbln_batch_size=1
    )

    # Generate transcription
    outputs = model.generate(input_features=input_features, return_timestamps=True)
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Transcription: {transcription}")
    ```
    """

    auto_model_class = AutoModelForSpeechSeq2Seq
    main_input_name = "input_features"
    _is_stateful = False

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        self.batch_size = self.rbln_config.batch_size
        self.dec_max_seq_len = self.rbln_config.dec_max_seq_len
        self.rbln_token_timestamps = self.rbln_config.token_timestamps
        self.use_attention_mask = self.rbln_config.use_attention_mask

        self.encoder = RBLNRuntimeEncoder(runtime=self.model[0], main_input_name="input_features")
        self.decoder = RBLNRuntimeDecoder(
            runtime=self.model[1],
            main_input_name="input_ids",
            batch_size=self.batch_size,
            dec_max_seq_len=self.dec_max_seq_len,
            use_attention_mask=self.use_attention_mask,
        )

        # skip encoder &  first decoder when language detected
        self.is_language_detected = False
        self.language_cross = None

        # Used in GenerationMixin.generate()
        # transformers/models/whisper/generation_whisper.py, line 505, in generate
        #     input_stride = self.model.encoder.conv1.stride[0] * self.model.encoder.conv2.stride[0]
        self.model = WhisperModel(self.config)
        self.pad_token_id = self.config.pad_token_id

    def can_generate(self):
        return True

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(WhisperForConditionalGeneration, __name)
        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    def _reorder_cache(self, past_key_values, beam_idx):
        # TODO(jongho): implement
        raise NotImplementedError

    @classmethod
    def _wrap_model_if_needed(self, model: "PreTrainedModel", rbln_config: RBLNWhisperForConditionalGenerationConfig):
        return WhisperWrapper(
            model,
            use_attention_mask=rbln_config.use_attention_mask,
            rbln_token_timestamps=rbln_config.token_timestamps,
        )

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model, rbln_config: RBLNWhisperForConditionalGenerationConfig):
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
    def _update_paged_attention_config(
        cls, model_config: "PretrainedConfig", rbln_config: RBLNWhisperForConditionalGenerationConfig
    ):
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
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNWhisperForConditionalGenerationConfig] = None,
    ) -> RBLNWhisperForConditionalGenerationConfig:
        expected_seq_len = model_config.max_source_positions * 2
        num_mel_bins = model_config.num_mel_bins
        rbln_config.enc_max_seq_len = model_config.max_source_positions

        # 'whisper-large-v3-turbo' doesn't have 'max_length', but PretrainedConfig have default value for the key 'max_length'
        rbln_config.dec_max_seq_len = getattr(model_config, "max_target_positions", None)
        if rbln_config.dec_max_seq_len is None:
            rbln_config.dec_max_seq_len = model_config.max_length

        cls._update_paged_attention_config(model_config, rbln_config)

        enc_input_info = [
            ("input_features", [1, num_mel_bins, expected_seq_len], "float32"),
            ("block_tables", [1], "int16"),
            (
                "cross_key_value_states",
                [
                    model_config.decoder_layers * 2,
                    rbln_config.batch_size,
                    model_config.decoder_attention_heads,
                    rbln_config.enc_max_seq_len,
                    model_config.d_model // model_config.decoder_attention_heads,
                ],
                "float32",
            ),
        ]

        dec_input_info = [
            ("decoder_input_ids", [rbln_config.batch_size, 1], "int64"),
            ("cache_position", [rbln_config.batch_size, 1], "int32"),
            ("block_tables", [rbln_config.batch_size, 1], "int16"),
        ]
        dec_input_info.extend(
            [
                (
                    "cross_key_value_states",
                    [
                        model_config.decoder_layers * 2,
                        rbln_config.batch_size,
                        model_config.decoder_attention_heads,
                        rbln_config.enc_max_seq_len,
                        model_config.d_model // model_config.decoder_attention_heads,
                    ],
                    "float32",
                )
            ]
        )
        dec_input_info.extend(
            [
                (
                    f"self_key_value_states_{i}",
                    [
                        rbln_config.batch_size,
                        model_config.decoder_attention_heads,
                        rbln_config.dec_max_seq_len,
                        model_config.d_model // model_config.encoder_attention_heads,
                    ],
                    "float32",
                )
                for i in range(model_config.decoder_layers * 2)
            ]
        )

        if rbln_config.use_attention_mask:
            dec_input_info.insert(
                1, ("decoder_attention_mask", [rbln_config.batch_size, rbln_config.dec_max_seq_len], "float32")
            )

        enc_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config.set_compile_cfgs([enc_compile_config, dec_compile_config])

        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNWhisperForConditionalGenerationConfig,
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # need for support transformers>=4.45.0
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "cache_position": cache_position,
        }

    # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/generation/utils.py#L512
    def _prepare_encoder_decoder_kwargs_for_generation(
        self,
        inputs_tensor: torch.Tensor,
        model_kwargs,
        model_input_name: Optional[str] = None,
        generation_config: Optional["GenerationConfig"] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size = inputs_tensor.shape[0]
        n_pad_to_batch = self.batch_size - batch_size
        if n_pad_to_batch > 0:
            inputs_tensor = torch.nn.functional.pad(inputs_tensor, (0, 0, 0, 0, 0, n_pad_to_batch))

        if not self.is_language_detected:
            for b in range(inputs_tensor.shape[0]):
                block_tables = torch.tensor([b], dtype=torch.int16)
                model_kwargs["encoder_outputs"] = self.encoder(
                    input_features=inputs_tensor[b].unsqueeze(0), block_tables=block_tables
                )
            self.decoder_attention_mask = torch.zeros(self.batch_size, self.dec_max_seq_len, dtype=torch.float32)
        else:
            model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=torch.tensor([[-1.0]]))

        return model_kwargs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Seq2SeqLMOutput] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        # default decoder pass
        if input_features is None and encoder_outputs is None:
            cross_attentions = []
            for step in cache_position:
                # skip step 0 if language_detection has been processed
                if step == 0 and self.is_language_detected:
                    cross_attentions.append(self.language_cross)
                    self.is_language_detected = False
                else:
                    self.decoder_attention_mask[:, step] = 1
                    decoder_output = self.decoder(
                        decoder_input_ids=input_ids[:, step : step + 1].contiguous(),
                        decoder_attention_mask=self.decoder_attention_mask,
                        cache_position=torch.full((self.batch_size, 1), step, dtype=torch.int32),
                    )
                    cross_attentions.append(decoder_output.cross_attentions)
                    lm_logits = decoder_output.logits

            if self.rbln_token_timestamps:
                cross_attentions = torch.cat(cross_attentions, dim=-2)
            else:
                cross_attentions = None

            return Seq2SeqLMOutput(logits=lm_logits, cross_attentions=cross_attentions)

        # detect language pass
        # https://github.com/huggingface/transformers/blob/174890280b340b89c5bfa092f6b4fb0e2dc2d7fc/src/transformers/models/whisper/generation_whisper.py#L1442
        else:
            # for language auto detection (generate with language=None)
            if encoder_outputs is None:
                for b in range(input_features.shape[0]):
                    block_tables = torch.tensor([b], dtype=torch.int16)
                    self.encoder(input_features=input_features[b].unsqueeze(0), block_tables=block_tables)

            self.decoder_attention_mask = torch.zeros(self.batch_size, self.dec_max_seq_len, dtype=torch.float32)
            self.is_language_detected = True
            self.decoder_attention_mask[:, 0] = 1
            decoder_output = self.decoder(
                decoder_input_ids=decoder_input_ids.contiguous(),
                decoder_attention_mask=self.decoder_attention_mask,
                cache_position=torch.zeros([self.rbln_config.batch_size, 1], dtype=torch.int32),
            )
            lm_logits = decoder_output.logits
            self.language_cross = decoder_output.cross_attentions
            return Seq2SeqLMOutput(logits=lm_logits)
