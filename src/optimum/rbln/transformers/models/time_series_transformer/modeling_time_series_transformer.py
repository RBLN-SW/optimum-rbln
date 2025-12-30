# Copyright 2024 Rebellions Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Portions of this software are licensed under the Apache License,
# Version 2.0. See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.

# All other portions of this software, including proprietary code,
# are the intellectual property of Rebellions Inc. and may not be
# copied, modified, or distributed without prior written permission
# from Rebellions Inc.

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import PretrainedConfig, TimeSeriesTransformerForPrediction, TimeSeriesTransformerModel
from transformers.modeling_outputs import SampleTSPredictionOutput, Seq2SeqTSModelOutput
from transformers.modeling_utils import no_init_weights

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...modeling_outputs import RBLNSeq2SeqTSDecoderOutput
from .configuration_time_series_transformer import RBLNTimeSeriesTransformerForPredictionConfig
from .time_series_transformers_architecture import TimeSeriesTransformersWrapper


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel


class RBLNRuntimeEncoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def __init__(
        self,
        runtime: rebel.Runtime,
        model: TimeSeriesTransformerModel,
        **kwargs: Any,
    ) -> None:
        super().__init__(runtime, **kwargs)
        self._origin_model = model

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
    ):
        # preprocess
        transformer_inputs, loc, scale, static_feat = self._origin_model.create_network_inputs(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
        )
        enc_input = transformer_inputs[:, : self._origin_model.config.context_length, ...]

        # enc_attn_key_value_caches is updated to device dram in-place
        _ = super().forward(inputs_embeds=enc_input)

        return Seq2SeqTSModelOutput(
            loc=loc,
            scale=scale,
            static_features=static_feat,
        )


class RBLNRuntimeDecoder(RBLNPytorchRuntime):
    mandatory_members = ["main_input_name"]

    def forward(
        self,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
    ):
        block_tables = torch.zeros(1, 1, dtype=torch.int16)
        outputs = super().forward(inputs_embeds, attention_mask, cache_position, block_tables)

        return RBLNSeq2SeqTSDecoderOutput(
            params=outputs[:-1],
            last_hidden_states=outputs[-1],
        )


class RBLNTimeSeriesTransformerForPrediction(RBLNModel):
    """
    The Time Series Transformer Model with a distribution head on top for time-series forecasting. e.g., for datasets like M4, NN5, or other time series forecasting benchmarks.
    This model inherits from [`RBLNModel`]. Check the superclass documentation for the generic methods the library implements for all its models.

    A class to convert and run pre-trained transformer-based `TimeSeriesTransformerForPrediction` models on RBLN devices.
    It implements the methods to convert a pre-trained transformers `TimeSeriesTransformerForPrediction` model into a RBLN transformer model by:

    - transferring the checkpoint weights of the original into an optimized RBLN graph,
    - compiling the resulting graph using the RBLN Compiler.
    """

    auto_model_class = None
    main_input_name = "inputs_embeds"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        self.batch_size = self.rbln_config.batch_size
        self.dec_max_seq_len = self.rbln_config.dec_max_seq_len
        self.num_parallel_samples = self.rbln_config.num_parallel_samples

        with no_init_weights():
            self._origin_model = TimeSeriesTransformerForPrediction._from_config(self.config)
        artifacts = torch.load(self.model_save_dir / self.subfolder / "torch_artifacts.pth", weights_only=False)
        self._origin_model.model.embedder.load_state_dict(artifacts["embedder"])
        self.encoder = RBLNRuntimeEncoder(
            runtime=self.model[0],
            main_input_name="inputs_embeds",
            model=self._origin_model.model,
        )
        self.decoder = RBLNRuntimeDecoder(
            runtime=self.model[1],
            main_input_name="inputs_embeds",
        )

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(TimeSeriesTransformerForPrediction, __name)
        if val is not None and isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)

    @classmethod
    def _wrap_model_if_needed(
        self, model: "PreTrainedModel", rbln_config: RBLNTimeSeriesTransformerForPredictionConfig
    ):
        return TimeSeriesTransformersWrapper(model, rbln_config.num_parallel_samples)

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model, rbln_config: RBLNTimeSeriesTransformerForPredictionConfig):
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
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNTimeSeriesTransformerForPredictionConfig,
    ):
        # If you are unavoidably running on a CPU rather than an RBLN device,
        # store the torch tensor, weight, etc. in this function.

        save_dict = {}
        save_dict["embedder"] = model.model.embedder.state_dict()
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional[RBLNTimeSeriesTransformerForPredictionConfig] = None,
    ) -> RBLNTimeSeriesTransformerForPredictionConfig:
        rbln_config.num_parallel_samples = rbln_config.num_parallel_samples or model_config.num_parallel_samples

        if rbln_config.dec_max_seq_len is None:
            predict_length = model_config.prediction_length
            rbln_config.dec_max_seq_len = (
                predict_length if predict_length % 64 == 0 else predict_length + (64 - predict_length % 64)
            )

        # model input info
        enc_input_info = [
            (
                "inputs_embeds",
                [rbln_config.batch_size, model_config.context_length, model_config.feature_size],
                "float32",
            ),
        ]
        enc_input_info.extend(
            [
                (
                    "cross_key_value_states",
                    [
                        model_config.decoder_layers * 2,
                        rbln_config.batch_size,
                        model_config.decoder_attention_heads,
                        model_config.context_length,
                        model_config.d_model // model_config.decoder_attention_heads,
                    ],
                    "float32",
                )
            ]
        )

        dec_input_info = [
            (
                "inputs_embeds",
                [rbln_config.batch_size * rbln_config.num_parallel_samples, 1, model_config.feature_size],
                "float32",
            ),
            ("attention_mask", [1, rbln_config.dec_max_seq_len], "float32"),
            ("cache_position", [], "int32"),
            ("block_tables", [1, 1], "int16"),
        ]
        dec_input_info.extend(
            [
                (
                    "cross_key_value_states",
                    [
                        model_config.decoder_layers * 2,  # 4
                        rbln_config.batch_size,  # 64
                        model_config.decoder_attention_heads,  # 2
                        model_config.context_length,  # 24
                        model_config.d_model // model_config.decoder_attention_heads,  # 13
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
                        1,
                        model_config.decoder_attention_heads
                        * rbln_config.num_parallel_samples
                        * rbln_config.batch_size,
                        rbln_config.dec_max_seq_len,
                        model_config.d_model // model_config.encoder_attention_heads,
                    ],
                    "float32",
                )
                for i in range(model_config.decoder_layers * 2)
            ]
        )
        enc_compile_config = RBLNCompileConfig(compiled_model_name="encoder", input_info=enc_input_info)
        dec_compile_config = RBLNCompileConfig(compiled_model_name="decoder", input_info=dec_input_info)

        rbln_config.set_compile_cfgs([enc_compile_config, dec_compile_config])
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNTimeSeriesTransformerForPredictionConfig,
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

    def validate_batch_size(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None and v.shape[0] != self.batch_size:
                raise RuntimeError(
                    f"Batch size mismatch in '{k}': Expected {self.batch_size}, but got {v.shape[0]}. \n"
                    f"Tensor shape: {v.shape} \n\n"
                    f"Note: `batch_size` is set at compile time. \n"
                    f"To change it, pass `export=True` along with `rbln_batch_size` when calling `from_pretrained()` to trigger recompilation."
                )

    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        future_time_features: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SampleTSPredictionOutput:
        """
        Generate pass for the RBLN-optimized Time Series Transformer model for time series forecasting.

        Args:
            past_values (torch.FloatTensor of shape (batch_size, sequence_length) or (batch_size, sequence_length, input_size)): Past values of the time series, that serve as context in order to predict the future.
            past_time_features (torch.FloatTensor of shape (batch_size, sequence_length, num_features)): Required time features, which the model internally will add to past_values.
            future_time_features (torch.FloatTensor of shape (batch_size, prediction_length, num_features)): Required time features for the prediction window, which the model internally will add to future_values.
            past_observed_mask (torch.BoolTensor of shape (batch_size, sequence_length) or (batch_size, sequence_length, input_size), optional): Boolean mask to indicate which past_values were observed and which were missing.
            static_categorical_features (torch.LongTensor of shape (batch_size, number of static categorical features), optional): Optional static categorical features for which the model will learn an embedding, which it will add to the values of the time series.
            static_real_features (torch.FloatTensor of shape (batch_size, number of static real features), optional): Optional static real features which the model will add to the values of the time series.

        Returns:
            The model outputs. If return_dict=False is passed, returns a tuple of tensors. Otherwise, returns a SampleTSPredictionOutput object.
        """
        self.validate_batch_size(**{k: v for k, v in locals().items() if isinstance(v, torch.Tensor)})

        outputs = self.encoder(
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            past_time_features=past_time_features,
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_time_features=future_time_features,
        )

        loc = outputs.loc
        scale = outputs.scale
        static_feat = outputs.static_features

        num_parallel_samples = self.num_parallel_samples
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_values = (
            past_values.repeat_interleave(repeats=num_parallel_samples, dim=0) - repeated_loc
        ) / repeated_scale

        expanded_static_feat = static_feat.unsqueeze(1).expand(-1, future_time_features.shape[1], -1)
        features = torch.cat((expanded_static_feat, future_time_features), dim=-1)
        repeated_features = features.repeat_interleave(repeats=num_parallel_samples, dim=0)

        # greedy decoding
        future_samples = []
        dec_attn_mask = torch.zeros(1, self.dec_max_seq_len)
        for k in range(self.config.prediction_length):
            lagged_sequence = self._origin_model.model.get_lagged_subsequences(
                sequence=repeated_past_values,
                subsequences_length=1 + k,
                shift=1,
            )

            lags_shape = lagged_sequence.shape
            reshaped_lagged_sequence = lagged_sequence.reshape(lags_shape[0], lags_shape[1], -1)
            decoder_input = torch.cat((reshaped_lagged_sequence, repeated_features[:, : k + 1]), dim=-1)

            dec_attn_mask[:, k] = 1
            dec_inputs_embeds = decoder_input[:, -1:]

            decoder_out = self.decoder(
                inputs_embeds=dec_inputs_embeds.contiguous(),
                attention_mask=dec_attn_mask,
                cache_position=torch.tensor(k, dtype=torch.int32),
            )
            params = decoder_out.params

            distr = self._origin_model.output_distribution(params, loc=repeated_loc, scale=repeated_scale)
            next_sample = distr.sample()

            repeated_past_values = torch.cat(
                (repeated_past_values, (next_sample - repeated_loc) / repeated_scale), dim=1
            )
            future_samples.append(next_sample)

        concat_future_samples = torch.cat(future_samples, dim=1)

        return SampleTSPredictionOutput(
            sequences=concat_future_samples.reshape(
                (-1, num_parallel_samples, self.config.prediction_length) + self._origin_model.target_shape,
            )
        )
