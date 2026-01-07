# Copyright 2025 Rebellions Inc. All rights reserved.
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

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import rebel
import torch
from rebel.compile_context import CompileContext
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.generation.utils import GenerationMixin

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from ....utils.runtime_utils import RBLNPytorchRuntime
from ...modeling_outputs import RBLNDecoderOnlyOutput
from .configuration_mamba2 import RBLNMamba2ForCausalLMConfig
from .mamba2_architecture import Mamba2StepWrapper


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer


def _patch_hf_mamba2_zero_mlp_splits(hf_model: PreTrainedModel):
    """
    Work around TVM frontend limitations with `aten.split_with_sizes` when some split sizes are 0.

    HF Mamba2 may compute `d_mlp == 0` and still do:
        projected_states.split([0, 0, intermediate, conv_dim, num_heads], dim=-1)
    which produces a split indices list with duplicates and triggers:
        tvm.error.InternalError: indices_or_sections need to be a sorted ascending list

    This patch rewrites the split logic for `d_mlp == 0` to avoid zero-sized splits.
    """
    from transformers.models.mamba2 import modeling_mamba2 as hf  # local import to match installed version

    Mamba2Mixer = hf.Mamba2Mixer

    def _safe_split_projected(self, projected_states: torch.Tensor, d_mlp: int):
        if d_mlp == 0:
            gate, hidden_states_B_C, dt = projected_states.split(
                [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
            )
            return gate, hidden_states_B_C, dt
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )
        return gate, hidden_states_B_C, dt

    def patched_torch_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # 1. Gated MLP's linear projection
        hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        gate, hidden_states_B_C, dt = _safe_split_projected(self, projected_states, int(d_mlp))

        # 2. Convolution sequence transformation
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=hidden_states_B_C, cache_init=False)

            conv_states = cache_params.conv_states[self.layer_idx].to(device=self.conv1d.weight.device)
            hidden_states_B_C = torch.sum(conv_states * self.conv1d.weight.squeeze(1), dim=-1)
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            if cache_params is not None:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                conv_states = torch.nn.functional.pad(
                    hidden_states_B_C_transposed,
                    (cache_params.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                )
                cache_params.update_conv_state(layer_idx=self.layer_idx, new_conv_state=conv_states, cache_init=True)

            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2)
            )

        hidden_states_B_C = hf.apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())
        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            cache_device = cache_params.ssm_states.device

            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)

            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = (
                A[..., None, None]
                .expand(self.num_heads, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            dB = dt[..., None] * B[..., None, :]

            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)

            cache_params.update_ssm_state(
                layer_idx=self.layer_idx,
                new_ssm_state=cache_params.ssm_states[self.layer_idx] * dA + dBx,
            )

            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])

            ssm_states = cache_params.ssm_states[self.layer_idx].to(device=C.device, dtype=C.dtype)
            ssm_states_reshaped = ssm_states.view(batch_size * self.num_heads, self.head_dim, self.ssm_state_size)
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)

            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            dt = torch.nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            C = C.repeat_interleave(self.num_heads // self.n_groups, dim=2, output_size=self.num_heads)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * hf.pad_tensor_by_size(hidden_states, pad_size)

            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            hidden_states, A, B, C = [
                hf.reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)
            ]

            A = A.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A, dim=-1)

            L = torch.exp(hf.segment_sum(A))

            G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
            G = G_intermediate.sum(dim=-1)

            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)

            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            if cache_params is not None and cache_position is not None and cache_position[0] > 0:
                previous_states = cache_params.ssm_states[self.layer_idx][:, None, ...].to(device=states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])
            states = torch.cat([previous_states, states], dim=1)
            decay_chunk = torch.exp(hf.segment_sum(torch.nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
            decay_chunk = decay_chunk.transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

            y = Y_diag + Y_off
            y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
            y = y + D_residual
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

            if ssm_state is not None and cache_params is not None:
                cache_params.update_ssm_state(layer_idx=self.layer_idx, new_ssm_state=ssm_state)

        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states

    def patched_cuda_kernels_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        hidden_states = hf.apply_mask_to_padding_states(hidden_states, attention_mask)
        projected_states = self.in_proj(hidden_states)

        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        if cache_params is not None and cache_position is not None and cache_position[0] > 0:
            ps = projected_states.squeeze(1)
            gate, hidden_states_B_C, dt = _safe_split_projected(self, ps, int(d_mlp))
            hidden_states_B_C = hf.causal_conv1d_update(
                hidden_states_B_C,
                cache_params.conv_states[self.layer_idx],
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self.D[:, None, ...].expand(-1, self.head_dim)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = hf.selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            hidden_states = self.norm(hidden_states, gate)
            out = self.out_proj(hidden_states)[:, None, ...]
            return out

        # Fallback to original implementation for non-single-step case (fast path usually unavailable in our env).
        return hf.Mamba2Mixer.cuda_kernels_forward(self, hidden_states, cache_params, cache_position, attention_mask)

    # NOTE: legacy patch retained for reference; current RBLN path uses `Mamba2StepWrapper`
    # and does not call HF Mamba2Mixer.forward at all.
    for _module in hf_model.modules():
        if isinstance(_module, Mamba2Mixer):
            _module.torch_forward = types.MethodType(patched_torch_forward, _module)
            _module.cuda_kernels_forward = types.MethodType(patched_cuda_kernels_forward, _module)


class RBLNMambaGenerationMixin(GenerationMixin):
    """
    Minimal GenerationMixin-compatible helpers (mirrors decoderonly logic, without inputs_embeds support).
    """

    _supports_cache_class = False  # transformers.GenerationMixin expects this attr
    _is_stateful = False

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        generate_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_inputs: Dict[str, torch.Tensor] = {}
        is_prefill = generate_idx is None

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if is_prefill:
            generate_idx = attention_mask.sum(dim=-1, keepdim=True).int()
            padded_cache_lengths = torch.zeros_like(generate_idx)
            cache_position = None
        else:
            input_ids = input_ids[:, -1:]
            cache_position = generate_idx + padded_cache_lengths if padded_cache_lengths is not None else generate_idx
            cache_position = cache_position.reshape(-1).to(dtype=torch.int32)
            generate_idx = generate_idx + 1

        model_inputs.update(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "generate_idx": generate_idx,
                "padded_cache_lengths": padded_cache_lengths,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs: RBLNDecoderOnlyOutput, model_kwargs: Dict[str, torch.Tensor], **kwargs
    ):
        model_kwargs["generate_idx"] = outputs.generate_idx
        model_kwargs["padded_cache_lengths"] = outputs.padded_cache_lengths
        return model_kwargs


class RBLNMamba2RuntimeModel(RBLNPytorchRuntime):
    """
    Minimal runtime wrapper for compiled Mamba2 graphs.

    This intentionally stays small; unlike decoder-only attention models, Mamba2 does not
    require paged KV cache tables. We expose cache tensors (conv/ssm states) explicitly.
    """

    def forward(
        self,
        *args,
        out: Optional[List[torch.Tensor]] = None,
    ):
        # Keep argument order consistent with compile input order.
        return super().forward(*args, out=out)


class RBLNMamba2ForCausalLM(RBLNModel, RBLNMambaGenerationMixin):
    """
    RBLN optimized wrapper for HuggingFace `Mamba2ForCausalLM`.

    This follows the high-level structure of `RBLNDecoderOnlyModelForCausalLM`, but uses
    SSM cache tensors:
      - conv_states: [num_layers, batch, conv_dim, conv_kernel]
      - ssm_states:  [num_layers, batch, num_heads, head_dim, state_size]
    """

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"

    @classmethod
    def get_hf_class(cls):
        # AutoModelForCausalLM resolves Mamba2ForCausalLM based on config.
        return cls.auto_model_class

    @classmethod
    def _wrap_model_if_needed(cls, model: PreTrainedModel, rbln_config: RBLNMamba2ForCausalLMConfig):
        # Not used: we compile separate wrappers for prefill/decode.
        return model

    @staticmethod
    def _get_cache_shapes(
        model_config: PretrainedConfig, batch_size: int
    ) -> Tuple[List[int], List[int], int, int]:
        num_layers = int(getattr(model_config, "num_hidden_layers"))
        hidden_size = int(getattr(model_config, "hidden_size"))
        expand = float(getattr(model_config, "expand"))
        intermediate_size = int(expand * hidden_size)
        state_size = int(getattr(model_config, "state_size"))
        n_groups = int(getattr(model_config, "n_groups"))
        num_heads = int(getattr(model_config, "num_heads"))
        head_dim = int(getattr(model_config, "head_dim"))
        conv_kernel = int(getattr(model_config, "conv_kernel"))

        conv_dim = intermediate_size + 2 * n_groups * state_size
        conv_states_shape = [num_layers, batch_size, conv_dim, conv_kernel]
        ssm_states_shape = [num_layers, batch_size, num_heads, head_dim, state_size]
        return conv_states_shape, ssm_states_shape, num_layers, conv_dim

    @classmethod
    def _get_compile_context(cls, compile_config: RBLNCompileConfig, example_inputs: Tuple[torch.Tensor, ...]):
        context = CompileContext(use_weight_sharing=True)

        static_tensors: Dict[str, torch.Tensor] = {}
        for (name, _, _), tensor in zip(compile_config.input_info, example_inputs):
            if name in {"conv_states", "ssm_states"}:
                static_tensors[name] = tensor
                context.mark_static_address(tensor, name)
        return context, static_tensors

    @classmethod
    @torch.inference_mode()
    def get_compiled_model(cls, model: PreTrainedModel, rbln_config: RBLNMamba2ForCausalLMConfig):
        step_model = Mamba2StepWrapper(model).eval()

        prefill_compile_config = rbln_config.compile_cfgs[0]
        prefill_example_inputs = prefill_compile_config.get_dummy_inputs(fill=0)
        context, static_tensors = cls._get_compile_context(prefill_compile_config, prefill_example_inputs)

        compiled_models: Dict[str, rebel.RBLNCompiledModel] = {}
        compiled_models["prefill_step"] = cls.compile(
            step_model,
            rbln_compile_config=prefill_compile_config,
            create_runtimes=rbln_config.create_runtimes,
            device=rbln_config.device,
            example_inputs=prefill_example_inputs,
            compile_context=context,
        )

        if rbln_config.can_generate:
            for batch_size, dec_compile_config in zip(rbln_config.decoder_batch_sizes, rbln_config.compile_cfgs[1:]):
                dec_example_inputs = dec_compile_config.get_dummy_inputs(fill=0, static_tensors=static_tensors)
                compiled_models[f"decoder_step_batch_{batch_size}"] = cls.compile(
                    step_model,
                    rbln_compile_config=dec_compile_config,
                    create_runtimes=rbln_config.create_runtimes,
                    device=rbln_config.device,
                    example_inputs=dec_example_inputs,
                    compile_context=context,
                )

        return compiled_models

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Optional[Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"]] = None,
        model: Optional[PreTrainedModel] = None,
        model_config: Optional[PretrainedConfig] = None,
        rbln_config: Optional[RBLNMamba2ForCausalLMConfig] = None,
    ) -> RBLNMamba2ForCausalLMConfig:
        if rbln_config.max_seq_len is None:
            raise ValueError(
                "`max_seq_len` must be specified for Mamba2 compilation (HF Mamba2Config has no max position limit)."
            )

        # Derive cache shapes from HF config.
        conv_states_shape, ssm_states_shape, _, _ = cls._get_cache_shapes(
            model_config=model_config, batch_size=rbln_config.batch_size
        )

        # Compile only STEP graph(s): we will implement prefill by looping token-by-token in python,
        # similar to decoderonly. This avoids the HF full-seq (chunked scan) path that often fails to compile.
        #
        # Prefill graph is still a step graph, but fixed to batch=1 (like decoderonly's prefill),
        # enabling prefill for arbitrary runtime batch by iterating batch_idx.
        conv_states_shape_1, ssm_states_shape_1, _, _ = cls._get_cache_shapes(model_config=model_config, batch_size=1)

        prefill_input_info = [
            ("input_ids", [1, 1], "int64"),
            ("conv_states", conv_states_shape_1, rbln_config.dtype),
            ("ssm_states", ssm_states_shape_1, rbln_config.dtype),
        ]
        compile_cfgs: List[RBLNCompileConfig] = [
            RBLNCompileConfig(compiled_model_name="prefill_step", input_info=prefill_input_info),
        ]

        # Decode graphs (token-by-token, batch in decoder_batch_sizes).
        if rbln_config.can_generate:
            for batch_size in rbln_config.decoder_batch_sizes:
                conv_states_shape_b, ssm_states_shape_b, _, _ = cls._get_cache_shapes(
                    model_config=model_config, batch_size=batch_size
                )
                dec_input_info = [
                    ("input_ids", [batch_size, 1], "int64"),
                    ("conv_states", conv_states_shape_b, rbln_config.dtype),
                    ("ssm_states", ssm_states_shape_b, rbln_config.dtype),
                ]
                compile_cfgs.append(
                    RBLNCompileConfig(compiled_model_name=f"decoder_step_batch_{batch_size}", input_info=dec_input_info)
                )

        rbln_config.set_compile_cfgs(compile_cfgs)
        return rbln_config

    @classmethod
    def _create_runtimes(
        cls,
        compiled_models: List[rebel.RBLNCompiledModel],
        rbln_config: RBLNMamba2ForCausalLMConfig,
    ) -> List[rebel.Runtime]:
        expected_model_names = ["prefill_step"]
        if rbln_config.can_generate:
            expected_model_names.extend([f"decoder_step_batch_{b}" for b in rbln_config.decoder_batch_sizes])

        if any(model_name not in rbln_config.device_map for model_name in expected_model_names):
            cls._raise_missing_compiled_file_error(expected_model_names)

        runtimes: List[rebel.Runtime] = [
            rebel.Runtime(
                compiled_models[0],
                tensor_type="pt",
                device=rbln_config.device_map["prefill_step"],
                activate_profiler=rbln_config.activate_profiler,
                timeout=rbln_config.timeout,
            )
        ]

        if rbln_config.can_generate:
            runtimes.extend(
                [
                    rebel.Runtime(
                        compiled_models[i + 1],
                        tensor_type="pt",
                        device=rbln_config.device_map[f"decoder_step_batch_{batch_size}"],
                        activate_profiler=rbln_config.activate_profiler,
                        timeout=rbln_config.timeout,
                    )
                    for i, batch_size in enumerate(rbln_config.decoder_batch_sizes)
                ]
            )

        return runtimes

    def __post_init__(self, **kwargs):
        self._setup_runtime()

    def can_generate(self) -> bool:
        return bool(getattr(self.rbln_config, "can_generate", False))

    def _setup_runtime(self):
        # Cache tensors live on host and are passed into runtimes explicitly.
        self._conv_states: Optional[torch.Tensor] = None
        self._ssm_states: Optional[torch.Tensor] = None

        self.prefill_step = RBLNMamba2RuntimeModel(runtime=self.model[0])
        self.decoders: Dict[int, RBLNMamba2RuntimeModel] = {}
        if self.rbln_config.can_generate:
            for i, bs in enumerate(self.rbln_config.decoder_batch_sizes):
                self.decoders[bs] = RBLNMamba2RuntimeModel(runtime=self.model[i + 1])
            self.decoder = self.decoders[self.rbln_config.batch_size]
        else:
            self.decoder = None

    def reset_cache(self):
        if self._conv_states is not None:
            self._conv_states.zero_()
        if self._ssm_states is not None:
            self._ssm_states.zero_()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generate_idx: Optional[torch.Tensor] = None,
        padded_cache_lengths: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("`input_ids` is required for RBLNMamba2ForCausalLM.")

        batch_size, seq_len = input_ids.shape
        if batch_size > self.rbln_config.batch_size:
            raise ValueError(
                f"Input batch({batch_size}) exceeds compiled batch_size({self.rbln_config.batch_size})."
            )

        if generate_idx is None:
            generate_idx = (
                attention_mask.sum(dim=-1, keepdim=True).int()
                if attention_mask is not None
                else torch.full((batch_size, 1), seq_len, dtype=torch.int32)
            )
            padded_cache_lengths = torch.zeros_like(generate_idx)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=self.dtype)
        else:
            attention_mask = attention_mask.to(dtype=self.dtype)

        # Prefill (no cache_position from generation mixin): run step-by-step to avoid compiling full-seq graph.
        if cache_position is None:
            # Allocate caches for this batch size (one-time).
            if self._conv_states is None or self._ssm_states is None or self._conv_states.shape[1] != batch_size:
                conv_shape, ssm_shape, _, _ = self._get_cache_shapes(self.config, batch_size)
                self._conv_states = torch.zeros(*conv_shape, dtype=self.dtype)
                self._ssm_states = torch.zeros(*ssm_shape, dtype=self.dtype)
            else:
                self.reset_cache()

            # Run per-batch-item prefill using the batch=1 compiled graph, writing into slice caches.
            vocab_size = int(getattr(self.config, "vocab_size"))
            logits = torch.zeros(batch_size, seq_len, vocab_size, dtype=torch.float32)
            for b_idx in range(batch_size):
                # views into cache tensors for this batch item
                conv_view = self._conv_states[:, b_idx : b_idx + 1]
                ssm_view = self._ssm_states[:, b_idx : b_idx + 1]
                # ensure cleared
                conv_view.zero_()
                ssm_view.zero_()
                for t in range(seq_len):
                    # Skip padded tokens if attention_mask exists and is 0
                    if attention_mask is not None and attention_mask[b_idx, t].item() == 0:
                        continue
                    token_ids = input_ids[b_idx : b_idx + 1, t : t + 1]
                    cp = torch.tensor([t], dtype=torch.int32)
                    am = torch.ones(1, 1, dtype=self.dtype)
                    step_logits = self.prefill_step(token_ids, conv_view, ssm_view)[0]
                    logits[b_idx, t : t + 1].copy_(step_logits)
        else:
            # Decode (token-by-token)
            if self._conv_states is None or self._ssm_states is None:
                raise RuntimeError("Cache is not initialized. Run prefill first (cache_position=None).")
            if batch_size not in self.decoders:
                raise ValueError(
                    f"No decoder runtime available for batch size {batch_size}. "
                    f"Available batch sizes are: {list(self.decoders.keys())}."
                )
            if input_ids.shape[1] != 1:
                # GenerationMixin already slices input_ids to last token, but guard anyway.
                input_ids = input_ids[:, -1:]
                attention_mask = attention_mask[:, -1:]
                cache_position = cache_position[:, -1:]

            logits = self.decoders[batch_size](
                input_ids,
                self._conv_states,
                self._ssm_states,
            )
            logits = logits[0]

        if not return_dict:
            return logits, generate_idx, padded_cache_lengths, None

        return RBLNDecoderOnlyOutput(
            logits=logits,
            generate_idx=generate_idx,
            padded_cache_lengths=padded_cache_lengths,
            hidden_states=None,
        )


