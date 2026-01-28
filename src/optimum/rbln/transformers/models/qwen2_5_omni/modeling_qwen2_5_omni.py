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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    DiTCodecEmbedding,
    DiTInputEmbedding,
    DiTTimestepEmbedding,
    Qwen2_5OmniDiTRotaryEmbedding,
    Qwen2_5OmniToken2WavBigVGANModel,
    Qwen2_5OmniToken2WavDiTModel,
)

from ....configuration_utils import RBLNCompileConfig
from ....modeling import RBLNModel
from ....utils.logging import get_logger
from .configuration_qwen2_5_omni import (
    RBLNQwen2_5OmniToken2WavBigVGANModelConfig,
    RBLNQwen2_5OmniToken2WavDiTModelConfig,
)
from .qwen2_5_omni_architecture import (
    Qwen2_5OmniToken2WavBigVGANWrapper,
    Qwen2_5OmniToken2WavDiTWrapper,
)


logger = get_logger(__name__)

if TYPE_CHECKING:
    from transformers import (
        AutoFeatureExtractor,
        AutoProcessor,
        AutoTokenizer,
    )


class RBLNQwen2_5OmniToken2WavModel(RBLNModel):
    """
    RBLN optimized Qwen2.5-Omni Token2Wav model.

    This class provides hardware-accelerated inference for Qwen2.5-Omni Token2Wav model
    on RBLN devices, which takes speech tokens as input and predicts mel spectrogram.
    """

    auto_model_class = None

    _rbln_submodules = [
        {"name": "code2wav_dit_model"},
        {"name": "code2wav_bigvgan_model"},
    ]

    def __post_init__(self, **kwargs):
        self.code2wav_dit_model = self.rbln_submodules[0]
        self.code2wav_bigvgan_model = self.rbln_submodules[1]

    def forward(
        self,
        code,
        conditioning,
        reference_mel,
        num_steps=10,
        guidance_scale=0.5,
        sway_coefficient=-1.0,
    ):
        mel_spectrogram = self.code2wav_dit_model.sample(
            conditioning_vector=conditioning,
            reference_mel_spectrogram=reference_mel,
            quantized_code=code,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            sway_coefficient=sway_coefficient,
        )

        waveform = self.code2wav_bigvgan_model(mel_spectrogram)

        return waveform


class RBLNQwen2_5OmniToken2WavDiTModel(RBLNModel):
    """
    RBLN optimized Qwen2.5-Omni Token2Wav DiT model.

    This class provides hardware-accelerated inference for Qwen2.5-Omni Token2Wav DiT model
    on RBLN devices, which takes speech tokens as input and predicts mel spectrogram.
    """

    auto_model_class = None
    _supports_non_fp32 = True

    def __post_init__(self, **kwargs):
        self.compiled_model = self.model[0]
        config = self.config

        self.mel_dim = config.mel_dim
        self.repeats = config.repeats
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.block_size = config.block_size

        with no_init_weights():
            self.text_embed = DiTCodecEmbedding(
                config.num_embeds,
                config.emb_dim,
                config.repeats,
            ).eval()
            self.time_embed = DiTTimestepEmbedding(config.hidden_size).eval()
            self.input_embed = DiTInputEmbedding(config).eval()
            self.rotary_embed = Qwen2_5OmniDiTRotaryEmbedding(config.head_dim).eval()

        artifacts = torch.load(
            self.model_save_dir / self.subfolder / "torch_artifacts.pth",
            weights_only=False,
        )
        self.text_embed.load_state_dict(artifacts["text_embed"])
        self.time_embed.load_state_dict(artifacts["time_embed"])
        self.input_embed.load_state_dict(artifacts["input_embed"])
        self.rotary_embed.load_state_dict(artifacts["rotary_embed"])

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen2_5OmniToken2WavDiTModel, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "Qwen2_5OmniToken2WavDiTModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwen2_5OmniToken2WavDiTModelConfig,
    ):
        save_dict = {
            "text_embed": model.text_embed.state_dict(),
            "time_embed": model.time_embed.state_dict(),
            "input_embed": model.input_embed.state_dict(),
            "rotary_embed": model.rotary_embed.state_dict(),
        }
        torch.save(save_dict, save_dir_path / subfolder / "torch_artifacts.pth")

    @classmethod
    def _wrap_model_if_needed(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNQwen2_5OmniToken2WavDiTModelConfig,
    ):
        return Qwen2_5OmniToken2WavDiTWrapper(model, rbln_config).eval()

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "PretrainedConfig" = None,
        rbln_config: Optional[RBLNQwen2_5OmniToken2WavDiTModelConfig] = None,
    ) -> RBLNQwen2_5OmniToken2WavDiTModelConfig:
        hidden_size = model_config.hidden_size
        head_dim = model_config.head_dim
        num_attention_heads = model_config.num_attention_heads
        max_seq_len = rbln_config.max_seq_len

        if max_seq_len is None:
            # Default max_seq_len = max_new_tokens * repeats
            # Hard coded for test (fix to 500)
            max_seq_len = 1000
            rbln_config.max_seq_len = max_seq_len

        # need to fix?
        # batch_size=2 for CFG (conditional + unconditional) after input_embed
        input_info = [
            ("hidden_states", [2, max_seq_len, hidden_size], rbln_config.dtype),
            ("time_embedding", [2, hidden_size], rbln_config.dtype),
            ("cos", [2, max_seq_len, head_dim], rbln_config.dtype),
            ("sin", [2, max_seq_len, head_dim], rbln_config.dtype),
            ("block_diff", [2, num_attention_heads, max_seq_len, max_seq_len], "int32"),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    def _create_block_diff(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Creates block difference matrix for attention masking."""
        batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        block_indices = torch.arange(seq_len, device=hidden_states.device) // self.block_size

        block_i = block_indices.unsqueeze(1)
        block_j = block_indices.unsqueeze(0)
        block_diff = block_j - block_i

        return block_diff.expand(batch, self.num_attention_heads, seq_len, seq_len)

    def _pad_to_max_seq_len(
        self,
        hidden_states: torch.Tensor,
        time_embedding: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        block_diff: torch.Tensor,
    ):
        max_seq_len = self.rbln_config.max_seq_len
        _, seq_len, _ = hidden_states.shape

        if seq_len == max_seq_len:
            return hidden_states, time_embedding, cos, sin, block_diff, seq_len

        # Pad hidden_states
        pad_len = max_seq_len - seq_len
        hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len), value=0)

        # Pad cos, sin
        cos = torch.nn.functional.pad(cos, (0, 0, 0, pad_len), value=0)
        sin = torch.nn.functional.pad(sin, (0, 0, 0, pad_len), value=0)

        # Pad block_diff
        block_diff = torch.nn.functional.pad(block_diff, (0, pad_len, 0, pad_len), value=0)

        return hidden_states, time_embedding, cos, sin, block_diff, seq_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition_vector: torch.Tensor,
        speaker_embedding: torch.Tensor,
        quantized_code: torch.Tensor,
        time_step: torch.Tensor,
        drop_audio_conditioning: bool = False,
        drop_code: bool = False,
        apply_cfg: bool = True,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        if time_step.ndim == 0:
            time_step = time_step.repeat(batch_size)

        time_embedding = self.time_embed(time_step)
        text_embedding = self.text_embed(quantized_code, drop_code=False if apply_cfg else drop_code)
        text_embedding_unconditioned = self.text_embed(quantized_code, drop_code=True) if apply_cfg else None

        hidden_states = self.input_embed(
            hidden_states,
            speaker_embedding,
            condition_vector,
            text_embedding,
            drop_audio_cond=drop_audio_conditioning,
            code_embed_uncond=text_embedding_unconditioned,
            apply_cfg=apply_cfg,
        )

        if apply_cfg:
            time_embedding = torch.cat([time_embedding, time_embedding], dim=0)

        cos, sin = self.rotary_embed(hidden_states)

        block_diff = self._create_block_diff(hidden_states)

        hidden_states, time_embedding, cos, sin, block_diff, original_seq_len = self._pad_to_max_seq_len(
            hidden_states, time_embedding, cos, sin, block_diff
        )

        output = self.compiled_model(
            hidden_states,
            time_embedding,
            cos,
            sin,
            block_diff.to(torch.int32),
        )

        output = output[:, :original_seq_len, :]

        return output

    @torch.no_grad()
    def sample(
        self,
        conditioning_vector: torch.Tensor,
        reference_mel_spectrogram: torch.Tensor,
        quantized_code: torch.Tensor,
        num_steps: int = 10,
        guidance_scale: float = 0.5,
        sway_coefficient: float = -1.0,
    ) -> torch.Tensor:
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import RungeKutta4ODESolver

        torch.manual_seed(0)
        noise_initialization = torch.randn(
            [1, 30000, self.mel_dim],
            dtype=reference_mel_spectrogram.dtype,
        )
        maximum_duration = quantized_code.shape[1] * self.repeats
        initial_state = noise_initialization[:, :maximum_duration].to(quantized_code.device)
        batch_size = reference_mel_spectrogram.shape[0]
        conditioning_vector = conditioning_vector.unsqueeze(1).repeat(1, maximum_duration, 1)

        if batch_size != 1:
            raise ValueError("Only batch size = 1 is currently supported")

        def ode_function(time_step, hidden_states):
            if guidance_scale < 1e-5:
                prediction = self(
                    hidden_states=hidden_states,
                    speaker_embedding=conditioning_vector,
                    condition_vector=reference_mel_spectrogram,
                    quantized_code=quantized_code,
                    time_step=time_step,
                    drop_audio_conditioning=False,
                    drop_code=False,
                )
                return prediction

            model_output = self(
                hidden_states=hidden_states,
                quantized_code=quantized_code,
                speaker_embedding=conditioning_vector,
                condition_vector=reference_mel_spectrogram,
                time_step=time_step,
                apply_cfg=True,
            )
            guided_prediction, null_prediction = torch.chunk(model_output, 2, dim=0)
            return guided_prediction + (guided_prediction - null_prediction) * guidance_scale

        initial_time = 0
        time_embedding = torch.linspace(
            initial_time,
            1,
            num_steps,
            device=quantized_code.device,
            dtype=conditioning_vector.dtype,
        )

        if sway_coefficient is not None:
            time_embedding += sway_coefficient * (torch.cos(torch.pi / 2 * time_embedding) - 1 + time_embedding)

        ode_solver = RungeKutta4ODESolver(function=ode_function, initial_value=initial_state)
        solution_trajectory = ode_solver.integrate(time_embedding)

        generated_waveform = solution_trajectory[-1]
        generated_mel_spectrogram = generated_waveform.permute(0, 2, 1)
        return generated_mel_spectrogram


class RBLNQwen2_5OmniToken2WavBigVGANModel(RBLNModel):
    """
    RBLN optimized Qwen2.5-Omni Token2Wav BigVGAN model (vocoder).

    This class provides hardware-accelerated inference for Qwen2.5-Omni Token2Wav BigVGAN model
    on RBLN devices, which converts mel spectrogram to audio waveform.
    """

    auto_model_class = None
    _supports_non_fp32 = True

    def __post_init__(self, **kwargs):
        self.compiled_model = self.model[0]
        config = self.config

        self.mel_dim = config.mel_dim
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsample_layers = len(config.upsample_rates)

        artifacts_path = Path(self.model_save_dir) / self.subfolder / "torch_artifacts.pth"
        if artifacts_path.exists():
            torch_artifacts = torch.load(artifacts_path, map_location="cpu", weights_only=True)

            from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
                SnakeBeta,
                TorchActivation1d,
            )

            self.conv_pre = torch.nn.Conv1d(config.mel_dim, config.upsample_initial_channel, 7, 1, padding=3)
            self.conv_pre.load_state_dict(torch_artifacts["conv_pre"])
            self.conv_pre.eval()

            final_channels = config.upsample_initial_channel // (2**self.num_upsample_layers)
            self.activation_post = TorchActivation1d(activation=SnakeBeta(final_channels))
            self.activation_post.load_state_dict(torch_artifacts["activation_post"])
            self.activation_post.eval()

            self.conv_post = torch.nn.Conv1d(final_channels, 1, 7, 1, padding=3, bias=False)
            self.conv_post.load_state_dict(torch_artifacts["conv_post"])
            self.conv_post.eval()

    def __getattr__(self, __name: str) -> Any:
        def redirect(func):
            return lambda *pargs, **kwargs: func(self, *pargs, **kwargs)

        val = getattr(Qwen2_5OmniToken2WavBigVGANModel, __name)

        if isinstance(val, Callable) and "self" in set(inspect.signature(val).parameters):
            return redirect(val)
        return val

    @classmethod
    def save_torch_artifacts(
        cls,
        model: "PreTrainedModel",
        save_dir_path: Path,
        subfolder: str,
        rbln_config: RBLNQwen2_5OmniToken2WavBigVGANModelConfig,
    ):
        torch_artifacts = {}

        torch_artifacts["conv_pre"] = model.conv_pre.state_dict()
        torch_artifacts["activation_post"] = model.activation_post.state_dict()
        torch_artifacts["conv_post"] = model.conv_post.state_dict()

        artifacts_path = save_dir_path / subfolder / "torch_artifacts.pth"
        torch.save(torch_artifacts, artifacts_path)

    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"],
        model: Optional["PreTrainedModel"] = None,
        model_config: "PretrainedConfig" = None,
        rbln_config: Optional[RBLNQwen2_5OmniToken2WavBigVGANModelConfig] = None,
    ) -> RBLNQwen2_5OmniToken2WavBigVGANModelConfig:
        max_mel_len = rbln_config.max_mel_len
        upsample_initial_channel = model_config.upsample_initial_channel

        if max_mel_len is None:
            # Default max_mel_len for testing
            max_mel_len = 1000
            rbln_config.max_mel_len = max_mel_len

        input_info = [
            ("hidden_representation", [1, upsample_initial_channel, max_mel_len], rbln_config.dtype),
        ]

        rbln_compile_config = RBLNCompileConfig(input_info=input_info)
        rbln_config.set_compile_cfgs([rbln_compile_config])

        return rbln_config

    @classmethod
    def _wrap_model_if_needed(
        cls,
        model: "PreTrainedModel",
        rbln_config: RBLNQwen2_5OmniToken2WavBigVGANModelConfig,
    ):
        return Qwen2_5OmniToken2WavBigVGANWrapper(model, rbln_config)

    def _pad_to_max_mel_len(
        self,
        tensor: torch.Tensor,
    ):
        max_mel_len = self.rbln_config.max_mel_len
        _, _, seq_len = tensor.shape

        if seq_len == max_mel_len:
            return tensor, seq_len

        pad_len = max_mel_len - seq_len
        tensor = torch.nn.functional.pad(tensor, (0, pad_len), value=0)

        return tensor, seq_len

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        original_mel_len = mel_spectrogram.shape[-1]
        processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(processed_spectrogram)
        hidden_representation, _ = self._pad_to_max_mel_len(hidden_representation)

        hidden_representation = self.compiled_model(hidden_representation)

        upsample_rates = self.config.upsample_rates
        total_upsample = 1
        for rate in upsample_rates:
            total_upsample *= rate
        original_output_len = original_mel_len * total_upsample

        hidden_representation = hidden_representation[..., :original_output_len].cpu()
        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        output_waveform = torch.clamp(output_waveform, min=-1.0, max=1.0)

        return output_waveform.squeeze().cpu()
