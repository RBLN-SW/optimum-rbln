# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model, Chronos2Output

from ..configuration_utils import RBLNCompileConfig
from ..modeling import RBLNModel
from .chronos2_architecture import Chronos2ModelWrapper, Chronos2Preprocessor
from .configuration_chronos2 import RBLNChronos2ModelConfig


class RBLNChronos2Model(RBLNModel):
    """Chronos-2 forecasting with native CPU preprocessing and all learned layers on the NPU.

    Context and batch inputs are padded to the compiled capacities. Forecasts use
    exactly the compiled number of future patches; compile a separate configuration
    to change that count. Install ``optimum-rbln[chronos]`` for this model.
    """

    rbln_config: RBLNChronos2ModelConfig
    auto_model_class = Chronos2Model
    _hf_class = Chronos2Model
    _supports_non_fp32 = False

    @classmethod
    def _update_rbln_config(cls, preprocessors, model, model_config, rbln_config):
        native = model_config.chronos_config
        context_length = rbln_config.context_length or native["context_length"]
        max_prediction = native["max_output_patches"] * native["output_patch_size"]
        prediction_length = rbln_config.prediction_length or max_prediction
        if context_length > native["context_length"]:
            raise ValueError("context_length exceeds the native model maximum")
        if prediction_length > max_prediction:
            raise ValueError("prediction_length exceeds the native model maximum")
        patch_size, stride = native["input_patch_size"], native["input_patch_stride"]
        padded_context = math.ceil(context_length / patch_size) * patch_size
        context_patches = (padded_context - patch_size) // stride + 1
        future_patches = math.ceil(prediction_length / native["output_patch_size"])
        rbln_config.context_length = context_length
        rbln_config.prediction_length = future_patches * native["output_patch_size"]
        batch = rbln_config.batch_size
        rbln_config.set_compile_cfgs(
            [
                RBLNCompileConfig(
                    input_info=[
                        ("patched_context", [batch, context_patches, 3 * patch_size], "float32"),
                        ("context_mask", [batch, context_patches], "bool"),
                        ("patched_future", [batch, future_patches, 3 * native["output_patch_size"]], "float32"),
                        ("group_ids", [batch], "int64"),
                    ]
                )
            ]
        )
        return rbln_config

    @classmethod
    def _wrap_model_if_needed(cls, model, rbln_config):
        return Chronos2ModelWrapper(model).eval()

    def __post_init__(self, **kwargs):
        assert self.rbln_config.context_length is not None
        assert self.rbln_config.prediction_length is not None
        self.context_length = self.rbln_config.context_length
        self.prediction_length = self.rbln_config.prediction_length
        self.config = Chronos2CoreConfig.from_dict(self.config.to_dict())
        self.preprocessor = Chronos2Preprocessor(self.config, self.context_length)
        self.chronos_config = self.preprocessor.chronos_config
        self.chronos_config.max_output_patches = self.prediction_length // self.chronos_config.output_patch_size
        self.num_quantiles = len(self.chronos_config.quantiles)
        self.quantiles = torch.tensor(self.chronos_config.quantiles, dtype=torch.float32)

    def forward(
        self,
        context,
        context_mask=None,
        group_ids=None,
        future_covariates=None,
        future_covariates_mask=None,
        num_output_patches=None,
        future_target=None,
        future_target_mask=None,
        output_attentions=False,
        return_dict=True,
    ):
        if future_target is not None or future_target_mask is not None or output_attentions:
            raise ValueError("Training targets and attention outputs are not supported during compiled inference")
        compiled_patches = self.chronos_config.max_output_patches
        if num_output_patches is not None and num_output_patches != compiled_patches:
            raise ValueError(f"num_output_patches must match the compiled value {compiled_patches}")
        self.preprocessor._validate_input(
            context,
            context_mask,
            group_ids,
            future_covariates,
            future_covariates_mask,
            compiled_patches,
            None,
            None,
        )
        actual_batch = context.shape[0]
        batch = self.rbln_config.batch_size
        if actual_batch < 1 or actual_batch > batch:
            raise ValueError(f"Input batch size must be between 1 and the compiled batch size {batch}")
        context_length = self.context_length
        context = context[:, -context_length:].to(device="cpu", dtype=torch.float32)
        left_pad = context_length - context.shape[-1]
        context = torch.nn.functional.pad(context, (left_pad, 0, 0, batch - actual_batch), value=float("nan"))
        if context_mask is not None:
            context_mask = context_mask[:, -context_length:].to(device="cpu", dtype=torch.float32)
            context_mask = torch.nn.functional.pad(context_mask, (left_pad, 0, 0, batch - actual_batch))
        if future_covariates is not None:
            future_covariates = future_covariates.to(device="cpu", dtype=torch.float32)
            future_covariates = torch.nn.functional.pad(
                future_covariates, (0, 0, 0, batch - actual_batch), value=float("nan")
            )
        if future_covariates_mask is not None:
            future_covariates_mask = future_covariates_mask.to(device="cpu", dtype=torch.float32)
            future_covariates_mask = torch.nn.functional.pad(future_covariates_mask, (0, 0, 0, batch - actual_batch))
        padded_groups = torch.arange(batch, dtype=torch.int64)
        if group_ids is not None:
            if group_ids.dtype != torch.int64:
                raise ValueError("group_ids must use torch.int64")
            _, inverse = torch.unique(group_ids.cpu(), return_inverse=True)
            padded_groups[:actual_batch] = inverse
        patched_context, mask, loc_scale = self.preprocessor._prepare_patched_context(context, context_mask)
        patched_future, _ = self.preprocessor._prepare_patched_future(
            future_covariates,
            future_covariates_mask,
            loc_scale,
            compiled_patches,
            batch,
        )
        normalized = self.model[0](patched_context, mask, patched_future, padded_groups)
        if isinstance(normalized, (list, tuple)):
            normalized = normalized[0].float() + normalized[1].float()
        prediction = self.preprocessor.instance_norm.inverse(normalized.flatten(1), loc_scale).reshape(
            batch,
            self.num_quantiles,
            self.prediction_length,
        )[:actual_batch]
        if not return_dict:
            return (prediction,)
        return Chronos2Output(quantile_preds=prediction)


__all__ = ["RBLNChronos2Model"]
