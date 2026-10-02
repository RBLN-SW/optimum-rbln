"""The Cosmos text guardrail is sized to the pipeline's prompt budget, not to a fixed 512.

cosmos_guardrail 0.3.1 reports any guardrail exception as "safe", so a Qwen3Guard compiled shorter
than the prompts the pipeline accepts never actually runs.
"""

import pytest

from optimum.rbln.diffusers.configurations.pipelines.configuration_cosmos import (
    RBLNCosmos2_5_PredictBasePipelineConfig,
    RBLNCosmos2_5_TransferPipelineConfig,
    RBLNCosmos2TextToImagePipelineConfig,
    RBLNCosmosTextToWorldPipelineConfig,
)
from optimum.rbln.diffusers.pipelines.cosmos.configuration_cosmos_guardrail import RBLNCosmosSafetyCheckerConfig
from optimum.rbln.diffusers.pipelines.cosmos.cosmos_guardrail import guardrail_max_seq_len


PIPELINES = [
    RBLNCosmosTextToWorldPipelineConfig,
    RBLNCosmos2TextToImagePipelineConfig,
    RBLNCosmos2_5_PredictBasePipelineConfig,
    RBLNCosmos2_5_TransferPipelineConfig,
]


class _Tokenizer:
    """Wraps a prompt in a fixed-size template, like Qwen3Guard-Gen's safety-policy chat template."""

    def __init__(self, template_tokens: int):
        self.template_tokens = template_tokens

    def apply_chat_template(self, messages, tokenize):
        assert not tokenize and messages == [{"role": "user", "content": ""}]
        return " ".join(["policy"] * self.template_tokens)

    def __call__(self, text):
        class Encoding:
            input_ids = text.split()

        return Encoding()


@pytest.mark.parametrize("pipeline_cls", PIPELINES)
def test_the_pipeline_prompt_budget_reaches_the_safety_checker(pipeline_cls):
    assert pipeline_cls(max_seq_len=1024).safety_checker.max_seq_len == 1024
    assert pipeline_cls().safety_checker.max_seq_len == pipeline_cls().text_encoder.max_seq_len == 512


def test_qwen3guard_is_sized_at_export_unless_set_explicitly():
    assert RBLNCosmosSafetyCheckerConfig(max_seq_len=512).qwen3guard.max_seq_len is None
    explicit = RBLNCosmosSafetyCheckerConfig(max_seq_len=512, qwen3guard={"max_seq_len": 2048})
    assert explicit.qwen3guard.max_seq_len == 2048


def test_guardrail_max_seq_len_adds_the_template_and_rounds_to_64():
    assert guardrail_max_seq_len(_Tokenizer(template_tokens=297), prompt_budget=512) == 832
    assert guardrail_max_seq_len(_Tokenizer(template_tokens=256), prompt_budget=512) == 768
