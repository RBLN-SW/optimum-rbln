from __future__ import annotations

from pydantic import Field, field_validator

from ....configuration_utils import RBLNModelConfig


class RBLNTimeSeriesTransformerForPredictionConfig(RBLNModelConfig):
    """
    Configuration class for RBLNTimeSeriesTransformerForPrediction.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Time Series Transformer models for time series forecasting tasks.
    """

    batch_size: int = Field(default=1, description="The batch size for inference.")
    enc_max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the encoder.")
    dec_max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the decoder.")
    num_parallel_samples: int | None = Field(
        default=None,
        description="Number of samples to generate in parallel during prediction.",
    )

    @field_validator("batch_size", mode="before")
    @classmethod
    def validate_batch_size(cls, v: int | None) -> int:
        if v is None:
            return 1
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {v}")
        return v
