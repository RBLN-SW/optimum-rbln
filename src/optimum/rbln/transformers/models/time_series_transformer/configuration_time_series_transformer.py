from __future__ import annotations

from pydantic import Field

from ....configuration_utils import PositiveIntDefaultOne, RBLNModelConfig


class RBLNTimeSeriesTransformerForPredictionConfig(RBLNModelConfig):
    """
    Configuration class for RBLNTimeSeriesTransformerForPrediction.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Time Series Transformer models for time series forecasting tasks.
    """

    batch_size: PositiveIntDefaultOne = Field(default=1, description="The batch size for inference.")
    enc_max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the encoder.")
    dec_max_seq_len: int | None = Field(default=None, description="Maximum sequence length for the decoder.")
    num_parallel_samples: int | None = Field(
        default=None,
        description="Number of samples to generate in parallel during prediction.",
    )
