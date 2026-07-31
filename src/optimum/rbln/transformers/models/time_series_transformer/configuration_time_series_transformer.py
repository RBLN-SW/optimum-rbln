from typing import Any

from ....configuration_utils import RBLNModelConfig


class RBLNTimeSeriesTransformerForPredictionConfig(RBLNModelConfig):
    """
    Configuration class for RBLNTimeSeriesTransformerForPrediction.

    This configuration class stores the configuration parameters specific to
    RBLN-optimized Time Series Transformer models for time series forecasting tasks.
    """

    def __init__(
        self,
        batch_size: int | None = None,
        enc_max_seq_len: int | None = None,
        dec_max_seq_len: int | None = None,
        num_parallel_samples: int | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            batch_size (int | None): The batch size for inference. Defaults to 1.
            enc_max_seq_len (int | None): Maximum sequence length for the encoder.
            dec_max_seq_len (int | None): Maximum sequence length for the decoder.
            num_parallel_samples (int | None): Number of samples to generate in parallel during prediction.
            kwargs: Additional arguments passed to the parent RBLNModelConfig.

        Raises:
            ValueError: If batch_size is not a positive integer.
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size or 1
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}")

        self.enc_max_seq_len = enc_max_seq_len
        self.dec_max_seq_len = dec_max_seq_len
        self.num_parallel_samples = num_parallel_samples
