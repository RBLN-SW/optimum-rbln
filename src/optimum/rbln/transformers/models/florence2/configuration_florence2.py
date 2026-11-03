# Copyright 2026 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from typing import Any

from ..seq2seq import RBLNModelForSeq2SeqLMConfig


class RBLNFlorence2ForConditionalGenerationConfig(RBLNModelForSeq2SeqLMConfig):
    """Compilation settings for Florence-2 image-conditioned text generation.

    ``enc_max_seq_len`` includes both image features and text tokens. Image size
    is inferred from the processor unless explicitly supplied. Cache layouts
    follow the common Seq2Seq compilation and paged-attention paths.
    """

    support_paged_attention = True

    def __init__(self, image_size: int | None = None, **kwargs: Any):
        """
        Args:
            image_size (int | None): Square input image size in pixels. Inferred from
                the image processor when omitted. A supplied processor must use the same size.
            kwargs: Common Seq2Seq compilation and runtime settings.
        """
        super().__init__(**kwargs)
        self.image_size = image_size


__all__ = ["RBLNFlorence2ForConditionalGenerationConfig"]
