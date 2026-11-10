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
