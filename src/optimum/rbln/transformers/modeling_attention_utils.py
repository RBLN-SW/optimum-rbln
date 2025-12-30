import math
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import rebel

from ..utils.logging import get_logger
from ..utils.runtime_utils import get_available_dram
from .models.decoderonly.configuration_decoderonly import RBLNDecoderOnlyModelForCausalLMConfig


logger = get_logger()

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


DEFAULT_FLASH_ATTN_PARTITION_LENGTH = 16_384
DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH = 32_768
MIN_FLASH_ATTN_MAX_SEQ_LEN = 8_192
MIN_FLASH_ATTN_PARTITION_LENGTH = 4_096
MAX_FLASH_ATTN_PARTITION_LENGTH = 32_768
MAX_SLIDING_WINDOW_SIZE = 32_768


def set_default_values(
    attn_impl: Optional[str] = None,
    kvcache_partition_len: Optional[int] = None,
    kvcache_block_size: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[str, int, int]:
    if attn_impl is None:
        attn_impl = "eager"

    if kvcache_partition_len is not None:
        if attn_impl == "eager":
            attn_impl = "flash_attn"
            logger.warning(
                "A non-null `kvcache_partition_len` was provided, but `attn_impl` was not explicitly set or "
                "set to 'eager'. Since KV cache partitioning is only supported with flash attention, "
                "`attn_impl` has been automatically switched to 'flash_attn'."
            )

    if kvcache_partition_len is None and attn_impl == "flash_attn":
        kvcache_partition_len = DEFAULT_FLASH_ATTN_PARTITION_LENGTH

    if kvcache_block_size is None:
        if attn_impl == "eager":
            kvcache_block_size = max_seq_len
        else:
            kvcache_block_size = kvcache_partition_len

    return attn_impl, kvcache_partition_len, kvcache_block_size


def validate_attention_method(attn_impl: str, kvcache_partition_len: int, kvcache_block_size: int, max_seq_len: int):
    if attn_impl not in ["eager", "flash_attn"]:
        raise ValueError(f"Unknown `attn_impl` : {attn_impl}. (Available : 'eager', 'flash_attn`)")

    ## Checking Constraints...
    # Constraint of eager attention:
    # - `max_seq_len` <= 32k

    # Constraints of flash attention:
    # 1. `max_seq_len` should be multiple of `partition_len`.
    # 2. 4k <= `partition_len` <= 32k.
    # 3. `max_seq_len` should be larger then 8k.
    if attn_impl == "eager" and max_seq_len > DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH:
        raise ValueError(
            f"`max_seq_len` is set to {max_seq_len}, "
            f"which exceeds the limit of {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} for 'eager' attention. "
            f"Please reduce the `max_seq_len` to {DEFAULT_MAX_EAGER_ATTN_SEQUENCE_LENGTH} or lower,"
            " or consider switching `attn_impl` to 'flash_attn' for larger sequence lengths."
        )

    if attn_impl == "flash_attn":
        if max_seq_len // kvcache_partition_len < 2 or max_seq_len % kvcache_partition_len != 0:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) must be a multiple of `kvcache_partition_len` ({kvcache_partition_len}) "
                f"when using 'flash_attn'. Please adjust either value to meet this requirement."
            )
        elif not (MIN_FLASH_ATTN_PARTITION_LENGTH <= kvcache_partition_len <= MAX_FLASH_ATTN_PARTITION_LENGTH):
            raise ValueError(
                f"`kvcache_partition_len` ({kvcache_partition_len}) is out of the supported range for 'flash_attn' "
                f"({MIN_FLASH_ATTN_PARTITION_LENGTH} <= `kvcache_partition_len` <= {MAX_FLASH_ATTN_PARTITION_LENGTH}). "
                f"Please provide a valid value within this range."
            )
        elif max_seq_len < MIN_FLASH_ATTN_MAX_SEQ_LEN:
            raise ValueError(
                f"`max_seq_len` ({max_seq_len}) is too small for 'flash_attn'. The minimum "
                f"supported value is {MIN_FLASH_ATTN_MAX_SEQ_LEN}. Please increase `max_seq_len` to meet "
                "this requirement, or consider switching `attn_impl` to 'eager' for shorter lengths."
            )

    if kvcache_block_size is not None:
        if attn_impl == "flash_attn" and kvcache_partition_len != kvcache_block_size:
            raise ValueError(
                f" When using 'flash attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `kvcache_partition_len` {kvcache_partition_len}."
            )
        elif attn_impl == "eager" and kvcache_block_size != max_seq_len:
            raise ValueError(
                f" When using 'eager attention', the `kvcache_block_size` ({kvcache_block_size})  "
                f"must always be set equal to the `max_seq_len` {max_seq_len}."
            )


def validate_sliding_window(rbln_config: RBLNDecoderOnlyModelForCausalLMConfig):
    if rbln_config.sliding_window > MAX_SLIDING_WINDOW_SIZE - rbln_config.prefill_chunk_size:
        raise ValueError(
            f"Sliding window size ({rbln_config.sliding_window}) must be less than 32768 - prefill_chunk_size ({32768 - rbln_config.prefill_chunk_size})"
        )

    if rbln_config.cache_impl == "sliding_window" and rbln_config.use_attention_mask:
        raise ValueError("`use_attention_mask` must be set to False when `cache_impl` is set to 'sliding_window'.")


def align(x: int, nbytes: int) -> int:
    return int(math.ceil(x / nbytes) * nbytes)


def align_2MB(x: int) -> int:
    return align(x, 2**21)


def get_alloc_memory_by_key(compiled_models: Dict[str, "rebel.RBLNCompiledModel"]) -> Dict[str, int]:
    alloc_memory_by_key = defaultdict(int)
    # Get the actual memory allocation of each node by key
    for compiled_model in compiled_models.values():
        alloc_per_node_by_key = compiled_model.get_alloc_per_node_by_key()
        for key, memory_per_node in alloc_per_node_by_key.items():
            alloc_memory_by_key[key] += sum(memory_per_node)

    return alloc_memory_by_key


def format_byte_size(nbytes: int) -> str:
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024**2:
        return f"{nbytes / 1024:.2f} KB"
    elif nbytes < 1024**3:
        return f"{nbytes / 1024**2:.2f} MB"
    else:
        return f"{nbytes / 1024**3:.2f} GB"


class RBLNDecoderOnlyFlashAttentionMixin:
    @classmethod
    def get_maximum_num_blocks_by_model(
        cls,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> int:
        tensor_parallel_size = rbln_config.tensor_parallel_size or 1
        available_dram = get_available_dram(rbln_config.npu) * tensor_parallel_size

        kernel_memory = cls._get_kernel_memory(model, model_config=model_config, rbln_config=rbln_config)
        buffer = cls._get_buffer(rbln_config)

        remaining_dram = available_dram - kernel_memory - buffer
        if remaining_dram <= 0:
            raise ValueError(
                "Insufficient available DRAM after accounting for kernel memory and buffer. "
                "Cannot allocate any KV cache blocks."
                f" (Available DRAM: {format_byte_size(available_dram)}, "
                f"Kernel Memory: {format_byte_size(kernel_memory)}, "
                f"Buffer: {format_byte_size(buffer)})"
            )
        estimated_num_blocks = cls._estimate_num_blocks(
            remaining_dram, model_config=model_config, rbln_config=rbln_config
        )

        return estimated_num_blocks

    @classmethod
    def _get_kernel_memory(
        cls,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> int:
        if model.get_output_embeddings() is None:
            lm_head_nbytes = 0
        else:
            lm_head_nbytes = cls._get_lm_head_memory(model_config, rbln_config)

        layer_nbytes = cls._get_layer_memory(model, model_config, rbln_config)
        return lm_head_nbytes + layer_nbytes

    @classmethod
    def _get_lm_head_memory(
        cls, model_config: "PretrainedConfig", rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ) -> int:
        tensor_parallel_size = rbln_config.tensor_parallel_size or 1
        vocab_size = model_config.vocab_size
        hidden_size = getattr(model_config, "n_embd", None) or model_config.hidden_size
        lm_head_params = align(vocab_size, 64) * hidden_size

        nbytes_per_param = 2  # Assuming lm_head is always not quantized
        lm_head_memory_in_bytes = (
            align_2MB(lm_head_params * nbytes_per_param / tensor_parallel_size) * tensor_parallel_size
        )

        return lm_head_memory_in_bytes

    @classmethod
    def _get_layer_memory(
        cls,
        model: "PreTrainedModel",
        model_config: "PretrainedConfig",
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> int:
        # This is an *APPROXIMATE* calculation based on the number of parameters
        tensor_parallel_size = rbln_config.tensor_parallel_size or 1
        num_hidden_layers = getattr(model_config, "n_layer", None) or model_config.num_hidden_layers

        n_model_params = sum(p.numel() for p in model.parameters())
        embed_token_params = sum(p.numel() for p in model.get_input_embeddings().parameters())

        # Check : `embed_token` is same as `lm_head`
        if model.get_output_embeddings() is not None:
            params = n_model_params - 2 * embed_token_params
        else:
            params = n_model_params - embed_token_params

        # Assuming all layers have the same number of parameters
        # and all linear layers are quantized if quantization is enabled (This is not always true)
        # TODO(jongho): More accurate calculation
        nbits_per_param = rbln_config.nbits_per_param
        layer_nbytes = (
            (align_2MB(params // num_hidden_layers * nbits_per_param // 8 / tensor_parallel_size))
            * num_hidden_layers
            * tensor_parallel_size
        )

        return layer_nbytes

    @classmethod
    def _get_buffer(cls, rbln_config) -> int:
        # TODO(jongho): Accurate buffer estimation
        buffer_per_runtime_per_core = 2**28  # 256MB per runtime
        num_runtimes = 1 if not rbln_config.can_generate else 1 + len(rbln_config.decoder_batch_sizes)
        tensor_parallel_size = rbln_config.tensor_parallel_size or 1

        buffer_per_core = buffer_per_runtime_per_core * num_runtimes
        buffer = buffer_per_core * tensor_parallel_size
        return buffer

    @classmethod
    def get_maximum_num_blocks_by_compiled_model(
        cls,
        compiled_models: Dict[str, "rebel.RBLNCompiledModel"],
        model_config: "PretrainedConfig",
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> int:
        tensor_parallel_size = rbln_config.tensor_parallel_size or 1
        available_dram = get_available_dram(rbln_config.npu) * tensor_parallel_size

        alloc_memory_by_key = get_alloc_memory_by_key(compiled_models)
        alloc_memory_by_key.pop("PortRecur", None)  # Old compiler's kv-cache Key
        alloc_memory_by_key.pop("DramTensor", None)  # kv-cache
        used_memory = sum(alloc_memory_by_key.values())

        remaining_dram = available_dram - used_memory

        if remaining_dram <= 0:
            logger.warning(
                "Insufficient available DRAM after accounting for kernel memory and buffer. "
                "Model cannot allocate any KV cache blocks."
            )

        estimated_num_blocks = cls._estimate_num_blocks(
            remaining_dram, model_config=model_config, rbln_config=rbln_config
        )

        return estimated_num_blocks

    @classmethod
    def _estimate_num_blocks(
        cls, available_dram: int, model_config: "PretrainedConfig", rbln_config: RBLNDecoderOnlyModelForCausalLMConfig
    ) -> int:
        """
        Estimate the maximum number of KV cache blocks that can be allocated.

        if all of the layers are full attention, the dram_per_block can be calculated simply as follows:
            num_blocks = available_dram // dram_per_block

        However, if the model contains a mix of full attention and sliding window attention layers,
        we need to consider the memory occupied by the sliding window attention layers first,
        since their memory usage is constant regardless of the number of blocks.
            num_blocks = (available_dram - swa_kv_nbytes) // dram_per_block

        """

        def get_dram_per_block(seq_len: int, num_key_value_heads: int, tensor_parallel_size: int) -> int:
            nbytes_per_param = 2  # Assuming kv-cache is always not quantized
            dram_per_block = (
                seq_len
                * align(head_dim, 64)
                * math.ceil(num_key_value_heads / tensor_parallel_size)
                * nbytes_per_param
                * tensor_parallel_size
                * 2
            )  # *2 for key and value

            return dram_per_block

        num_attention_heads = getattr(model_config, "n_head", None) or model_config.num_attention_heads
        head_dim = getattr(model_config, "head_dim", None) or model_config.hidden_size // num_attention_heads
        num_hidden_layers = getattr(model_config, "n_layer", None) or model_config.num_hidden_layers
        num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or num_attention_heads
        tensor_parallel_size = rbln_config.tensor_parallel_size or 1

        # Consider layer types if available
        # If layer types are not found, assume all layers are full attention
        layer_types = getattr(model_config, "layer_types", None)
        if layer_types:
            layer_types_dict = Counter(layer_types)
            num_full_attention = layer_types_dict.pop("full_attention", 0)
            num_sliding_window_attention = layer_types_dict.pop("sliding_attention", 0)
            if len(layer_types_dict) > 0:
                raise ValueError(f"Unknown layer types found in the config: {layer_types_dict.keys()}")

        else:
            num_full_attention = num_hidden_layers
            num_sliding_window_attention = 0

        # Reduce available DRAM by sliding window attention kv-cache
        # Since memory occupation of swa layer is constant regardless of num_blocks
        swa_kv_nbytes = 0
        if num_sliding_window_attention > 0:
            sliding_window = getattr(model_config, "sliding_window", None)
            if sliding_window is None:
                logger.warning(
                    "`sliding_window` is not found in the config while `sliding_attention` layers are present. "
                    "Assuming maximum sliding window size for estimation."
                )
                sliding_window = rbln_config.kvcache_block_size

            swa_kv_nbytes = num_sliding_window_attention * get_dram_per_block(
                seq_len=sliding_window,
                num_key_value_heads=num_key_value_heads,
                tensor_parallel_size=tensor_parallel_size,
            )

            available_dram -= swa_kv_nbytes

        dram_per_block = num_full_attention * get_dram_per_block(
            seq_len=rbln_config.kvcache_block_size,
            num_key_value_heads=num_key_value_heads,
            tensor_parallel_size=tensor_parallel_size,
        )

        if dram_per_block == 0:
            raise ValueError("DRAM per block is calculated as zero, cannot estimate maximum number of blocks.")

        max_n_blocks = available_dram // dram_per_block
        return max_n_blocks

    @classmethod
    def maybe_suggest_kvcache_num_blocks(
        cls,
        compiled_models: Dict[str, "rebel.RBLNCompiledModel"],
        model_config: "PretrainedConfig",
        rbln_config: RBLNDecoderOnlyModelForCausalLMConfig,
    ) -> None:
        max_num_blocks = cls.get_maximum_num_blocks_by_compiled_model(
            compiled_models=compiled_models,
            model_config=model_config,
            rbln_config=rbln_config,
        )

        # Since our estimation logic is not always accurate,
        # users can set `kvcache_num_blocks` to `max_num_blocks`.
        # If the memory is not enough, the model will fail to compile.
        if rbln_config.kvcache_num_blocks < max_num_blocks:
            logger.warning(
                f"Current `kvcache_num_blocks` setting is {rbln_config.kvcache_num_blocks}. "
                "Our analysis indicates that additional memory is available for more blocks. "
                f"Consider increasing `kvcache_num_blocks` to {max_num_blocks} for potentially improved performance. "
                "Please be advised that our memory estimation algorithm has limitations, "
                "and increasing this value may not guarantee successful model compilation."
            )
