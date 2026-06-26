import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import snapshot_download

from ....configuration_utils import RBLNSerializableConfigProtocol
from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNLoRAAdapterConfig(RBLNSerializableConfigProtocol):
    """
    Configuration class for individual LoRA adapter settings.

    This class represents a single LoRA adapter that will be compiled into the RBLN model.
    Since RBLN NPU requires all adapters to be determined at compile time, each adapter
    must be fully specified including its weights.

    Examples:
        ```python
        from transformers import AutoTokenizer

        from optimum.rbln import RBLNLlamaForCausalLM, RBLNLlamaForCausalLMConfig, RBLNLoRAAdapterConfig, RBLNLoRAConfig


        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        lora_ids = [
            "nvidia/llama-3.1-nemoguard-8b-topic-control",
            "reissbaker/llama-3.1-8b-abliterated-lora",
        ]
        prompt = "What are the safety considerations for AI systems?"
        tp_size = 4

        # adapter id should be higher than 0
        # 0 is reserved for base model
        lora_config = RBLNLoRAConfig(
            adapters=[
                RBLNLoRAAdapterConfig(1, "nemoguard", lora_ids[0]),
                RBLNLoRAAdapterConfig(2, "abliterated", lora_ids[1]),
            ],
        )

        model = RBLNLlamaForCausalLM.from_pretrained(
            model_id,
            rbln_config=RBLNLlamaForCausalLMConfig(lora_config=lora_config, tensor_parallel_size=tp_size, max_seq_len=8192),
            dtype="auto",
        )


        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token


        prompt_template = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant. Always be concise"},
                {"role": "user", "content": prompt},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer([prompt_template], return_tensors="pt")
        input_len = inputs["input_ids"].shape[-1]

        for adapter_name in lora_config.adapter_names:
            model.set_adapter(adapter_name)
            decoder_outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            generated_text = tokenizer.decode(decoder_outputs[0][input_len:], skip_special_tokens=True)
            print(generated_text + "\n")
        ```

    """

    def __init__(
        self,
        lora_int_id: int,
        lora_name: str,
        lora_path: Union[str, Path],
        r: Optional[int] = None,
        lora_alpha: Optional[float] = None,
        target_modules: Optional[List[str]] = None,
        bias: Optional[str] = None,
        use_rslora: Optional[bool] = None,
        scaling_factor: Optional[float] = None,
    ):
        """
        Args:
            lora_int_id (int): Unique identifier for this LoRA adapter (e.g., 0, 1, 2).
                This ID will be used during runtime to select which adapter to use.
            lora_name (str): Human-readable name for this adapter (e.g., "math_tuned", "code_tuned").
            lora_path (Union[str, Path]): Path to the LoRA adapter weights directory or file.
                Must be accessible at compile time to load the weights.
            r (Optional[int]): The rank of the LoRA approximation for this adapter. If None,
                will be loaded from adapter config file.
            lora_alpha (Optional[float]): The LoRA scaling parameter for this adapter. If None,
                will be loaded from adapter config file.
            target_modules (Optional[List[str]]): List of module names to apply LoRA to.
                If None, will be loaded from adapter config file or inherit from parent RBLNLoRAConfig.
            bias (Optional[str]): Bias handling strategy. Options: "none", "all", "lora_only".
                If None, will be loaded from adapter config file.
            use_rslora (Optional[bool]): Whether to use Rank-Stabilized LoRA. If None,
                will be loaded from adapter config file.
            scaling_factor (Optional[float]): Additional scaling factor for this adapter. Defaults to 1.0.
            **kwargs: Additional adapter-specific arguments.

        Raises:
            ValueError: If lora_int_id is None.
            ValueError: If lora_path doesn't exist.
            ValueError: If r is not a positive integer.
            ValueError: If lora_alpha is not positive.
        """
        if lora_int_id is None:
            raise ValueError("lora_int_id cannot be None")

        if not isinstance(lora_int_id, int):
            raise ValueError(f"lora_int_id must be an integer, got {type(lora_int_id)}")

        self.lora_int_id = lora_int_id
        self.lora_name = lora_name

        # Keep original lora_path as provided by user (for serialization)
        self.lora_path = Path(lora_path)

        # Resolve to local directory path (for actual weight loading)
        self.local_adapter_path = self._resolve_adapter_path(self.lora_path)

        # Load adapter config and use as defaults
        adapter_config = self._load_adapter_config()

        # Set values from adapter config if not explicitly provided
        self.r = r if r is not None else adapter_config.get("r", 8)
        self.lora_alpha = lora_alpha if lora_alpha is not None else adapter_config.get("lora_alpha", 8.0)
        self.target_modules = (
            target_modules if target_modules is not None else adapter_config.get("target_modules", None)
        )
        self.bias = bias if bias is not None else adapter_config.get("bias", "none")
        if self.bias not in ["none"]:
            raise NotImplementedError("bias != 'none' is not supported yet")

        self.use_rslora = use_rslora if use_rslora is not None else adapter_config.get("use_rslora", False)
        self.scaling_factor = scaling_factor if scaling_factor is not None else 1.0

        # Validate the final values
        if not isinstance(self.r, int) or self.r <= 0:
            raise ValueError(f"r must be a positive integer, got {self.r}")

        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")

        if self.bias not in ["none", "all", "lora_only"]:
            raise ValueError(f"bias must be one of ['none', 'all', 'lora_only'], got {self.bias}")

    def _resolve_adapter_path(self, path: Path) -> Path:
        """
        Resolve the adapter path, downloading from HuggingFace Hub if necessary.

        Args:
            path: Local path or HuggingFace Hub model ID

        Returns:
            Path object pointing to local adapter directory

        Raises:
            ValueError: If the adapter cannot be found locally or downloaded
        """
        # If it's a local path and exists, return it
        if path.exists():
            return path

        # If it's an absolute path that doesn't exist, raise error
        if path.is_absolute():
            raise ValueError(f"LoRA adapter path does not exist: {path.as_posix()}")

        # Try to interpret as HuggingFace Hub model ID and download
        try:
            local_dir = snapshot_download(str(path), allow_patterns=["*.safetensors", "*.bin", "*.json"])
            return Path(local_dir)
        except Exception as e:
            raise ValueError(
                f"Failed to download LoRA adapter '{path.as_posix()}' from HuggingFace Hub. "
                f"Please check if the model ID is correct or provide a valid local path. "
                f"Error: {e}"
            ) from e

    def _load_adapter_config(self) -> Dict[str, Any]:
        """
        Load adapter configuration from adapter_config.json file.

        Returns:
            Dictionary containing adapter configuration

        Raises:
            ValueError: If adapter_config.json is not found or cannot be parsed
        """
        config_path = self.local_adapter_path / "adapter_config.json"

        if not config_path.exists():
            logger.warning(f"No adapter_config.json found at {config_path}, using default values")
            return {}

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
            logger.info(f"Loaded adapter config from {config_path}")
            return adapter_config
        except Exception as e:
            logger.warning(f"Failed to load adapter config from {config_path}: {e}, using default values")
            return {}

    def _prepare_for_serialization(self) -> Dict[str, Any]:
        config_dict = {
            "lora_int_id": self.lora_int_id,
            "lora_name": self.lora_name,
            "lora_path": str(self.lora_path),
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "use_rslora": self.use_rslora,
            "scaling_factor": self.scaling_factor,
        }
        return config_dict


class RBLNLoRABaseAdapterConfig(RBLNLoRAAdapterConfig):
    """
    Special adapter config for the reserved base model adapter (lora_int_id = 0).
    This adapter carries zero-effective LoRA weights by targeting no modules,
    thereby producing no LoRA delta and yielding pure base-model behavior.
    """

    def __init__(
        self,
        lora_int_id: int = 0,
        lora_name: str = "base",
        lora_path: Union[str, Path] = "__reserved_base__",
        r: Optional[int] = 1,
        lora_alpha: Optional[float] = 1.0,
        target_modules: Optional[List[str]] = None,
        bias: Optional[str] = "none",
        use_rslora: Optional[bool] = False,
        scaling_factor: Optional[float] = 1.0,
    ):
        if lora_int_id != 0:
            raise ValueError("RBLNLoRABaseAdapterConfig must have lora_int_id=0")

        self.lora_int_id = 0
        self.lora_name = lora_name
        # Keep original lora_path for serialization purposes but do not resolve it.
        self.lora_path = Path(str(lora_path))
        self.local_adapter_path = None

        # Set minimal defaults; target_modules empty disables LoRA on all projections
        self.r = 1 if r is None else r
        self.lora_alpha = 1.0 if lora_alpha is None else lora_alpha
        self.target_modules = []
        self.bias = "none"
        self.use_rslora = False
        self.scaling_factor = 1.0

        # Validate minimal settings
        if not isinstance(self.r, int) or self.r <= 0:
            raise ValueError(f"r must be a positive integer, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")


class RBLNLoRAConfig(RBLNSerializableConfigProtocol):
    """
    Configuration class for multi-LoRA support in RBLN decoder-only models.

    This class manages all LoRA adapters that will be compiled into the RBLN model.
    Since RBLN NPU requires all adapters to be determined at compile time, this
    configuration must specify all adapters upfront with their weights.

    Key constraints for RBLN multi-LoRA:
    1. All LoRA adapters must be specified at compile time
    2. Adapter weights must be available during compilation
    3. The number of adapters is fixed after compilation
    4. Runtime can only switch between pre-compiled adapters
    """

    def __init__(
        self, adapters: List[Union[Dict[str, Any], RBLNLoRAAdapterConfig]], max_lora_rank: Optional[int] = None
    ):
        """
        Args:
            adapters (List[Union[Dict[str, Any], RBLNLoRAAdapterConfig]]): List of LoRA adapters
                to be compiled into the model. Each adapter must be fully specified with weights
                accessible at compile time.
            max_lora_rank (Optional[int]): Maximum rank across all adapters. If None, automatically
                determined from the provided adapters. Used for memory allocation optimization.

        Raises:
            ValueError: If adapters list is empty.
            ValueError: If adapter IDs are not unique.
            ValueError: If any adapter path doesn't exist.
        """
        if not adapters:
            raise ValueError("adapters list cannot be empty")

        # Convert dict adapters to RBLNLoRAAdapterConfig objects
        self.adapters: List[RBLNLoRAAdapterConfig] = []
        for adapter in adapters:
            if isinstance(adapter, dict):
                self.adapters.append(RBLNLoRAAdapterConfig(**adapter))
            elif isinstance(adapter, RBLNLoRAAdapterConfig):
                self.adapters.append(adapter)
            else:
                raise ValueError(f"Invalid adapter type: {type(adapter)}")

        # Disallow user-provided adapter with id 0: it's reserved for base model
        if any(ad.lora_int_id == 0 for ad in self.adapters):
            raise ValueError(
                "lora_int_id=0 is reserved for base model and cannot be provided. Please renumber your adapters to start from 1."
            )

        # Inject a reserved zero-weight adapter for base model at id=0
        base_adapter = RBLNLoRABaseAdapterConfig()
        self.adapters.insert(0, base_adapter)

        # Sort adapters by ID to make IDs align with indices
        self.adapters.sort(key=lambda a: a.lora_int_id)

        # Validate unique and contiguous adapter IDs starting from 0
        adapter_ids = [adapter.lora_int_id for adapter in self.adapters]
        if len(adapter_ids) != len(set(adapter_ids)):
            raise ValueError("All adapter IDs must be unique")
        expected_ids = list(range(len(self.adapters)))
        if adapter_ids != expected_ids:
            raise ValueError(
                f"Adapter IDs must be contiguous and start from 0. Found {adapter_ids}, expected {expected_ids}."
            )

        # Calculate max_lora_rank if not provided
        if max_lora_rank is None:
            self.max_lora_rank = max(adapter.r for adapter in self.adapters)
        else:
            self.max_lora_rank = max_lora_rank
            # Validate that max_lora_rank is sufficient
            actual_max_rank = max(adapter.r for adapter in self.adapters)
            if self.max_lora_rank < actual_max_rank:
                raise ValueError(
                    f"max_lora_rank ({self.max_lora_rank}) must be >= actual max rank ({actual_max_rank})"
                )

    @property
    def num_adapters(self) -> int:
        return len(self.adapters)

    @property
    def adapter_ids(self) -> List[int]:
        return [adapter.lora_int_id for adapter in self.adapters]

    @property
    def adapter_names(self) -> List[str]:
        return [adapter.lora_name for adapter in self.adapters]

    def get_adapter_by_id(self, lora_int_id: int) -> Optional[RBLNLoRAAdapterConfig]:
        for adapter in self.adapters:
            if adapter.lora_int_id == lora_int_id:
                return adapter
        return None

    def get_adapter_by_name(self, lora_name: str) -> Optional[RBLNLoRAAdapterConfig]:
        for adapter in self.adapters:
            if adapter.lora_name == lora_name:
                return adapter
        return None

    def validate_adapter_weights(self) -> Dict[int, bool]:
        validation_results = {}
        for adapter in self.adapters:
            try:
                # The reserved base adapter (id=0) always validates to True
                if adapter.lora_int_id == 0:
                    validation_results[adapter.lora_int_id] = True
                    continue
                # Check if adapter path exists and contains expected files
                adapter_path = adapter.local_adapter_path
                if adapter_path is not None and adapter_path.is_file():
                    # Single file adapter (e.g., safetensors)
                    validation_results[adapter.lora_int_id] = adapter_path.exists()
                else:
                    # Directory adapter - check for common LoRA files
                    expected_files = ["adapter_model.safetensors", "adapter_config.json"]
                    alternative_files = ["pytorch_model.bin", "adapter_model.bin"]

                    has_weights = adapter_path is not None and any(
                        (adapter_path / f).exists() for f in expected_files + alternative_files
                    )
                    has_config = adapter_path is not None and (adapter_path / "adapter_config.json").exists()

                    validation_results[adapter.lora_int_id] = has_weights and has_config
            except Exception as e:
                logger.warning(f"Failed to validate adapter {adapter.lora_int_id}: {e}")
                validation_results[adapter.lora_int_id] = False

        return validation_results

    def _prepare_for_serialization(self) -> Dict[str, Any]:
        # Do not serialize the reserved base adapter (id=0)
        serializable_adapters = [adapter for adapter in self.adapters if adapter.lora_int_id != 0]
        serializable_map = {
            "adapters": [adapter._prepare_for_serialization() for adapter in serializable_adapters],
            "max_lora_rank": self.max_lora_rank,
        }
        return serializable_map
