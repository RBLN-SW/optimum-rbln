from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_serializer, field_validator, model_validator

from ....utils.logging import get_logger


logger = get_logger(__name__)


class RBLNLoRAAdapterConfig(BaseModel):
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
                RBLNLoRAAdapterConfig(lora_int_id=1, lora_name="nemoguard", lora_path=lora_ids[0]),
                RBLNLoRAAdapterConfig(lora_int_id=2, lora_name="abliterated", lora_path=lora_ids[1]),
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
            print(generated_text + "\\n")
        ```

    Attributes:
        lora_int_id: Unique identifier for this LoRA adapter (e.g., 0, 1, 2).
        lora_name: Human-readable name for this adapter (e.g., "math_tuned", "code_tuned").
        lora_path: Path to the LoRA adapter weights directory or file.
        r: The rank of the LoRA approximation for this adapter.
        lora_alpha: The LoRA scaling parameter for this adapter.
        target_modules: List of module names to apply LoRA to.
        bias: Bias handling strategy. Options: "none", "all", "lora_only".
        use_rslora: Whether to use Rank-Stabilized LoRA.
        scaling_factor: Additional scaling factor for this adapter.
    """

    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    lora_int_id: int
    lora_name: str
    lora_path: Path
    r: int = 8
    lora_alpha: float = 8.0
    target_modules: list[str] | None = None
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    scaling_factor: float = 1.0

    _local_adapter_path: Path | None = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        # Convert string path to Path
        if "lora_path" in data and isinstance(data["lora_path"], str):
            data["lora_path"] = Path(data["lora_path"])

        # Resolve local adapter path before validation
        lora_path = data.get("lora_path")
        if lora_path is not None:
            local_adapter_path = self._resolve_adapter_path_static(Path(lora_path))
            # Load adapter config and merge defaults
            adapter_config = self._load_adapter_config_static(local_adapter_path)

            # Set defaults from adapter config if not provided
            if "r" not in data or data["r"] is None:
                data["r"] = adapter_config.get("r", 8)
            if "lora_alpha" not in data or data["lora_alpha"] is None:
                data["lora_alpha"] = adapter_config.get("lora_alpha", 8.0)
            if "target_modules" not in data or data["target_modules"] is None:
                data["target_modules"] = adapter_config.get("target_modules", None)
            if "bias" not in data or data["bias"] is None:
                data["bias"] = adapter_config.get("bias", "none")
            if "use_rslora" not in data or data["use_rslora"] is None:
                data["use_rslora"] = adapter_config.get("use_rslora", False)

        super().__init__(**data)

        # Store resolved local adapter path
        if lora_path is not None:
            self._local_adapter_path = local_adapter_path

    @property
    def local_adapter_path(self) -> Path | None:
        return self._local_adapter_path

    @staticmethod
    def _resolve_adapter_path_static(path: Path) -> Path:
        """Resolve the adapter path, downloading from HuggingFace Hub if necessary."""
        if path.exists():
            return path

        if path.is_absolute():
            raise ValueError(f"LoRA adapter path does not exist: {path.as_posix()}")

        try:
            local_dir = snapshot_download(str(path), allow_patterns=["*.safetensors", "*.bin", "*.json"])
            return Path(local_dir)
        except Exception as e:
            raise ValueError(
                f"Failed to download LoRA adapter '{path.as_posix()}' from HuggingFace Hub. "
                f"Please check if the model ID is correct or provide a valid local path. "
                f"Error: {e}"
            ) from e

    @staticmethod
    def _load_adapter_config_static(local_adapter_path: Path) -> dict[str, Any]:
        """Load adapter configuration from adapter_config.json file."""
        config_path = local_adapter_path / "adapter_config.json"

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

    @field_validator("lora_int_id")
    @classmethod
    def validate_lora_int_id(cls, v: int) -> int:
        if not isinstance(v, int):
            raise ValueError(f"lora_int_id must be an integer, got {type(v)}")
        return v

    @field_validator("r")
    @classmethod
    def validate_r(cls, v: int) -> int:
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"r must be a positive integer, got {v}")
        return v

    @field_validator("lora_alpha")
    @classmethod
    def validate_lora_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"lora_alpha must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def validate_bias_support(self) -> "RBLNLoRAAdapterConfig":
        if self.bias not in ["none"]:
            raise NotImplementedError("bias != 'none' is not supported yet")
        return self

    @field_serializer("lora_path")
    def serialize_lora_path(self, value: Path) -> str:
        return str(value)


class RBLNLoRABaseAdapterConfig(BaseModel):
    """
    Special adapter config for the reserved base model adapter (lora_int_id = 0).
    This adapter carries zero-effective LoRA weights by targeting no modules,
    thereby producing no LoRA delta and yielding pure base-model behavior.
    """

    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    lora_int_id: Literal[0] = 0
    lora_name: str = "base"
    lora_path: Path = Path("__reserved_base__")
    r: int = 1
    lora_alpha: float = 1.0
    target_modules: list[str] = []
    bias: Literal["none"] = "none"
    use_rslora: bool = False
    scaling_factor: float = 1.0

    _local_adapter_path: Path | None = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        # Convert string path to Path
        if "lora_path" in data and isinstance(data["lora_path"], str):
            data["lora_path"] = Path(data["lora_path"])

        # Validate lora_int_id
        if data.get("lora_int_id", 0) != 0:
            raise ValueError("RBLNLoRABaseAdapterConfig must have lora_int_id=0")

        # Force target_modules to empty list
        data["target_modules"] = []

        super().__init__(**data)

    @property
    def local_adapter_path(self) -> Path | None:
        return self._local_adapter_path

    @field_validator("r")
    @classmethod
    def validate_r(cls, v: int) -> int:
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"r must be a positive integer, got {v}")
        return v

    @field_validator("lora_alpha")
    @classmethod
    def validate_lora_alpha(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"lora_alpha must be positive, got {v}")
        return v

    @field_serializer("lora_path")
    def serialize_lora_path(self, value: Path) -> str:
        return str(value)


class RBLNLoRAConfig(BaseModel):
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

    Attributes:
        adapters: List of LoRA adapters to be compiled into the model.
        max_lora_rank: Maximum rank across all adapters.
    """

    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True, arbitrary_types_allowed=True)

    adapters: list[RBLNLoRAAdapterConfig | RBLNLoRABaseAdapterConfig]
    max_lora_rank: int | None = None

    def __init__(self, **data: Any):
        adapters_input = data.get("adapters", [])

        if not adapters_input:
            raise ValueError("adapters list cannot be empty")

        # Convert dict adapters to RBLNLoRAAdapterConfig objects
        converted_adapters: list[RBLNLoRAAdapterConfig | RBLNLoRABaseAdapterConfig] = []
        for adapter in adapters_input:
            if isinstance(adapter, dict):
                converted_adapters.append(RBLNLoRAAdapterConfig(**adapter))
            elif isinstance(adapter, (RBLNLoRAAdapterConfig, RBLNLoRABaseAdapterConfig)):
                converted_adapters.append(adapter)
            else:
                raise ValueError(f"Invalid adapter type: {type(adapter)}")

        # Disallow user-provided adapter with id 0: it's reserved for base model
        if any(ad.lora_int_id == 0 for ad in converted_adapters):
            raise ValueError(
                "lora_int_id=0 is reserved for base model and cannot be provided. "
                "Please renumber your adapters to start from 1."
            )

        # Inject a reserved zero-weight adapter for base model at id=0
        base_adapter = RBLNLoRABaseAdapterConfig()
        converted_adapters.insert(0, base_adapter)

        # Sort adapters by ID to make IDs align with indices
        converted_adapters.sort(key=lambda a: a.lora_int_id)

        data["adapters"] = converted_adapters

        # Calculate max_lora_rank if not provided
        max_lora_rank = data.get("max_lora_rank")
        if max_lora_rank is None:
            data["max_lora_rank"] = max(adapter.r for adapter in converted_adapters)

        super().__init__(**data)

    @model_validator(mode="after")
    def validate_adapters(self) -> "RBLNLoRAConfig":
        """Validate adapter IDs are unique and contiguous."""
        adapter_ids = [adapter.lora_int_id for adapter in self.adapters]
        if len(adapter_ids) != len(set(adapter_ids)):
            raise ValueError("All adapter IDs must be unique")

        expected_ids = list(range(len(self.adapters)))
        if adapter_ids != expected_ids:
            raise ValueError(
                f"Adapter IDs must be contiguous and start from 0. Found {adapter_ids}, expected {expected_ids}."
            )

        # Validate that max_lora_rank is sufficient
        if self.max_lora_rank is not None:
            actual_max_rank = max(adapter.r for adapter in self.adapters)
            if self.max_lora_rank < actual_max_rank:
                raise ValueError(
                    f"max_lora_rank ({self.max_lora_rank}) must be >= actual max rank ({actual_max_rank})"
                )

        return self

    @property
    def num_adapters(self) -> int:
        return len(self.adapters)

    @property
    def adapter_ids(self) -> list[int]:
        return [adapter.lora_int_id for adapter in self.adapters]

    @property
    def adapter_names(self) -> list[str]:
        return [adapter.lora_name for adapter in self.adapters]

    def get_adapter_by_id(self, lora_int_id: int) -> RBLNLoRAAdapterConfig | RBLNLoRABaseAdapterConfig | None:
        for adapter in self.adapters:
            if adapter.lora_int_id == lora_int_id:
                return adapter
        return None

    def get_adapter_by_name(self, lora_name: str) -> RBLNLoRAAdapterConfig | RBLNLoRABaseAdapterConfig | None:
        for adapter in self.adapters:
            if adapter.lora_name == lora_name:
                return adapter
        return None

    def validate_adapter_weights(self) -> dict[int, bool]:
        validation_results: dict[int, bool] = {}
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

    @field_serializer("adapters")
    def serialize_adapters(
        self, adapters: list[RBLNLoRAAdapterConfig | RBLNLoRABaseAdapterConfig]
    ) -> list[dict[str, Any]]:
        # Do not serialize the reserved base adapter (id=0)
        return [adapter.model_dump() for adapter in adapters if adapter.lora_int_id != 0]
