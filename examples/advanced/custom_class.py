"""
# Advanced Tutorial: Creating Custom RBLN Classes

This tutorial demonstrates how to create custom RBLN model classes for use with the optimum-rbln framework.

## Requirements for Custom Class Creation:

1. You must create two classes:
   - A custom model class that inherits from RBLNModel
   - A custom config class that inherits from RBLNModelConfig

2. The naming convention is critical:
   - Model class: RBLN<OriginalModelName> (e.g., RBLNResNetModel)
   - Config class: RBLN<OriginalModelName>Config (e.g., RBLNResNetModelConfig)

3. Your custom class MUST implement the _update_rbln_config method
   - This method configures input shapes and compilation settings

4. You must register both classes with the auto registration system
   - This enables the framework to discover and use your custom classes

Important: Custom RBLN classes can only be created for models that exist in
the transformers or diffusers libraries. You cannot create custom RBLN classes
for completely custom architectures without a corresponding HuggingFace implementation.

This example demonstrates creating a custom RBLN class for the ResNet model from transformers.
"""

from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import ResNetModel  # noqa: F401
from transformers.models.resnet.modeling_resnet import BaseModelOutputWithPoolingAndNoAttention

from optimum.rbln import RBLNAutoConfig, RBLNAutoModel, RBLNCompileConfig, RBLNModel, RBLNModelConfig


if TYPE_CHECKING:
    from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel


# --------------------------------------------------------
# STEP 1: Create a custom model class that extends RBLNModel
# --------------------------------------------------------
# A custom RBLN model class must:
# - Follow the naming pattern: RBLN<OriginalModelName>
# - Implement the _update_rbln_config method (required)
# - Define a proper forward method matching the original model's inputs/outputs
#
# The _update_rbln_config method is critical - it sets up the input tensor
# specifications and other compilation parameters needed for the RBLN compiler.
#
# For our ResNet example, we need to:
# 1. Specify the input tensor shape for images (batch_size, channels, height, width)
# 2. Set the proper configuration options
# 3. Return the updated RBLN configuration
class RBLNResNetModel(RBLNModel):
    @classmethod
    def _update_rbln_config(
        cls,
        preprocessors: Union["AutoFeatureExtractor", "AutoProcessor", "AutoTokenizer"] | None = None,
        model: Optional["PreTrainedModel"] = None,
        model_config: Optional["PretrainedConfig"] = None,
        rbln_config: Optional["RBLNResNetModelConfig"] = None,
    ) -> "RBLNResNetModelConfig":
        # Set image_size if not provided
        if rbln_config.image_size is None:
            if rbln_config.image_size is None:
                rbln_config.image_size = model_config.image_size

            if rbln_config.image_size is None:
                raise ValueError("`image_size` should be specified!")

        # Define input tensor specification for the compiler
        # Format: (tensor_name, tensor_shape, dtype)
        input_info = [
            (
                "pixel_values",
                [rbln_config.batch_size, 3, rbln_config.image_size[0], rbln_config.image_size[1]],
                "float32",
            )
        ]

        # Configure compilation settings
        rbln_config.set_compile_cfgs([RBLNCompileConfig(input_info=input_info)])
        return rbln_config

    def forward(self, pixel_values, return_dict: bool | None = None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # self.model is a list of rebel.Runtime objects
        # See https://docs.rbln.ai/software/api/python/python_api.html#rebel.rebel_runtime.Runtime for more details
        output = self.model[0](pixel_values)

        if not return_dict:
            return output
        else:
            return BaseModelOutputWithPoolingAndNoAttention(last_hidden_state=output[0], pooler_output=output[1])


# ----------------------------------------------------------------
# STEP 2: Create a custom configuration class that extends RBLNModelConfig
# ----------------------------------------------------------------
# The configuration class must:
# - Follow the naming pattern: RBLN<OriginalModelName>Config
# - Define model-specific parameters needed for compilation
#
# For ResNet, we need:
# - batch_size: Batch size for inference
# - image_size: Input image dimensions (height, width)
class RBLNResNetModelConfig(RBLNModelConfig):
    def __init__(self, batch_size: int = None, image_size: tuple[int, int] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size or 1
        self.image_size = image_size or (224, 224)


# ----------------------------------------------------------------
# STEP 3: Register both classes with the auto registration system
# ----------------------------------------------------------------
# Registration is essential for the RBLN framework to:
# 1. Discover your custom classes when needed
# 2. Enable using RBLNAutoModel.from_pretrained() with your model type
# 3. Connect your configuration class to the model class in the framework
RBLNAutoModel.register(RBLNResNetModel)
RBLNAutoConfig.register(RBLNResNetModelConfig)


# ----------------------------------------------------------------
# STEP 4: Usage Example - Creating and using the custom RBLN model
# ----------------------------------------------------------------
# Initialize the model from a pretrained HuggingFace model
# The 'export=True' parameter triggers the model compilation process
# rbln_image_size and rbln_batch_size are passed to the custom configuration
my_model = RBLNResNetModel.from_pretrained(
    "microsoft/resnet-50", export=True, rbln_image_size=(224, 224), rbln_batch_size=1
)

# Save the compiled model for later use
my_model.save_pretrained("my_resnet_model_saved")

# Create a random input tensor for demonstration
# The shape must match what we defined in _update_rbln_config
random_image_input = torch.randn(1, 3, 224, 224)

# Run inference with the compiled model
output = my_model(random_image_input)

# Print each key-value pair in the output dictionary
for key, value in output.items():
    print(key, value)


# ----------------------------------------------------------------
# STEP 5: Loading a saved RBLN model
# ----------------------------------------------------------------
# Load the model we just saved - no need to recompile
my_model_reloaded = RBLNResNetModel.from_pretrained("my_resnet_model_saved")
print(my_model_reloaded)
