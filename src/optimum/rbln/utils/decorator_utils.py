# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from functools import wraps

from .logging import get_logger


logger = get_logger(__name__)


def remove_compile_time_kwargs(func):
    """
    Decorator to handle compile-time parameters during inference.

    For RBLN-optimized pipelines, several parameters must be determined during compilation
    and cannot be modified during inference. This decorator:
    1. Removes and warns about image dimension parameters (height, width)
    2. Removes and warns about LoRA scale in cross_attention_kwargs

    Args:
        func: The pipeline's __call__ method to be wrapped
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        check_params = {"height", "width"}
        params = inspect.signature(self.original_class.__call__).parameters

        # If height and width exist in the base pipeline's __call__ method arguments
        # Otherwise, if there is no height or width of kwargs, it is filled based on the compiled size.
        if check_params.issubset(params):
            compiled_image_size = self.get_compiled_image_size()
            if compiled_image_size is not None:
                height_exists = "height" in kwargs and kwargs["height"] is not None
                width_exists = "width" in kwargs and kwargs["width"] is not None
                if height_exists or width_exists:
                    if not (
                        kwargs.get("height", None) == compiled_image_size[0]
                        and kwargs.get("width", None) == compiled_image_size[1]
                    ):
                        logger.warning(
                            "Image dimension parameters (`height`, `width`) will be ignored during inference. "
                            "Image dimensions (%s, %s) must be specified during model compilation using from_pretrained(), (%s, %s).",
                            str(kwargs.get("height", None)),
                            str(kwargs.get("width", None)),
                            str(compiled_image_size[0]),
                            str(compiled_image_size[1]),
                        )
                kwargs["height"] = compiled_image_size[0]
                kwargs["width"] = compiled_image_size[1]

        if "cross_attention_kwargs" in kwargs:
            cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
            if not cross_attention_kwargs:
                return func(self, *args, **kwargs)

            has_scale = "scale" in cross_attention_kwargs
            if has_scale:
                logger.warning(
                    "LoRA scale in cross_attention_kwargs will be ignored during inference. "
                    "To adjust LoRA scale, specify it during model compilation using from_pretrained()."
                )

                # If scale is the only key, set to None
                # Otherwise, remove scale and preserve other settings
                if len(cross_attention_kwargs) == 1:
                    kwargs["cross_attention_kwargs"] = None
                else:
                    kwargs["cross_attention_kwargs"].pop("scale")

        return func(self, *args, **kwargs)

    return wrapper


def deprecate_method(
    version: str,
    new_method: Optional[str] = None,
    raise_if_greater_or_equal_version: bool = True,
    additional_message: Optional[str] = None,
):
    """
    Decorator to mark a method as deprecated, optionally pointing to a replacement method.
    This decorator allows you to:
    - Notify users when a method is deprecated.
    - Optionally specify a new method name that should be used instead.
    - Raise an error if the deprecated method is called after the specified version.
    Parameters:
        version (`str`):
            The version in which the method was (or will be) deprecated.
        new_method (`Optional[str]`, *optional*):
            The name of the new method to use instead. If specified, users will be directed to use this method.
        raise_if_greater_or_equal_version (`bool`, *optional*, defaults to `True`):
            Whether to raise `ValueError` if current `optimum.rbln` version is greater than or equal to the deprecated version.
        additional_message (`Optional[str]`, *optional*):
            An additional message to append to the default deprecation message.
    Returns:
        Callable:
            A wrapped function that handles the deprecation warning or error.
    Examples:
        >>> class MyClass:
        ...     @deprecate_method(version="0.12.0", new_method="from_pretrained")
        ...     def load(self, path):
        ...         return self.from_pretrained(path)
    """

    deprecated_version = packaging.version.parse(version)
    current_version = packaging.version.parse(__version__)
    is_greater_or_equal_version = current_version >= deprecated_version

    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Get class + method name for better warning message
            method_name = func.__name__
            if is_instance_method:
                method_name = f"{args[0].__class__.__name__}.{method_name}"
            elif is_class_method:
                method_name = f"{args[0].__name__}.{method_name}"

            # Build deprecation message
            if new_method is not None:
                message = f"`{method_name}` is deprecated {version_message}. Use `{new_method}` instead."
            else:
                message = f"`{method_name}` is deprecated {version_message}."

            if additional_message is not None:
                message = f"{message} {additional_message}"

            # Determine action based on version
            if is_greater_or_equal_version and raise_if_greater_or_equal_version:
                raise ValueError(message)
            else:
                logger.warning(message, stacklevel=2)

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper