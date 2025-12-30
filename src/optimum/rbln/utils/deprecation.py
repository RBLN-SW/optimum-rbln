# Copyright 2024 The HuggingFace Team. All rights reserved.
# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# **********************************************************************************
# * NOTE: This file has been modified from its original version in              *
# * the Hugging Face transformers library.                                      *
# * Original source:                                                            *
# * https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/utils/deprecation.py
# **********************************************************************************

import inspect
from enum import Enum
from functools import wraps
from typing import Callable, Optional

import packaging.version

from ..__version__ import __version__
from .logging import get_logger


logger = get_logger(__name__)


def warn_deprecated_npu(npu: Optional[str] = None):
    import rebel

    npu = npu or rebel.get_npu_name()
    if npu == "RBLN-CA02":
        logger.warning_once(
            "Support for the RBLN-CA02 device is provided only up to optimum-rbln v0.8.0 and has reached end of life.",
        )


class Action(Enum):
    NONE = "none"
    NOTIFY = "notify"
    RAISE = "raise"


# Scenario Table for Deprecation Strategy Example
# Assume that current version is v0.9.6 and the deprecated version is v0.10.0
# |--------------------|----------------|----------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------|
# | Type               | v0.9.6 (as_is) | v0.9.6 (to_be) | v0.9.6 Patch                                | v0.9.7 Action                                                                        | v0.10.0+ Action                                                      |
# |--------------------|----------------|----------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------|
# | Modify (Key Name)  | a: bool        | a': bool       | Add a', Keep a                              | 1. Only 'a' provided: replace a -> a' & future warning                               | In v0.10.0, raise error once, then remove decorator.                 |
# |                    |                |                |                                             | 2. Both 'a' & 'a'' provided: ignore 'a' value & future warning                       |                                                                      |
# |--------------------|----------------|----------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------|
# | Modify (Value Type)| b: bool        | b: str         | b: Union[str, bool]                         | 'bool' value provided for 'b': replace with corresponding 'str' & future warning     | In v0.10.0, raise error once, then remove decorator.                 |
# |                    |                |                |                                             |                                                                                      |                                                                      |
# |--------------------|----------------|----------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------|
# | Deletion           | c              | -              | Delete c or Keep c (flexible)               | ignore c & future warning                                                            | In v0.10.0, raise error once, then remove decorator.                 |
# |                    |                |                |                                             |                                                                                      |                                                                      |
# |--------------------|----------------|----------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------|
# | Addition           | -              | d              | Add d, set default_value for d              | No action needed as default value is set                                             | Keep default value                                                   |
# |--------------------|----------------|----------------|---------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------|


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: Optional[str] = None,
    deprecated_type: Optional[type] = None,
    value_replacer: Optional[Callable] = None,
    raise_if_greater_or_equal_version: bool = True,
    raise_if_both_names: bool = False,
    additional_message: Optional[str] = None,
):
    """
    Function or method decorator to notify users about deprecated keyword arguments, replacing them with a new name if specified,
    or handling deprecated value types.

    This decorator allows you to:
    - Notify users when a keyword argument name is deprecated (Scenario 'a', 'c').
    - Notify users when a specific value type for an argument is deprecated (Scenario 'b').
    - Automatically replace deprecated keyword arguments with new ones.
    - Automatically replace deprecated values with new ones using a replacer function.
    - Raise an error if deprecated arguments are used, depending on the specified conditions.

    By default, the decorator notifies the user about the deprecated argument while the `optimum.rbln.__version__` < specified `version`
    in the decorator. To keep notifications with any version `warn_if_greater_or_equal_version=True` can be set.

    Parameters:
        old_name (`str`):
            Name of the deprecated keyword argument, or the argument with a deprecated value type.
        version (`str`):
            The version in which the keyword argument or value type was (or will be) deprecated.
        new_name (`Optional[str]`, *optional*):
            The new name for the deprecated keyword argument. If specified, the deprecated keyword argument will be replaced with this new name (Scenario 'a').
        deprecated_type (`type`, *optional*):
            The deprecated type for the keyword argument specified by `old_name` (Scenario 'b').
            If this is set, `new_name` should typically be `None`.
        value_replacer (`Callable`, *optional*):
            A function that takes the old (deprecated type) value and returns a new value (Scenario 'b').
            Used in conjunction with `deprecated_type`. If provided, the value will be automatically converted.
        raise_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if current `optimum.rbln.` version is greater or equal to the deprecated version.
        raise_if_both_names (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if both deprecated and new keyword arguments are set (only for Scenario 'a').
        additional_message (`Optional[str]`, *optional*):
            An additional message to append to the default deprecation message.

    Raises:
        ValueError:
            If raise_if_greater_or_equal_version is True and the current version is greater than or equal to the deprecated version, or if raise_if_both_names is True and both old and new keyword arguments are provided.

    Returns:
        Callable:
            A wrapped function that handles the deprecated keyword arguments according to the specified parameters.
    """

    deprecated_version = packaging.version.parse(version)
    current_version = packaging.version.parse(__version__)
    is_greater_or_equal_version = current_version >= deprecated_version

    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        # Required for better warning message
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Get class + function name (just for better warning message)
            func_name = func.__name__
            if is_instance_method:
                func_name = f"{args[0].__class__.__name__}.{func_name}"
            elif is_class_method:
                func_name = f"{args[0].__name__}.{func_name}"

            minimum_action = Action.NONE
            message = None

            # Scenario A: Rename (e.g., a -> a')
            if new_name is not None:
                if old_name in kwargs and new_name in kwargs:
                    minimum_action = Action.RAISE if raise_if_both_names else Action.NOTIFY
                    message = f"Both `{old_name}` and `{new_name}` are set for `{func_name}`. Using `{new_name}={kwargs[new_name]}` and ignoring deprecated `{old_name}={kwargs[old_name]}`."
                    kwargs.pop(old_name)

                elif old_name in kwargs and new_name not in kwargs:
                    minimum_action = Action.NOTIFY
                    message = (
                        f"`{old_name}` is deprecated {version_message} for `{func_name}`. Use `{new_name}` instead."
                    )
                    kwargs[new_name] = kwargs.pop(old_name)

            # Scenario B: Value Type Change (e.g., b: bool -> str)
            if deprecated_type is not None:
                key_to_check = old_name if new_name is None else new_name  # For Senario A + B Mixed
                if key_to_check in kwargs and isinstance(kwargs[key_to_check], deprecated_type):
                    minimum_action = Action.NOTIFY
                    old_value = kwargs[key_to_check]
                    message = f"Using type `{deprecated_type.__name__}` for argument `{key_to_check}` in `{func_name}` is deprecated {version_message}."

                    if value_replacer:
                        try:
                            new_value = value_replacer(old_value)
                            kwargs[key_to_check] = new_value
                            message += f" Value `{old_value}` has been automatically replaced with `{new_value}`."
                        except Exception as e:
                            logger.error(f"Error during deprecated value replacement for {key_to_check}: {e}")
                            message += f" Automatic replacement failed: {e}. Passing original value."
                    else:
                        raise ValueError(
                            f"value_replacer should be provided when deprecated_type is set for {key_to_check} in {func_name}"
                        )

            # Scenario C: Deletion (e.g., c)
            if old_name in kwargs and new_name is None and deprecated_type is None:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`."
                kwargs.pop(old_name)

            if message is not None and additional_message is not None:
                message = f"{message} {additional_message}"

            # update minimum_action if argument is ALREADY deprecated (current version >= deprecated version)
            if is_greater_or_equal_version:
                # change to NOTIFY -> RAISE  in case we want to raise error for already deprecated arguments
                if raise_if_greater_or_equal_version:
                    minimum_action = Action.RAISE

            # raise error or notify user
            if minimum_action == Action.RAISE:
                raise ValueError(message)
            elif minimum_action == Action.NOTIFY:
                # DeprecationWarning is ignored by default, so we use FutureWarning instead
                logger.warning(message, stacklevel=2)

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper
