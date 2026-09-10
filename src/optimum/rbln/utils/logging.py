# Copyright 2020 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

"""
Logging utilities.
Modified from `transformers.utils.logging.py`
"""

import functools
import logging
import os
import sys
import threading
from typing import Any, cast


_lock = threading.Lock()
_default_handler: logging.Handler | None = None


log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.INFO


class Logger(logging.Logger):
    """Typing view of the loggers returned by `get_logger`.

    `warning_once` is installed on `logging.Logger` below, so every logger in the process has it;
    this class only makes that visible to type checkers.
    """

    def warning_once(self, *args: Any, **kwargs: Any) -> None: ...


@functools.lru_cache(None)
def _warning_once(self: logging.Logger, *args: Any, **kwargs: Any) -> None:
    self.warning(*args, **kwargs)


logging.Logger.warning_once = _warning_once


class Logger(logging.Logger):
    """Typing view of the loggers returned by `get_logger`.

    `warning_once` is installed on `logging.Logger` below, so every logger in the process has it;
    this class only makes that visible to type checkers.
    """

    def warning_once(self, *args: Any, **kwargs: Any) -> None: ...


@functools.lru_cache(None)
def _warning_once(self: logging.Logger, *args: Any, **kwargs: Any) -> None:
    self.warning(*args, **kwargs)


logging.Logger.warning_once = _warning_once


def _get_default_logging_level():
    env_level_str = os.getenv("OPTIMUM_RBLN_VERBOSE", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option OPTIMUM_RBLN_VERBOSE={env_level_str}, "
                f"has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level


def _get_library_name() -> str:
    return "optimum.rbln"


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        # set defaults based on https://github.com/pyinstaller/pyinstaller/issues/7334#issuecomment-1357447176
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler = logging.StreamHandler(sys.stderr)

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        _default_handler.setFormatter(formatter)

        library_root_logger.propagate = False


def get_logger(name: str | None = None) -> Logger:
    """
    Return a logger with the specified name.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return cast(Logger, logging.getLogger(name))
