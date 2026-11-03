# Copyright 2026 Rebellions Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule

from ....utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_florence2 import *
    from .modeling_florence2 import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
