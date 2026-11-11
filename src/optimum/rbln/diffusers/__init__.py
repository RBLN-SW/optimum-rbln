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

from typing import TYPE_CHECKING

from diffusers.pipelines.pipeline_utils import ALL_IMPORTABLE_CLASSES, LOADABLE_CLASSES
from transformers.utils import _LazyModule

from ..utils.import_utils import define_import_structure


LOADABLE_CLASSES["optimum.rbln"] = {
    "RBLNBaseModel": ["save_pretrained", "from_pretrained"],
    "RBLNCosmosSafetyChecker": ["save_pretrained", "from_pretrained"],
}
ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES["optimum.rbln"])


if TYPE_CHECKING:
    from .configurations import *
    from .modeling_diffusers import *
    from .models import *
    from .pipelines import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
