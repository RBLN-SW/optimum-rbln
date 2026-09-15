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

from transformers.utils import _LazyModule

from .__version__ import __version__
from .utils import check_version_compats
from .utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_utils import *
    from .diffusers import *
    from .modeling import *
    from .transformers import *
else:
    import sys

    _file = globals()["__file__"]
    import_structure = define_import_structure(_file)
    for modules in import_structure.values():
        modules.pop("__version__", None)
    sys.modules[__name__] = _LazyModule(
        __name__,
        _file,
        import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )


check_version_compats()
