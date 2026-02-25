"""Safe wrapper around griffe_pydantic that handles two bugs in v1.3.1.

Bug 1: Static analysis labels unannotated class attributes (e.g. `support_paged_attention = True`)
        as pydantic-field with annotation=None, causing template rendering to crash with
        'NoneType' object is not iterable in pydantic_model.html.jinja.

Bug 2: _model_fields/_model_validators iterate cls.all_members without checking for None labels
        on inherited Alias objects.

Fix: Patch static._process_attribute to skip unannotated attributes, and patch
     common._model_fields/_model_validators to guard against None labels.
"""

from griffe_pydantic._internal import common, static

# --- Fix Bug 1: Skip unannotated attributes in static analysis ---
_orig_process_attribute = static._process_attribute


def _safe_process_attribute(attr, cls, *, processed):
    # Unannotated attributes (annotation=None) cannot be Pydantic fields.
    # griffe_pydantic incorrectly labels them as pydantic-field, and the
    # template crashes when trying to render the None annotation.
    if attr.annotation is None:
        return
    return _orig_process_attribute(attr, cls, processed=processed)


static._process_attribute = _safe_process_attribute

# --- Fix Bug 2: Guard against None labels in member iteration ---
_orig_model_fields = common._model_fields


def _safe_model_fields(cls):
    return {
        name: attr
        for name, attr in cls.all_members.items()
        if getattr(attr, "labels", None) is not None and "pydantic-field" in attr.labels
    }


common._model_fields = _safe_model_fields

_orig_model_validators = common._model_validators


def _safe_model_validators(cls):
    return {
        name: func
        for name, func in cls.all_members.items()
        if getattr(func, "labels", None) is not None and "pydantic-validator" in func.labels
    }


common._model_validators = _safe_model_validators

# Re-export the (now-safe) PydanticExtension
from griffe_pydantic import PydanticExtension  # noqa: E402
