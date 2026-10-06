from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest

from optimum.rbln.utils.deprecation import deprecate_kwarg, deprecate_method


# A deprecation notifies until its cutoff and refuses from it on, whatever release
# suffix the version carries; both decorators answer to the same table.
AROUND_A_CUTOFF = pytest.mark.parametrize(
    "current_version, expect_raise",
    [
        pytest.param("1.9.5", False, id="below"),
        pytest.param("1.9.99.post1", False, id="below-post"),
        pytest.param("1.10.0.dev0", True, id="dev"),
        pytest.param("1.10.0a1", True, id="alpha"),
        pytest.param("1.10.0b2", True, id="beta"),
        pytest.param("1.10.0rc1", True, id="rc"),
        pytest.param("1.10.0", True, id="final"),
        pytest.param("1.10.0.post1", True, id="post"),
        pytest.param("1.10.1", True, id="patch-above"),
    ],
)


@AROUND_A_CUTOFF
def test_deprecate_method_raises_at_or_past_cutoff(current_version, expect_raise):
    expectation = pytest.raises(ValueError, match="deprecated") if expect_raise else does_not_raise()

    with patch("optimum.rbln.utils.deprecation.__version__", current_version):

        @deprecate_method(version="1.10.0", new_method="from_pretrained")
        def stub():
            pass

    with expectation:
        stub()


@AROUND_A_CUTOFF
def test_deprecate_kwarg_raises_at_or_past_cutoff(current_version, expect_raise):
    """The decorator reads the version where it is applied, so the stub is decorated
    under the patch and called outside it."""
    with patch("optimum.rbln.utils.deprecation.__version__", current_version):

        @deprecate_kwarg(old_name="gone", version="1.10.0")
        def stub(**kwargs):
            return kwargs

    if expect_raise:
        with pytest.raises(ValueError, match="gone"):
            stub(gone=1)
    else:
        assert stub(gone=1) == {}, "below the cutoff the argument is dropped, not passed on"
    assert stub(kept=1) == {"kept": 1}
