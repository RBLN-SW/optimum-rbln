import pytest

from optimum.rbln.cli import parse_extra_args


@pytest.mark.parametrize("extra_args", [["--batch-size", "4"], ["--batch-size=4"]])
def test_value_given_separately_or_inline(extra_args):
    assert parse_extra_args(extra_args) == ({"batch_size": 4}, {})


def test_inline_value_keeps_everything_after_the_first_equals_sign():
    assert parse_extra_args(["--device=0,1", "--prefix=a=b"]) == ({"device": [0, 1], "prefix": "a=b"}, {})


def test_bare_name_is_a_boolean_flag():
    assert parse_extra_args(["--create-runtimes", "--batch-size", "2"]) == (
        {"create_runtimes": True, "batch_size": 2},
        {},
    )


def test_negative_number_is_a_value():
    assert parse_extra_args(["--offset", "-1"]) == ({"offset": -1}, {})


def test_dotted_name_sets_a_nested_entry():
    assert parse_extra_args(["--unet.batch-size=2", "--unet.image-size", "512"]) == (
        {"unet": {"batch_size": 2, "image_size": 512}},
        {},
    )


def test_hf_prefixed_name_becomes_a_model_kwarg():
    assert parse_extra_args(["--hf-torch-dtype=float16", "--hf-trust-remote-code"]) == (
        {},
        {"torch_dtype": "float16", "trust_remote_code": True},
    )
