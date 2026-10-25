"""Reject an unreviewed source before any candidate artifact can execute."""

import copy

import pytest

from scripts.diagnostics.chronos2_candidate_validation import CANDIDATE, OVERLAYS, require_candidate_source


def source_record():
    return {
        "source_candidate": CANDIDATE,
        "candidate_source_overlay": [{"function": name, "sha256": digest} for name, digest in OVERLAYS.items()],
    }


def test_exact_reviewed_patch_accepts_function_order_independently():
    record = source_record()
    require_candidate_source(record)
    record["candidate_source_overlay"].reverse()
    require_candidate_source(record)


@pytest.mark.parametrize("change", ["head", "missing", "extra", "digest"])
def test_changed_candidate_cannot_bypass_source_binding(change):
    record = copy.deepcopy(source_record())
    if change == "head":
        record["source_candidate"] = "0" * 40
    elif change == "missing":
        record["candidate_source_overlay"].pop()
    elif change == "extra":
        record["candidate_source_overlay"].append({"function": "unreviewed", "sha256": "0" * 64})
    else:
        record["candidate_source_overlay"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"candidate source|reviewed patch"):
        require_candidate_source(record)
