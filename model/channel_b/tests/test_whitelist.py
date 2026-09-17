"""Unit test: the loader MUST NOT read any TRAIN_ONLY field during input
tensor construction. Runs by attempting to read each blacklisted key
via _read_input_label and asserting each raises."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from model.channel_b.data import _read_input_label, _read_arch
from model.channel_b.constants import (
    INPUT_TENSOR_LABEL_WHITELIST, TARGET_ONLY_LABEL_KEYS,
    TRAIN_ONLY_LABEL_KEYS, ARCH_ALLOWED_KEYS,
)


def test_input_and_trainonly_are_disjoint():
    assert INPUT_TENSOR_LABEL_WHITELIST.isdisjoint(TRAIN_ONLY_LABEL_KEYS), (
        "A key is in both INPUT and TRAIN_ONLY: "
        f"{INPUT_TENSOR_LABEL_WHITELIST & TRAIN_ONLY_LABEL_KEYS}"
    )


def test_input_and_target_are_disjoint():
    """Target-consumed GOLD fields MUST NOT accidentally enter the input
    whitelist. This is the guard against the class of bug where a target
    field (e.g. guide_span_in_active_noncoding) sits in the input whitelist
    and gets accidentally consumed for input construction."""
    assert INPUT_TENSOR_LABEL_WHITELIST.isdisjoint(TARGET_ONLY_LABEL_KEYS), (
        "A key is in both INPUT_TENSOR_LABEL_WHITELIST and TARGET_ONLY_LABEL_KEYS: "
        f"{INPUT_TENSOR_LABEL_WHITELIST & TARGET_ONLY_LABEL_KEYS}"
    )


def test_read_input_label_permits_whitelist():
    labels = {k: "ok" for k in INPUT_TENSOR_LABEL_WHITELIST}
    for k in INPUT_TENSOR_LABEL_WHITELIST:
        assert _read_input_label(labels, k) == "ok"


def test_read_input_label_rejects_trainonly():
    labels = {k: "leaked" for k in TRAIN_ONLY_LABEL_KEYS}
    for k in TRAIN_ONLY_LABEL_KEYS:
        with pytest.raises(AssertionError, match=r"TRAIN_ONLY_LABEL_KEYS"):
            _read_input_label(labels, k)


def test_read_input_label_rejects_targetonly():
    """Target-only GOLD fields (e.g. guide_span_in_active_noncoding) MUST
    NOT be readable via the input path — they are consumed by _build_target
    directly."""
    for k in TARGET_ONLY_LABEL_KEYS:
        with pytest.raises(AssertionError, match=r"not in INPUT_TENSOR_LABEL_WHITELIST|TRAIN_ONLY_LABEL_KEYS"):
            _read_input_label({k: "leaked"}, k)


def test_read_input_label_rejects_unknown():
    with pytest.raises(AssertionError, match=r"not in INPUT_TENSOR_LABEL_WHITELIST"):
        _read_input_label({"never_seen": "surprise"}, "never_seen")


def test_read_arch_permits_whitelist():
    arch = {k: "ok" for k in ARCH_ALLOWED_KEYS}
    for k in ARCH_ALLOWED_KEYS:
        assert _read_arch(arch, k) == "ok"


def test_read_arch_rejects_unknown():
    with pytest.raises(AssertionError, match=r"not in ARCH_ALLOWED_KEYS"):
        _read_arch({"mm_concentration": "clustered"}, "mm_concentration")


if __name__ == "__main__":
    import subprocess
    r = subprocess.run(["pytest", __file__, "-v"], cwd=str(Path(__file__).resolve().parents[3]))
    sys.exit(r.returncode)
