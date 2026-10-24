"""Mock resource ownership and tensor lifetime controls; no NPU is used."""

from contextlib import contextmanager

import torch

from scripts.diagnostics.chronos2_sequential_runtime import SequentialRuntimeChain


def test_stages_unload_before_next_load_and_outputs_survive_runtime_release(tmp_path):
    active = []
    transitions = []
    events = []
    buffers = []

    @contextmanager
    def loader(path, device):
        assert not active
        assert device == 0
        active.append(path.stem)
        transitions.append(("load", path.stem))

        def runtime(*inputs):
            if path.stem == "embedding":
                output = (inputs[0] + 1, inputs[1].clone(), inputs[2].clone())
            elif path.stem == "encoder_0":
                assert inputs[2].shape == (8, 1, 512, 512)
                output = inputs[0] + 2
            else:
                output = (inputs[0] + 3, inputs[0] + 4)
            buffers.extend(output if isinstance(output, tuple) else [output])
            return output

        try:
            yield runtime
        finally:
            # Simulate runtime storage becoming unusable as soon as it unloads.
            for value in buffers:
                value.fill_(float("nan"))
            buffers.clear()
            transitions.append(("unload", active.pop()))

    stages = [
        {"name": name, "file": f"{name}.rbln", "alloc_per_node": [1234]}
        for name in ("embedding", "encoder_0", "forecast")
    ]
    chain = SequentialRuntimeChain(tmp_path, stages, 0, events.append, loader=loader)
    inputs = (torch.zeros(512, 8, 2), torch.ones(512, 1, 1, 8), torch.zeros(8, 1, 512, 512))
    result = chain(*inputs)
    torch.testing.assert_close(result[0], torch.full_like(inputs[0], 6))
    torch.testing.assert_close(result[1], torch.full_like(inputs[0], 7))
    repeated = chain(*inputs)
    for actual, expected in zip(repeated, result, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not active
    assert transitions == [
        (phase, name)
        for _ in range(2)
        for name in ("embedding", "encoder_0", "forecast")
        for phase in ("load", "unload")
    ]
    assert len([event for event in events if event["phase"] == "cold_load"]) == 6
    assert all(event["alloc_per_node"] == [1234] for event in events if event["phase"] == "cold_load")
    assert len([event for event in events if event["phase"] == "request"]) == 2
