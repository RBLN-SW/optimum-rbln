import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from optimum.rbln.transformers.models.dinov2 import modeling_dinov2 as dino

from .test_dinov2 import rbln_classes, tiny_model


def segmented(classification=False, chunk=1, dtype=torch.float32, **kwargs):
    cls, config_cls = rbln_classes(classification)
    model = tiny_model(classification).eval().to(dtype)
    config = cls.update_rbln_config(
        None,
        model,
        model.config,
        config_cls(image_size=(28, 42), batch_size=2, encoder_chunk_size=chunk, **kwargs),
    )
    return cls, model, config


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("chunk", [1, 2, 5])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_segmented_full_outputs_and_shared_compile_storage(classification, chunk, dtype, monkeypatch):
    cls, native, config = segmented(classification, chunk, dtype=dtype, create_runtimes=False)
    calls, marks = [], []
    monkeypatch.setattr(
        dino,
        "CompileContext",
        lambda **kwargs: SimpleNamespace(mark_static_address=lambda tensor, name: marks.append((tensor, name))),
    )

    def compile(cls, model, cfg, **kwargs):
        calls.append((model, cfg, kwargs))
        return model

    monkeypatch.setattr(cls, "compile", classmethod(compile))
    jobs = cls.get_compiled_model(native, config)
    assert list(jobs) == ["embedding", *[f"encoder_{i}" for i in range(0, 2, chunk)], "tail"]
    assert len(marks) == 1
    assert marks[0][1] == "activation"
    for _, cfg, kwargs in calls:
        assert kwargs["create_runtimes"] is False
        assert kwargs["compile_context"] is calls[0][2]["compile_context"]
        index = [name for name, *_ in cfg.input_info].index("activation")
        assert kwargs["example_inputs"][index] is marks[0][0]
        assert list(marks[0][0].shape) == [2, 7, 64]
        assert marks[0][0].dtype == dtype
    assert calls[0][2]["example_inputs"][0].dtype == dtype
    assert calls[-1][2]["example_inputs"][-1].dtype == torch.float32
    pixels = torch.randn(2, 3, 28, 42, dtype=dtype)
    # CPU stand-in for the in-place device cache update used by the compiler.
    monkeypatch.setattr(torch.ops.rbln_custom_ops, "rbln_cache_update", lambda target, value, *_: target.copy_(value))
    activation = marks[0][0]
    with torch.inference_mode():
        for _ in range(2):
            jobs["embedding"](pixels, activation)
            for name, job in jobs.items():
                if name.startswith("encoder_"):
                    job(activation, torch.zeros(1, dtype=torch.int16))
            assert (activation * torch.ones(1)[0]).dtype == dtype
            output = jobs["tail"](activation, torch.ones(1))
            for actual, expected in zip(output, native(pixels).to_tuple(), strict=True):
                assert actual.dtype == dtype
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("value", [0, -1, 1.2, True, "2"])
@pytest.mark.parametrize("classification", [False, True])
def test_invalid_chunk_size(value, classification):
    with pytest.raises(ValueError, match="positive integer"):
        rbln_classes(classification)[1](encoder_chunk_size=value)


@pytest.mark.parametrize("classification", [False, True])
def test_saved_config_and_legacy_default(classification, tmp_path):
    from optimum.rbln import RBLNAutoConfig

    cls, native, config = segmented(classification, create_runtimes=False)
    config.save(tmp_path)
    loaded = RBLNAutoConfig.from_pretrained(tmp_path, create_runtimes=False)
    assert loaded.encoder_chunk_size == 1
    assert [cfg.asdict() for cfg in loaded.compile_cfgs] == [cfg.asdict() for cfg in config.compile_cfgs]
    default = cls._update_rbln_config(None, native, native.config, rbln_classes(classification)[1](image_size=28))
    assert default.encoder_chunk_size is None
    assert [cfg.compiled_model_name for cfg in default.compile_cfgs] == ["compiled_model"]


class Compiled:
    def __init__(self, session, stage, events, fail=False):
        self.session, self.stage, self.events, self.fail = session, stage, events, fail

    def select_executor(self):
        return SimpleNamespace(get_context_id=lambda: self.session)

    def create_runtime(self, **kwargs):
        assert kwargs["tensor_type"] == "pt"
        assert kwargs["device"] == [0]
        self.events.append(("create", self.stage))

        def run(value):
            self.events.append((threading.current_thread().name, self.stage))
            if self.fail:
                self.fail = False
                raise RuntimeError("stage failed")
            if self.stage == "embedding":
                self.events.append(("input", value.clone()))
            if self.stage == "tail":
                return torch.ones(2, 7, 64), torch.ones(2, 64)

        return run


def chain(session, config, events, fail_stage=None):
    cls = rbln_classes(False)[0]
    compiled = [
        Compiled(session, c.compiled_model_name, events, c.compiled_model_name == fail_stage)
        for c in config.compile_cfgs
    ]
    return cls._create_runtimes(compiled, config)[0]


def test_duplicate_instances_share_lock_normalized_devices_and_weak_lifecycle():
    _, _, config = segmented(create_runtimes=False)
    first = chain("copied-artifact-session", config, [])
    config.device = [0]
    second = chain("copied-artifact-session", config, [])
    assert first.lock is second.lock
    assert chain("other-session", config, []).lock is not first.lock
    config.device = None
    assert chain("copied-artifact-session", config, []).lock is first.lock
    ref = weakref.ref(first.lock)
    del first, second
    gc.collect()
    assert ref() is None


def test_whole_chain_concurrency_and_exception_release():
    _, _, config = segmented(create_runtimes=False)
    events = []
    first, second = chain("concurrent", config, events), chain("concurrent", config, events)
    events.clear()
    entered, release, attempted = threading.Event(), threading.Event(), threading.Event()
    original = first.runtimes[0]

    def pause(value):
        original(value)
        entered.set()
        assert release.wait(5)

    first.runtimes[0] = pause
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(first, torch.zeros(2, 3, 28, 42))
        assert entered.wait(5)

        def alternate():
            attempted.set()
            return second(torch.ones(2, 3, 28, 42))

        b = pool.submit(alternate)
        assert attempted.wait(5)
        assert not b.done()
        release.set()
        a.result(timeout=5)
        b.result(timeout=5)
    stages = [stage for thread, stage in events if thread not in ("input", "create")]
    assert stages == ["embedding", "encoder_0", "encoder_1", "tail"] * 2
    failing = chain("concurrent", config, [], fail_stage="encoder_0")
    with pytest.raises(RuntimeError, match="stage failed"):
        failing(torch.zeros(2, 3, 28, 42))
    with ThreadPoolExecutor(1) as pool:
        pool.submit(second, torch.zeros(2, 3, 28, 42)).result(timeout=5)


def test_runtime_construction_waits_for_active_chain():
    _, _, config = segmented(create_runtimes=False)
    existing = chain("construction", config, [])
    events = []
    attempted = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        with existing.lock:

            def construct():
                attempted.set()
                return chain("construction", config, events)

            future = pool.submit(construct)
            assert attempted.wait(5)
            assert not events
        assert future.result(timeout=5).lock is existing.lock


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_composite_preserves_public_forward(classification, return_dict):
    cls, native, config = segmented(classification, create_runtimes=False)
    runtime = object.__new__(cls)
    runtime.config, runtime.rbln_config = native.config, config
    expected = (torch.ones(2, 5),) if classification else (torch.ones(2, 7, 64), torch.ones(2, 64))
    events = []
    runtime.model = [
        dino.Dinov2RuntimeChain(
            [
                lambda x: events.append("embedding"),
                lambda x: events.append("encoder"),
                lambda x: expected,
            ],
            threading.RLock(),
        )
    ]
    actual = runtime(pixel_values=torch.zeros(2, 3, 28, 42), return_dict=return_dict)
    assert events == ["embedding", "encoder"]
    for a, e in zip(actual.to_tuple() if return_dict else actual, expected, strict=True):
        torch.testing.assert_close(a, e)


def test_reject_mismatched_placements_and_sessions():
    cls, _, config = segmented(create_runtimes=False)
    compiled = [Compiled("session", cfg.compiled_model_name, []) for cfg in config.compile_cfgs]
    config.device_map = {cfg.compiled_model_name: 0 for cfg in config.compile_cfgs}
    config.device_map["tail"] = 1
    with pytest.raises(ValueError, match="same device"):
        cls._create_runtimes(compiled, config)
    config.device_map["tail"] = [0]
    compiled[-1].session = "different"
    with pytest.raises(ValueError, match="compile session"):
        cls._create_runtimes(compiled, config)


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("chunk", [None, 1])
def test_create_runtimes_false_uses_no_runtime(classification, chunk, monkeypatch, tmp_path):
    from optimum.rbln.utils.runtime_utils import UnavailableRuntime

    cls, native, config = segmented(classification, chunk, create_runtimes=False)
    monkeypatch.setattr(cls, "_create_runtimes", classmethod(lambda *args: pytest.fail("runtime created")))
    compiled = {cfg.compiled_model_name: object() for cfg in config.compile_cfgs}
    result = cls._from_compiled_models(compiled, config, native.config, tmp_path, "")
    assert isinstance(result.model, UnavailableRuntime)
    assert len(result.compiled_models) == len(compiled)


@pytest.mark.parametrize("batch", [2**i for i in range(11)])
def test_native_geometry_keeps_full_batch(batch):
    cls, config_cls = rbln_classes(False)
    native = tiny_model()
    config = cls._update_rbln_config(
        None,
        native,
        native.config,
        config_cls(image_size=518, batch_size=batch, encoder_chunk_size=1),
    )
    for cfg in config.compile_cfgs:
        assert next(shape for name, shape, _ in cfg.input_info if name == "activation") == [batch, 1370, 64]


def test_legacy_saved_single_artifact_delegates(monkeypatch):
    from optimum.rbln.modeling import RBLNModel

    cls, native, config = segmented(chunk=None, create_runtimes=False)
    # Artifact layout governs loading even when a runtime option is overridden.
    config.encoder_chunk_size = 1
    expected = object()
    monkeypatch.setattr(RBLNModel, "_create_runtimes", classmethod(lambda *args: expected))
    monkeypatch.setattr(RBLNModel, "get_compiled_model", classmethod(lambda *args: expected))
    assert cls._create_runtimes([object()], config) is expected
    assert cls.get_compiled_model(native, config) is expected


@pytest.mark.parametrize("device", [None, 0, [0, 1]])
def test_multi_device_default_normalization(device):
    assert dino._job_devices(device, 2) == (0, 1)
