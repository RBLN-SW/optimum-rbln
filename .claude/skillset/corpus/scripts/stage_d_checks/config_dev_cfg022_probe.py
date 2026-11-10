"""CFG-022 probe: what happens when rbln_config and rbln_* kwargs are both given (dev)."""
import sys, tempfile, warnings
warnings.filterwarnings("ignore")
from optimum.rbln import RBLNResNetForImageClassificationConfig as C
from optimum.rbln.configuration_utils import RBLNCompileConfig

def run(label, fn):
    try:
        print(label, "->", fn())
    except Exception as e:
        print(label, "-> RAISES", type(e).__name__, str(e)[:100])

# compile path (RBLNModel.from_pretrained(export=True) -> prepare_rbln_config -> initialize_from_kwargs)
run("init dict{batch_size:4}+rbln_batch_size=16", lambda: C.initialize_from_kwargs({"batch_size": 4}, rbln_batch_size=16)[0].batch_size)
run("init instance(batch_size=4)+rbln_batch_size=16", lambda: C.initialize_from_kwargs(C(batch_size=4), rbln_batch_size=16)[0].batch_size)
run("init dict{device:0}+rbln_device=1", lambda: C.initialize_from_kwargs({"device": 0}, rbln_device=1)[0].device)

# load path: RBLNModelConfig.from_pretrained(path, rbln_config=dict, rbln_*)
d = tempfile.mkdtemp()
c = C(batch_size=4, image_size=224)
c.set_compile_cfgs([RBLNCompileConfig(input_info=[("pixel_values", (4, 3, 224, 224), "float32")])])
c.freeze(); c.save(d)
run("load dict{device:0}+rbln_device=1", lambda: C.from_pretrained(d, rbln_config={"device": 0}, rbln_device=1).device)
run("load dict{batch_size:4}+rbln_batch_size=4 (same as saved)", lambda: C.from_pretrained(d, rbln_config={"batch_size": 4}, rbln_batch_size=4).batch_size)
run("load rbln_batch_size=8 (differs from saved)", lambda: C.from_pretrained(d, rbln_batch_size=8).batch_size)
# modeling_base load path: dict converted to rbln_ kwargs with kwargs.update (dict overwrites prefix), emulate exactly
def mb_emulate():
    kwargs = {"rbln_device": 1}
    rbln_config = {"device": 0}
    kwargs.update({f"rbln_{k}": v for k, v in rbln_config.items()})  # modeling_base.py:246-247
    return C.from_pretrained(d, rbln_config=None, **kwargs).device
run("RBLNBaseModel load emulation dict{device:0}+rbln_device=1", mb_emulate)
