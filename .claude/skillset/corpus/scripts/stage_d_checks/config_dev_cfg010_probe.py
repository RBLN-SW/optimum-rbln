"""CFG-010 probe: are deprecated config keys remapped on ctor, setattr and rbln_-prefix paths? (dev)"""
import warnings; warnings.filterwarnings("ignore")
import torch
from optimum.rbln import RBLNResNetForImageClassificationConfig as C, RBLNLlamaForCausalLMConfig as L
def run(label, fn):
    try: print(label, "->", fn())
    except Exception as e: print(label, "-> RAISES", type(e).__name__, str(e)[:90])
run("ctor tensor_parallel_size=2 -> num_devices", lambda: C(tensor_parallel_size=2).num_devices)
def sa():
    c = C(); c.tensor_parallel_size = 2
    return (c.num_devices, "tensor_parallel_size" in c._prepare_for_serialization())
run("setattr tensor_parallel_size=2 -> (num_devices, serialized_as_unknown)", sa)
def td():
    c = C(); c.torch_dtype = torch.bfloat16; return c.dtype
run("setattr torch_dtype -> dtype", td)
run("instance + rbln_tensor_parallel_size=2 -> num_devices", lambda: C.initialize_from_kwargs(C(), rbln_tensor_parallel_size=2)[0].num_devices)
run("dict + rbln_tensor_parallel_size=2 -> num_devices", lambda: C.initialize_from_kwargs({}, rbln_tensor_parallel_size=2)[0].num_devices)
run("ctor kvcache_metas=[] -> cache_metas", lambda: L(kvcache_metas=[]).cache_metas)
def km():
    c = L(); c.kvcache_metas = []; return ("kvcache_metas" in c._prepare_for_serialization(), hasattr(c, "cache_metas"))
run("setattr kvcache_metas -> (serialized_as_unknown, has cache_metas)", km)
