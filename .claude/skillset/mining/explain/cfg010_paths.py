import inspect
from optimum.rbln import RBLNLlamaForCausalLMConfig as C
print("initialize_from_kwargs signature:", inspect.signature(C.initialize_from_kwargs))
cases = {
  "A from_pretrained(rbln_tensor_parallel_size=4)            ": dict(rbln_config=None, kw={"rbln_tensor_parallel_size": 4}),
  "B from_pretrained(rbln_config={'tensor_parallel_size':4}) ": dict(rbln_config={"tensor_parallel_size": 4}, kw={}),
  "C from_pretrained(rbln_config=C(), rbln_tensor_parallel_size=4)": dict(rbln_config=C(), kw={"rbln_tensor_parallel_size": 4}),
}
for name, c in cases.items():
    try:
        out = C.initialize_from_kwargs(c["rbln_config"], **c["kw"]); cfg = out[0] if isinstance(out, tuple) else out
        print(name, "-> num_devices =", cfg.num_devices, "| saved old key:", "tensor_parallel_size" in cfg._attributes_map)
    except Exception as e:
        print(name, "-> error:", type(e).__name__, str(e)[:120])
