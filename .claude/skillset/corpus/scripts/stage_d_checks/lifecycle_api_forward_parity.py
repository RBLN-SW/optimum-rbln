"""Compare RBLN*.forward signatures with the HF class forward they replace.
usage: PYTHONPATH=<src> python3 lifecycle_api_forward_parity.py
For every class exported from optimum.rbln that subclasses RBLNBaseModel and defines its own forward (somewhere in an
RBLN class below RBLNModel), resolve the HF class via cls.get_hf_class() and report:
  ORDER   - parameters present in both appear in a different relative order
  EXTRA   - RBLN forward has a named parameter that HF forward lacks (and HF has no **kwargs)
  MISSING - HF forward has a data input (from DATA set) that RBLN forward lacks and RBLN has no **kwargs
"""
import inspect, warnings, json, sys
warnings.filterwarnings("ignore")
import optimum.rbln as o
from optimum.rbln.modeling_base import RBLNBaseModel
from optimum.rbln.modeling import RBLNModel
DATA = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "inputs_embeds", "pixel_values",
        "input_values", "input_features", "decoder_input_ids", "decoder_attention_mask", "encoder_outputs",
        "pixel_mask", "image_grid_thw", "video_grid_thw", "cache_position", "sample", "timestep",
        "encoder_hidden_states", "labels"}
SKIP_ALWAYS = {"self", "return_dict", "output_attentions", "output_hidden_states", "use_cache", "past_key_values",
               "labels", "kwargs", "args"}
def params(f):
    sig = inspect.signature(f)
    names, varkw = [], False
    for p in sig.parameters.values():
        if p.kind is p.VAR_KEYWORD: varkw = True
        elif p.kind is p.VAR_POSITIONAL: pass
        elif p.name != "self": names.append(p.name)
    return names, varkw
rows, ok, checked = [], 0, 0
for name in sorted(o._import_structure_names if hasattr(o, "_import_structure_names") else dir(o)):
    if not name.startswith("RBLN"): continue
    try: cls = getattr(o, name)
    except Exception as e: continue
    if not (inspect.isclass(cls) and issubclass(cls, RBLNBaseModel)) or name.startswith("RBLNAuto"): continue
    fwd_owner = next((k for k in cls.__mro__ if "forward" in k.__dict__), None)
    if fwd_owner in (RBLNModel, RBLNBaseModel, None): continue
    try: hf = cls.get_hf_class()
    except Exception: hf = None
    if hf is None or not hasattr(hf, "forward"): continue
    checked += 1
    r, rk = params(cls.forward); h, hk = params(hf.forward)
    common = [p for p in r if p in h]
    problems = []
    if common != [p for p in h if p in common]: problems.append(("ORDER", f"rbln={common} hf={[p for p in h if p in common]}"))
    extra = [p for p in r if p not in h and p not in SKIP_ALWAYS]
    if extra and not hk: problems.append(("EXTRA", ",".join(extra)))
    missing = [p for p in h if p in DATA and p not in r and p != "labels"]
    if missing and not rk: problems.append(("MISSING", ",".join(missing)))
    src = inspect.getsourcefile(fwd_owner); line = inspect.getsourcelines(fwd_owner.forward)[1]
    if problems:
        rows.append({"cls": name, "fwd_owner": fwd_owner.__name__, "hf": hf.__name__, "path": src.split("src/")[-1], "line": line, "problems": problems})
    else: ok += 1
print(json.dumps({"checked": checked, "ok": ok, "violations": rows}, indent=1))
