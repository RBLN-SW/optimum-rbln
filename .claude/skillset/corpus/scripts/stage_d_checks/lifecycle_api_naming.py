"""API-011: every public (exported from optimum.rbln) model / pipeline class is named RBLN<HFName> where <HFName> is a
transformers/diffusers class, and its config is RBLN<HFName>Config. usage: PYTHONPATH=<src> python3 this.py"""
import inspect, json, warnings; warnings.filterwarnings("ignore")
import transformers, diffusers, optimum.rbln as o
from optimum.rbln.modeling_base import RBLNBaseModel
from optimum.rbln.diffusers.modeling_diffusers import RBLNDiffusionMixin
from optimum.rbln.configuration_utils import RBLNModelConfig
names = sorted(n for n in o.__dir__() if not n.startswith("_") and "." not in n)
ok, bad, cfg_ok, cfg_bad = 0, [], 0, []
for n in names:
    try: c = getattr(o, n)
    except Exception: continue
    if not inspect.isclass(c) or n.startswith("RBLNAuto"): continue
    if issubclass(c, RBLNModelConfig):
        if not n.startswith("RBLN") or not n.endswith("Config"): cfg_bad.append(n)
        else: cfg_ok += 1
        continue
    if not issubclass(c, (RBLNBaseModel, RBLNDiffusionMixin)): continue
    hf = n[4:] if n.startswith("RBLN") else None
    src = inspect.getsourcefile(c).split("src/")[-1]; line = inspect.getsourcelines(c)[1]
    if hf and (hasattr(transformers, hf) or hasattr(diffusers, hf)): ok += 1
    else: bad.append({"path": "src/" + src, "line": line, "snippet": f"class {n}", "note": "no transformers/diffusers class named " + str(hf)})
print(json.dumps({"ok": ok, "bad": bad, "cfg_ok": cfg_ok, "cfg_bad": cfg_bad}, indent=1))
