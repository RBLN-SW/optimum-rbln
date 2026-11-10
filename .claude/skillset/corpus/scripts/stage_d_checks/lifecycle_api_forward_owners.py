import inspect, warnings; warnings.filterwarnings("ignore")
import optimum.rbln as o
from optimum.rbln.modeling_base import RBLNBaseModel
from optimum.rbln.modeling import RBLNModel
owners = {}; generic=[]; nohf=[]
for name in dir(o):
    if not name.startswith("RBLN") or name.startswith("RBLNAuto"): continue
    try: cls = getattr(o, name)
    except Exception: continue
    if not (inspect.isclass(cls) and issubclass(cls, RBLNBaseModel)): continue
    fo = next((k for k in cls.__mro__ if "forward" in k.__dict__), None)
    try: hf = cls.get_hf_class()
    except Exception: hf=None
    if fo in (RBLNModel, RBLNBaseModel): generic.append(name); continue
    if hf is None or not hasattr(hf, "forward"): nohf.append(name); continue
    hp = inspect.signature(hf.forward).parameters
    rp = inspect.signature(cls.forward).parameters
    tt = ("token_type_ids" in hp, "token_type_ids" in rp, any(p.kind==p.VAR_KEYWORD for p in rp.values()))
    owners.setdefault(fo.__qualname__, []).append((name, tt))
print("generic-forward classes", len(generic)); print("no hf", nohf)
print("distinct custom forward owners", len(owners))
for k,v in owners.items():
    tts=[x for x in v if x[1][0] and not x[1][1]]
    if tts: print("token_type_ids in HF but not RBLN:", k, tts)
