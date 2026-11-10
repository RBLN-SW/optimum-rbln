"""CFG-018: every RBLNModelConfig subclass that redefines subclass_non_save_attributes must include
its parent's effective list; Callable-typed __init__ params must be registered (runtime-only)."""
import inspect, sys, warnings, typing
warnings.filterwarnings("ignore")
import optimum.rbln as o
from optimum.rbln.configuration_utils import RBLNModelConfig
def all_subclasses(c):
    out = set()
    for s in c.__subclasses__():
        out.add(s); out |= all_subclasses(s)
    return out
for name in dir(o):  # force lazy imports
    if name.endswith("Config"):
        try: getattr(o, name)
        except Exception: pass
classes = sorted(all_subclasses(RBLNModelConfig), key=lambda c: c.__name__)
print("n_config_classes", len(classes))
redef = ok = 0
for c in classes:
    if "subclass_non_save_attributes" in c.__dict__:
        parent_attr = set(super(c, c).subclass_non_save_attributes) if hasattr(super(c, c), "subclass_non_save_attributes") else set()
        # effective parent = nearest base in MRO
        parent_attr = set(c.__mro__[1].subclass_non_save_attributes)
        redef += 1
        missing = parent_attr - set(c.__dict__["subclass_non_save_attributes"])
        if missing: print("SHADOW", c.__module__, c.__name__, "missing", sorted(missing))
        else: ok += 1
print("redefining", redef, "superset_ok", ok)
# Callable-typed params
for c in classes:
    if "__init__" not in c.__dict__: continue
    for p in inspect.signature(c.__init__).parameters.values():
        a = str(p.annotation)
        if "Callable" in a:
            reg = p.name in c.subclass_non_save_attributes or p.name in c.non_save_attributes
            print("CALLABLE", c.__name__, p.name, a, "registered" if reg else "NOT_REGISTERED")
