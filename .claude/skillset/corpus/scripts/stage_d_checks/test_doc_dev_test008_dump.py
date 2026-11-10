"""TEST-008: every public user-facing RBLN model/pipeline class is registered (RBLN_CLASS = X) in tests/test_*.py.
Exports: `_import_structure` in src/optimum/rbln/__init__.py (main) or every module `__all__` (dev layout).
Model class = RBLN*, not *Config, not RBLNAuto*, not RBLNBaseModel/RBLNModel.
COMPONENT = submodule / mixin / generic base, exercised through its parent (excluded from the denominator).
COVERED_VIA = sub-pipelines constructed by a tested combined pipeline.
Usage: python3 test_doc_test008.py <repo_root>"""
import ast, re, sys, pathlib, json
root = pathlib.Path(sys.argv[1])
names = set()
tree = ast.parse((root / "src/optimum/rbln/__init__.py").read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_import_structure" for t in node.targets):
        for k, v in zip(node.value.keys, node.value.values):
            if k.value in ("transformers", "diffusers"):
                names |= {e.value for e in v.elts}
if not names:
    for p in list((root / "src/optimum/rbln/transformers").rglob("*.py")) + list((root / "src/optimum/rbln/diffusers").rglob("*.py")):
        for node in ast.parse(p.read_text()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets):
                names |= {e.value for e in node.value.elts if isinstance(e, ast.Constant)}
models = sorted(n for n in names if n.startswith("RBLN") and not n.endswith("Config")
                and not n.startswith("RBLNAuto") and n not in ("RBLNBaseModel", "RBLNModel"))
COMPONENT = re.compile(r"Vision|QFormer|TextModel|Encoder$|Decoder$|Backbone|TransformerPretrainedModel|^RBLNDecoderOnly"
                       r"|UNet|ControlNetModel$|VQModel|PriorTransformer|Transformer[23]DModel|SafetyChecker|Mixin|AutoencoderKL|AudioEncoder")
COVERED_VIA = {"RBLNKandinskyV22Pipeline": "RBLNKandinskyV22CombinedPipeline",
               "RBLNKandinskyV22PriorPipeline": "RBLNKandinskyV22CombinedPipeline",
               "RBLNKandinskyV22Img2ImgPipeline": "RBLNKandinskyV22Img2ImgCombinedPipeline"}
code = re.sub(r"#.*", "", "\n".join(p.read_text() for p in (root / "tests").glob("test_*.py")))
registered = set(re.findall(r"RBLN_CLASS\s*=\s*(RBLN\w+)", code))
out = {"registered": [], "covered_via_parent": [], "component_untested": [], "missing": []}
for n in models:
    if n in registered: out["registered"].append(n)
    elif COVERED_VIA.get(n) in registered: out["covered_via_parent"].append(n)
    elif COMPONENT.search(n): out["component_untested"].append(n)
    else: out["missing"].append(n)
print(json.dumps({k: len(v) for k, v in out.items()}), "exported model classes:", len(models))
print("REG:", " ".join(out["registered"])); print("COMP:", " ".join(out["component_untested"])); print("VIA:", out["covered_via_parent"])
