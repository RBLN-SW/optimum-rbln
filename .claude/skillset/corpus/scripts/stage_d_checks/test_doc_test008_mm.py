# For each test class in tests/test_llm.py, resolve RBLN_CLASS and IS_MULTIMODAL through inheritance.
import ast, pathlib, re, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
src = (root / "tests/test_llm.py").read_text()
tree = ast.parse(src)
classes = {}
def collect(body, prefix=""):
    for n in body:
        if isinstance(n, ast.ClassDef):
            attrs = {}
            for s in n.body:
                if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name):
                    attrs[s.targets[0].id] = s.value
            bases = [ast.unparse(b) for b in n.bases]
            classes[prefix + n.name] = (n, bases, attrs)
            collect(n.body, prefix + n.name + ".")
collect(tree.body)
def resolve(name, key):
    if name not in classes: return None
    n, bases, attrs = classes[name]
    if key in attrs: return ast.unparse(attrs[key])
    for b in bases:
        r = resolve(b, key)
        if r is not None: return r
    return None
VLM = re.compile(r"Llava|Idefics|Qwen\d?(_\d)?_?VL|Qwen3VL|Gemma3ForConditionalGeneration|Blip2|PaliGemma|Gemma4ForConditionalGeneration|Exaone4_5_For|Qwen3_5ForConditional")
for name, (n, bases, attrs) in classes.items():
    rc = resolve(name, "RBLN_CLASS")
    if not rc or "." in name: continue
    mm = resolve(name, "IS_MULTIMODAL")
    is_vlm = bool(VLM.search(rc))
    flag = mm == "True"
    status = "OK" if is_vlm == flag else "MISMATCH"
    print(f"{status}\t{n.lineno}\t{name}\t{rc}\tvlm={is_vlm}\tIS_MULTIMODAL={mm}")
