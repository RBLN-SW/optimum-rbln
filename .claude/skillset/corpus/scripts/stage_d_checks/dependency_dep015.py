"""DEP-015: no reads of externally-deprecated APIs. AST-based (docstrings/comments ignored).
Deprecated set (pinned transformers/diffusers/torch): config.use_return_dict, diffusers.models.controlnet (non-plural) import,
torch_dtype= keyword to from_pretrained, use_auth_token= keyword, torch.cuda.amp.*. Usage: python3 dependency_dep015.py [ROOT]"""
import ast, pathlib, sys
root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
ok, bad = 0, []
for p in sorted((root / "src/optimum/rbln").rglob("*.py")):
    src = p.read_text(); lines = src.splitlines()
    for n in ast.walk(ast.parse(src)):
        hit = None
        if isinstance(n, ast.Attribute) and n.attr in ("use_return_dict", "return_dict") and isinstance(n.value, ast.Attribute) and n.value.attr == "config":
            if n.attr == "use_return_dict": hit = "config.use_return_dict deprecated; use config.return_dict"
            else: ok += 1
        elif isinstance(n, (ast.ImportFrom,)) and n.module and (n.module.startswith("diffusers.models.controlnet") and not n.module.startswith("diffusers.models.controlnets")):
            hit = "old diffusers controlnet import path"
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("diffusers.models.controlnets"):
            ok += 1
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "from_pretrained":
            for k in n.keywords:
                if k.arg in ("torch_dtype", "use_auth_token"): hit = f"{k.arg}= deprecated kwarg to HF from_pretrained"
        elif isinstance(n, ast.Attribute) and n.attr == "amp" and isinstance(n.value, ast.Attribute) and n.value.attr == "cuda":
            hit = "torch.cuda.amp deprecated"
        if hit:
            bad.append((str(p.relative_to(root)), n.lineno, lines[n.lineno - 1].strip()[:120], hit))
print("compliant", ok, "violations", len(bad))
for b in bad: print("BAD", *b)
