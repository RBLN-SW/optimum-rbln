import tempfile
from optimum.rbln import RBLNLlamaForCausalLMConfig as C
from optimum.rbln.configuration_utils import RBLNModelConfig

print("[컴파일 경로: from_pretrained(export=True) -> from_model -> initialize_from_kwargs]")
cfg, _ = C.initialize_from_kwargs({"device": 0}, rbln_device=1)
print("  rbln_config={'device':0}, rbln_device=1  -> device =", cfg.device)
cfg, _ = C.initialize_from_kwargs(C(device=0), rbln_device=1)
print("  rbln_config=C(device=0), rbln_device=1   -> device =", cfg.device)

print("[로드 경로: from_pretrained(컴파일된 디렉터리) -> _from_pretrained]")
d = tempfile.mkdtemp()
C(batch_size=1).save(d)
# _from_pretrained (modeling_base.py:245-247) 이 하는 일을 그대로 재현
def load_like_model(rbln_config, **kwargs):
    if isinstance(rbln_config, dict):
        kwargs.update({f"rbln_{k}": v for k, v in rbln_config.items()})
        rbln_config = None
    cfg, _ = C.from_pretrained(d, rbln_config=rbln_config, return_unused_kwargs=True, **kwargs)
    return cfg
print("  rbln_config={'device':0}, rbln_device=1  -> device =", load_like_model({"device": 0}, rbln_device=1).device)
print("  rbln_config=C(device=0), rbln_device=1   -> device =", load_like_model(C(device=0), rbln_device=1).device)
try:
    load_like_model(None, rbln_batch_size=4)
except Exception as e:
    print("  저장된 컴파일 키 변경 rbln_batch_size=4      ->", type(e).__name__, str(e)[:90])
