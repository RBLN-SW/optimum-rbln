import logging, json, warnings, sys
logging.basicConfig(level=logging.WARNING, format="  [log] %(message)s")
warnings.simplefilter("always")
from optimum.rbln import RBLNLlamaForCausalLMConfig as C

print("1) 생성자 인자로 옛 이름을 넘김")
c1 = C(tensor_parallel_size=4)
print("   num_devices =", c1.num_devices, "| 저장 키에 tensor_parallel_size?", "tensor_parallel_size" in c1._attributes_map)

print("2) 만든 뒤 속성으로 옛 이름을 대입")
c2 = C()
c2.tensor_parallel_size = 4
print("   num_devices =", c2.num_devices, "| 저장 키에 tensor_parallel_size?", "tensor_parallel_size" in c2._attributes_map)

print("3) from_pretrained 경로처럼 rbln_ 접두사로 넘김")
c3 = C.initialize_from_kwargs(C(), rbln_tensor_parallel_size=4)[0] if hasattr(C, "initialize_from_kwargs") else None
print("   num_devices =", getattr(c3, "num_devices", "n/a"))
